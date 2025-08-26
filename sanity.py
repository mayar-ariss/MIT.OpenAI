# -*- coding: utf-8 -*-
"""
Network suitability checker for fire-incident optimization (OFH / hydrant selection)
- Works with EPANET INP files via WNTR
- Accepts either: (a) path to .inp, or (b) an existing WaterNetworkModel
- Returns a structured report + pretty print

Key checks:
- Flow units, demand model (DDA vs PDD)
- Node coordinates (for distance-to-fire objective)
- Elevations present
- Demand hygiene (negative/zero), patterns
- Pumps & curves
- Valves inventory (PRV/TCV/etc.)
- Hydrant representation (emitters or HYD/FH names)
- Connectivity (isolated nodes/links)
"""

from typing import Union, Dict, Any, List, Tuple
import math
import json

try:
    import wntr
    from wntr.network.model import WaterNetworkModel
except ImportError as e:
    raise SystemExit("This script requires the 'wntr' package. Install with: pip install wntr") from e


# ----------------------------
# Helpers
# ----------------------------

def _flow_units(wn: WaterNetworkModel) -> str:
    try:
        fu = wn.options.hydraulic.flow_units
        # fu may be an Enum or string-like
        return str(getattr(fu, "name", fu)).upper()
    except Exception:
        return "UNKNOWN"

def _demand_model(wn: WaterNetworkModel) -> str:
    try:
        dm = wn.options.hydraulic.demand_model
        return str(getattr(dm, "name", dm)).upper()
    except Exception:
        return "UNKNOWN"

def _has_coords(wn: WaterNetworkModel) -> Tuple[int, int]:
    with_xy = 0
    jlist = getattr(wn, "junction_name_list", [])
    for j in jlist:
        try:
            n = wn.get_node(j)
            if n.coordinates and all(isinstance(c, (int, float)) for c in n.coordinates):
                with_xy += 1
        except Exception:
            pass
    return with_xy, len(jlist)

def _has_pressures(wn: WaterNetworkModel) -> bool:
    for j in getattr(wn, "junction_name_list", []):
        node = wn.get_node(j)
        if node.elevation is None or not math.isfinite(node.elevation):
            return False
    return True

def _controls_summary(wn: WaterNetworkModel) -> Dict[str, int]:
    # WNTR returns a list-like; be defensive
    out = {"simple_controls": 0, "rule_controls": 0, "all_controls": 0}
    try:
        ctrls = wn.controls()
        out["all_controls"] = len(ctrls)
        for c in ctrls:
            ctyp = getattr(getattr(c, "control_type", None), "name", "").upper()
            if ctyp == "RULE":
                out["rule_controls"] += 1
            elif ctyp == "SIMPLE":
                out["simple_controls"] += 1
    except Exception:
        pass
    return out

def _pump_curve_issues(wn: WaterNetworkModel) -> Dict[str, List[str]]:
    missing, bad = [], []
    for pid in getattr(wn, "pump_name_list", []):
        try:
            p = wn.get_link(pid)
            ptype = str(getattr(getattr(p, "pump_type", None), "name", "UNKNOWN")).upper()
            if ptype in ("UNKNOWN", "", None):
                missing.append(pid); continue
            if ptype == "POWER":
                # no curve required
                continue
            cname = getattr(p, "pump_curve_name", None)
            if cname is None or cname not in getattr(wn, "curve_name_list", []):
                missing.append(pid); continue
            curve = wn.get_curve(cname)
            pts = getattr(curve, "point_list", None) or []
            if not pts or any(len(pt) < 2 for pt in pts):
                bad.append(pid)
        except Exception:
            missing.append(pid)
    return {"pumps_missing_curve": missing, "pumps_bad_curve": bad}

def _valve_inventory(wn: WaterNetworkModel) -> Dict[str, Any]:
    valve_ids = list(getattr(wn, "valve_name_list", []))
    vt_counts, tcv_like_prv = {}, []
    for vid in valve_ids:
        try:
            v = wn.get_link(vid)
            vtype = str(getattr(v, "valve_type", "UNKNOWN")).upper()
            vt_counts[vtype] = vt_counts.get(vtype, 0) + 1
            if vtype == "TCV":
                tcv_like_prv.append(vid)
        except Exception:
            vt_counts["UNKNOWN"] = vt_counts.get("UNKNOWN", 0) + 1
    return {"by_type": vt_counts, "tcv_candidates_for_prv": tcv_like_prv, "all_valves": valve_ids}

def _link_status_summary(wn: WaterNetworkModel) -> Dict[str, int]:
    open_cnt = closed_cnt = cv_cnt = 0
    for lid in getattr(wn, "link_name_list", []):
        try:
            link = wn.get_link(lid)
            st = str(getattr(link, "initial_status", "OPEN")).upper()
            if "CLOSED" in st:
                closed_cnt += 1
            elif "CV" in st or "CHECK" in st:
                cv_cnt += 1
            else:
                open_cnt += 1
        except Exception:
            open_cnt += 1
    return {"open": open_cnt, "closed": closed_cnt, "check_valve": cv_cnt}

def _demands_summary(wn: WaterNetworkModel) -> Dict[str, Any]:
    total = 0.0
    negative_nodes, zero_nodes = [], []
    has_patterns = 0
    for j in getattr(wn, "junction_name_list", []):
        try:
            node = wn.get_node(j)
            dlist = getattr(node, "demand_timeseries_list", None) or []
            base_sum = 0.0
            for dd in dlist:
                base_sum += float(getattr(dd, "base_value", 0.0) or 0.0)
                if getattr(dd, "pattern_name", None):
                    has_patterns += 1
            total += base_sum
            if base_sum < 0:
                negative_nodes.append(j)
            if base_sum == 0:
                zero_nodes.append(j)
        except Exception:
            pass
    return {
        "total_base_demand": total,
        "junctions_with_patterns": has_patterns,
        "negative_demand_nodes": negative_nodes,
        "zero_demand_nodes": zero_nodes,
    }

def _detect_hydrants(wn: WaterNetworkModel) -> Dict[str, Any]:
    """
    Hydrants are not native elements. Common encodings:
      - Junctions with emitter_coefficient > 0
      - Name/tag heuristics: HYD, HYDRANT, FH
    """
    hydrant_candidates, emitter_based, tag_based = set(), set(), set()
    for j in getattr(wn, "junction_name_list", []):
        node = wn.get_node(j)
        name_upper = node.name.upper()
        # emitter-based
        if getattr(node, "emitter_coefficient", 0) and node.emitter_coefficient > 0:
            emitter_based.add(node.name)
            hydrant_candidates.add(node.name)
            continue
        # name/tag heuristic
        if any(k in name_upper for k in ["HYD", "HYDRANT", "FH"]):
            tag_based.add(node.name)
            hydrant_candidates.add(node.name)
    return {
        "count_candidates": len(hydrant_candidates),
        "by_emitter": sorted(emitter_based),
        "by_name_or_tag": sorted(tag_based),
        "unique_list": sorted(hydrant_candidates),
    }

def _disconnected_elements(wn: WaterNetworkModel) -> Dict[str, List[str]]:
    # WNTR: get networkx graph; prefer to_graph (not deprecated get_graph)
    try:
        G = wn.to_graph()
    except Exception:
        G = wn.get_graph()  # fallback for very old WNTR
    isolated_nodes = [n for n, deg in G.degree() if deg == 0]
    iso_links = []
    for lid in getattr(wn, "link_name_list", []):
        link = wn.get_link(lid)
        if link.start_node_name in isolated_nodes or link.end_node_name in isolated_nodes:
            iso_links.append(lid)
    return {"isolated_nodes": isolated_nodes, "links_touching_isolated_nodes": iso_links}


# ----------------------------
# Public API
# ----------------------------

def check_network_suitability(wn_or_path: Union[str, "WaterNetworkModel"]) -> Dict[str, Any]:
    """
    Main entry point.
    Returns a dict with flags, issues, and an overall suitability verdict for fire optimization.
    """
    if isinstance(wn_or_path, WaterNetworkModel):
        wn = wn_or_path
        src = getattr(wn, "name", "WaterNetworkModel")
    else:
        wn = wntr.network.WaterNetworkModel(wn_or_path)
        src = str(wn_or_path)

    report: Dict[str, Any] = {"source": src, "ok": True, "errors": [], "warnings": [], "info": {}}

    # basic inventory via name lists (robust across WNTR builds)
    report["info"]["counts"] = {
        "nodes": len(getattr(wn, "node_name_list", [])),
        "junctions": len(getattr(wn, "junction_name_list", [])),
        "links": len(getattr(wn, "link_name_list", [])),
        "pipes": len(getattr(wn, "pipe_name_list", [])),
        "valves": len(getattr(wn, "valve_name_list", [])),
        "pumps": len(getattr(wn, "pump_name_list", [])),
        "tanks": len(getattr(wn, "tank_name_list", [])),
        "reservoirs": len(getattr(wn, "reservoir_name_list", [])),
    }

    # hydraulics
    flow_units = _flow_units(wn)
    demand_model = _demand_model(wn)
    report["info"]["hydraulics"] = {"flow_units": flow_units, "demand_model": demand_model}

    # coordinates
    with_xy, total_j = _has_coords(wn)
    if with_xy == 0:
        report["warnings"].append("No node coordinates found; distance-to-fire objective will be unavailable.")
    elif with_xy < total_j:
        report["warnings"].append(f"Only {with_xy}/{total_j} junctions have coordinates.")

    # elevations
    if not _has_pressures(wn):
        report["errors"].append("Some junctions lack elevations; pressure-driven analysis will fail.")
        report["ok"] = False

    # demand summary
    dem = _demands_summary(wn)
    report["info"]["demands"] = dem
    if dem["negative_demand_nodes"]:
        report["errors"].append(f"Negative base demands at nodes: {dem['negative_demand_nodes'][:10]} ...")
        report["ok"] = False

    # controls
    report["info"]["controls"] = _controls_summary(wn)

    # pumps & curves
    pump_issues = _pump_curve_issues(wn)
    report["info"]["pump_issues"] = pump_issues
    if pump_issues["pumps_missing_curve"] or pump_issues["pumps_bad_curve"]:
        report["warnings"].append("Some pumps are missing/invalid curves; fire scenarios may be inaccurate.")

    # valves
    v_inv = _valve_inventory(wn)
    report["info"]["valves"] = v_inv
    n_prv = v_inv["by_type"].get("PRV", 0)
    n_tcv = v_inv["by_type"].get("TCV", 0)
    if n_prv == 0 and n_tcv > 0:
        report["warnings"].append(
            "No PRVs found; TCVs present. If TCVs mimic PRVs, convert/mark them explicitly to use pressure setpoints."
        )
    elif n_prv == 0 and n_tcv == 0:
        report["warnings"].append("No PRVs/TCVs found; valve‑control optimization will be limited.")

    # demand model suitability
    if demand_model != "PDD":
        report["warnings"].append(
            "Demand model is DDA. For firefighting studies, enable PDD to model supply deficit realistically."
        )

    # hydrants
    hydr = _detect_hydrants(wn)
    report["info"]["hydrants"] = hydr
    if hydr["count_candidates"] == 0:
        report["warnings"].append(
            "No hydrant representation detected (no emitters and no HYD/FH names). "
            "Represent hydrants as junctions with emitter_coefficient or provide a hydrant node list."
        )

    # connectivity
    conn = _disconnected_elements(wn)
    report["info"]["connectivity"] = conn
    if conn["isolated_nodes"]:
        report["errors"].append(f"Isolated nodes detected: {conn['isolated_nodes'][:10]} ...")
        report["ok"] = False

    # link status
    report["info"]["link_status"] = _link_status_summary(wn)

    # minimal must-haves for our optimization pipeline (as warnings)
    must = []
    if n_prv == 0 and n_tcv == 0: must.append("No controllable valves (PRV/TCV).")
    if hydr["count_candidates"] == 0: must.append("No hydrant candidates.")
    if with_xy == 0: must.append("No node coordinates (cannot compute distance-to-fire).")
    report["warnings"].extend(must)

    return report


def pretty_print_report(report: Dict[str, Any]) -> None:
    print("=" * 72)
    print("NETWORK SUITABILITY REPORT")
    print("=" * 72)
    print(f"Source: {report.get('source')}")
    counts = report["info"].get("counts", {})
    print(f"Counts: nodes={counts.get('nodes',0)}, junctions={counts.get('junctions',0)}, "
          f"links={counts.get('links',0)}, valves={counts.get('valves',0)}, pumps={counts.get('pumps',0)}")

    hyd = report["info"].get("hydraulics", {})
    print(f"Hydraulics: flow_units={hyd.get('flow_units')}  demand_model={hyd.get('demand_model')}")
    print()

    if report["errors"]:
        print("ERRORS:")
        for e in report["errors"]:
            print(f"  - {e}")
    else:
        print("ERRORS: none")

    if report["warnings"]:
        print("\nWARNINGS / ACTIONS:")
        for w in report["warnings"]:
            print(f"  - {w}")
    else:
        print("\nWARNINGS: none")

    print("\nDETAILS:")
    v_inv = report["info"].get("valves", {})
    print(f"  Valves by type: {v_inv.get('by_type', {})}")
    hydr = report["info"].get("hydrants", {})
    print(f"  Hydrant candidates: {hydr.get('count_candidates',0)} "
          f"(emitters={len(hydr.get('by_emitter',[]))}, tag/name={len(hydr.get('by_name_or_tag',[]))})")
    dem = report["info"].get("demands", {})
    print(f"  Total base demand: {dem.get('total_base_demand',0):.4f} "
          f"({dem.get('junctions_with_patterns',0)} junctions with patterns)")
    ctrl = report["info"].get("controls", {})
    print(f"  Controls: simple={ctrl.get('simple_controls',0)}, rule={ctrl.get('rule_controls',0)}, "
          f"total={ctrl.get('all_controls',0)}")
    pump = report["info"].get("pump_issues", {})
    if pump.get("pumps_missing_curve") or pump.get("pumps_bad_curve"):
        print(f"  Pumps missing curves: {pump.get('pumps_missing_curve', [])}")
        print(f"  Pumps with bad curves: {pump.get('pumps_bad_curve', [])}")
    conn = report["info"].get("connectivity", {})
    if conn.get("isolated_nodes"):
        print(f"  Isolated nodes: {conn['isolated_nodes'][:10]} ...")
    ls = report["info"].get("link_status", {})
    print(f"  Link status summary: {ls}")

    print("\nOVERALL:", "OK" if report.get("ok") and not report["errors"] else "NOT READY")
    print("=" * 72)
