# fireflow_checker.py
# Solid OFH/Fire-flow checker for EPANET INP files
# Requires: wntr, pandas, networkx (wntr brings nx)
from __future__ import annotations

import re, math, sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import wntr
import networkx as nx

__version__ = "0.2.1"

SEV_ERR = "ERROR"
SEV_WRN = "WARN"
SEV_INF = "INFO"


# --------------------------- raw INP helpers ---------------------------

def _read_inp_text(inp_path: str | Path) -> str:
    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _parse_simple_section(text: str, name: str) -> List[str]:
    """Return raw non-empty, non-comment lines inside [NAME] section."""
    pat = re.compile(rf"^\[{re.escape(name)}\]\s*$", re.I | re.M)
    m = pat.search(text)
    if not m:
        return []
    start = m.end()
    m2 = re.search(r"^\[[^\]]+\]\s*$", text[start:], re.M)
    block = text[start:] if not m2 else text[start:start+m2.start()]
    lines = []
    for ln in block.splitlines():
        s = ln.strip()
        if not s or s.startswith(";"):
            continue
        lines.append(s)
    return lines

def _parse_kv(lines: List[str]) -> Dict[str, str]:
    """Crudely parse 'key value...' lines, handling some two-word keys."""
    out = {}
    for ln in lines:
        parts = re.split(r"\s+", ln, maxsplit=1)
        if not parts:
            continue
        if len(parts) == 1:
            k, v = parts[0], ""
        else:
            k, v = parts[0], parts[1]
        if k.endswith(":"):
            k = k[:-1]
        if k.lower() in {"demand", "start"} and (" " in v):
            # Merge two-word keys like "Demand Model", "Start ClockTime"
            k2, v2 = v.split(None, 1)
            k = f"{k} {k2}"
            v = v2
        out[k.strip()] = v.strip()
    return out

def _parse_options(text: str) -> Dict[str, str]:
    return _parse_kv(_parse_simple_section(text, "OPTIONS"))

def _parse_times(text: str) -> Dict[str, str]:
    return _parse_kv(_parse_simple_section(text, "TIMES"))

def _parse_backdrop(text: str) -> Dict[str, Any]:
    lines = _parse_simple_section(text, "BACKDROP")
    kv: Dict[str, Any] = {}
    for ln in lines:
        up = ln.upper()
        if up.startswith("DIMENSIONS"):
            parts = ln.split()
            if len(parts) >= 5:
                try:
                    kv["Xmin"] = float(parts[1]); kv["Ymin"] = float(parts[2])
                    kv["Xmax"] = float(parts[3]); kv["Ymax"] = float(parts[4])
                except:  # noqa: E722
                    pass
        elif up.startswith("UNITS"):
            parts = ln.split()
            kv["Units"] = parts[1] if len(parts) > 1 else ""
        elif up.startswith("FILE"):
            f = ln.split(None, 1)
            kv["File"] = f[1].strip() if len(f) > 1 else ""
        elif up.startswith("OFFSET"):
            parts = ln.split()
            if len(parts) >= 3:
                try:
                    kv["Xoff"] = float(parts[1]); kv["Yoff"] = float(parts[2])
                except:  # noqa: E722
                    pass
    return kv

def _parse_pumps(text: str) -> Dict[str, Dict[str, str]]:
    lines = _parse_simple_section(text, "PUMPS")
    pumps: Dict[str, Dict[str, str]] = {}
    for ln in lines:
        toks = re.split(r"\s+", ln)
        if len(toks) < 3:
            continue
        pid = toks[0]
        if "HEAD" in toks:
            i = toks.index("HEAD")
            curve = toks[i+1] if i+1 < len(toks) else ""
            pumps[pid] = {"ParamType": "HEAD", "CurveID": curve}
        elif "POWER" in toks:
            i = toks.index("POWER")
            val = toks[i+1] if i+1 < len(toks) else ""
            pumps[pid] = {"ParamType": "POWER", "Power": val}
        else:
            pumps[pid] = {"ParamType": "UNKNOWN"}
    return pumps

def _parse_curves(text: str) -> Dict[str, List[Tuple[float, float]]]:
    lines = _parse_simple_section(text, "CURVES")
    curves: Dict[str, List[Tuple[float, float]]] = {}
    for ln in lines:
        toks = re.split(r"\s+", ln)
        if len(toks) >= 3:
            cid = toks[0]
            try:
                x = float(toks[1]); y = float(toks[2])
            except:  # noqa: E722
                continue
            curves.setdefault(cid, []).append((x, y))
    for k, v in curves.items():
        curves[k] = sorted(v, key=lambda p: p[0])
    return curves

def _add(issue_list: List[Dict[str, Any]], sev: str, cat: str, item: str, detail: str, hint: str = ""):
    issue_list.append({"severity": sev, "category": cat, "item": item, "detail": detail, "suggestion": hint})

def _fmt_list(vals: List[Any], maxn: int = 8) -> str:
    vals = list(vals)
    if len(vals) <= maxn:
        return ", ".join(map(str, vals))
    return ", ".join(map(str, vals[:maxn])) + f", … (+{len(vals)-maxn} more)"


# --------------------------- main checker class ---------------------------

class FireFlowChecker:
    def __init__(self, inp_path: str | Path):
        self.inp_path = Path(inp_path)
        self.text = _read_inp_text(self.inp_path)
        self.wn = wntr.network.WaterNetworkModel(str(self.inp_path))
        self.issues: List[Dict[str, Any]] = []
        self.opt = _parse_options(self.text)
        self.topt = _parse_times(self.text)
        self.backdrop = _parse_backdrop(self.text)
        self.pumps_inp = _parse_pumps(self.text)
        self.curves = _parse_curves(self.text)

    def run_all(self) -> "FireFlowChecker":
        self.check_units_and_model()
        self.check_times()
        self.check_coordinates_and_elevations()
        self.check_demands_and_patterns()
        self.check_pumps_and_curves()
        self.check_valves_prvs_fcv()
        self.check_hydrant_proxies()
        self.check_connectivity_and_status()
        self.check_controls_vs_tanks()
        self.check_sources()
        return self

    # --------------------------- checks ---------------------------

    def check_units_and_model(self):
        units = self.opt.get("Units", "").upper()
        headloss = self.opt.get("Headloss", "").upper() or self.opt.get("Headloss:", "").upper()
        demand_model = self.opt.get("Demand Model", "").upper()
        emitter_exp = self.opt.get("Emitter Exponent", "")

        if units in {"GPM", "MGD"}:
            _add(self.issues, SEV_INF, "Units", "Flow units", f"{units} (US customary)",
                 "For OFH, gpm/psi are typical; ensure pressure units are psi for reporting.")
        elif units in {"LPS", "LPM", "MLD", "CMS", "CMH"}:
            _add(self.issues, SEV_WRN, "Units", "Flow units", f"{units} (SI)",
                 "If fire-flow targets are in gpm/psi, set report conversions or switch to GPM.")
        else:
            if units:
                _add(self.issues, SEV_WRN, "Units", "Flow units", f"{units}", "Unusual/unknown flow unit—double-check.")
            else:
                _add(self.issues, SEV_WRN, "Units", "Flow units", "Missing Units in [OPTIONS].", "Add 'Units GPM' or preferred.")

        if headloss not in {"H-W", "D-W", "C-M"}:
            _add(self.issues, SEV_WRN, "Hydraulics", "Headloss model", f"{headloss or '(missing)'}",
                 "Set H-W (typical for DI/CI) or D-W for high accuracy.")

        dm = getattr(self.wn.options.hydraulic, "demand_model", None)
        if demand_model and demand_model not in {"PDD", "DDA"}:
            _add(self.issues, SEV_WRN, "Demand Model", "Demand Model", f"{demand_model}",
                 "Use PDD for fire-flow deficits; DDA masks shortages.")
        if not dm:
            if demand_model != "PDD":
                _add(self.issues, SEV_WRN, "Demand Model", "WNTR demand_model", "(unknown, default likely DDA)",
                     "Explicitly set wn.options.hydraulic.demand_model='PDD' for OFH.")
        else:
            if str(dm).upper() != "PDD":
                _add(self.issues, SEV_WRN, "Demand Model", "WNTR demand_model", str(dm),
                     "Switch to PDD for realistic residuals under deficit.")

        req = self.opt.get("Required Pressure", "")
        minp = self.opt.get("Minimum Pressure", "")
        pexp = self.opt.get("Pressure Exponent", "") or emitter_exp
        if str(dm).upper() == "PDD" or demand_model == "PDD":
            if not req or not minp:
                _add(self.issues, SEV_WRN, "Demand Model", "PDD parameters",
                     "Missing Required/Minimum Pressure.",
                     "Add 'Required Pressure 15' and 'Minimum Pressure 0-5' (m).")
            if not pexp:
                _add(self.issues, SEV_WRN, "Demand Model", "Pressure Exponent", "Missing.",
                     "Use ~0.5 (Wagner-like) unless calibrated.")

    def check_times(self):
        dur = self.topt.get("Duration", "")
        if dur and not dur.startswith("0"):
            _add(self.issues, SEV_INF, "Timing", "Duration", dur,
                 "For single-snapshot OFH, consider 'Duration 0:00' and set Pattern Start for non-zero demand hour.")
        rpt = self.topt.get("Report Timestep", "")
        if rpt and rpt.strip() not in {"0:05", "0:01", "0:15"}:
            _add(self.issues, SEV_INF, "Timing", "Report Timestep", rpt,
                 "5 min is common; use 0:05 unless you need finer granularity.")

    def check_coordinates_and_elevations(self):
        missing_xy = []
        xs, ys = [], []
        for nname in self.wn.node_name_list:
            n = self.wn.get_node(nname)
            xy = getattr(n, "coordinates", None)
            if not xy or any([v is None or (isinstance(v, float) and math.isnan(v)) for v in xy]):
                missing_xy.append(nname)
            else:
                xs.append(xy[0]); ys.append(xy[1])
        if missing_xy:
            _add(self.issues, SEV_WRN, "Geometry", "Missing coordinates", _fmt_list(missing_xy),
                 "Add X,Y for all nodes (maps, nearest hydrant search, PRV debugging).")

        if self.backdrop:
            xmin = self.backdrop.get("Xmin"); xmax = self.backdrop.get("Xmax")
            ymin = self.backdrop.get("Ymin"); ymax = self.backdrop.get("Ymax")
            if None not in (xmin, xmax, ymin, ymax) and xs and ys:
                gxmin, gxmax = min(xs), max(xs)
                gymin, gymax = min(ys), max(ys)
                if (gxmin < xmin-1e-6 or gxmax > xmax+1e-6 or gymin < ymin-1e-6 or gymax > ymax+1e-6):
                    _add(self.issues, SEV_INF, "Geometry", "Backdrop mismatch",
                         (f"Network bbox ({gxmin:.1f},{gymin:.1f})–({gxmax:.1f},{gymax:.1f}) vs "
                          f"backdrop ({xmin:.1f},{ymin:.1f})–({xmax:.1f},{ymax:.1f})"),
                         "Check BACKDROP DIMENSIONS/OFFSET.")

        bad_elev, neg_elev = [], []
        for jn in self.wn.junction_name_list:
            j = self.wn.get_node(jn)
            el = getattr(j, "elevation", None)
            if el is None or not math.isfinite(el):
                bad_elev.append(jn)
            elif el < -5:
                neg_elev.append(jn)
        if bad_elev:
            _add(self.issues, SEV_ERR, "Data", "Missing/NaN elevations", _fmt_list(bad_elev),
                 "Set elevations for all junctions (pressures depend on it).")
        if neg_elev:
            _add(self.issues, SEV_WRN, "Data", "Negative elevations", _fmt_list(neg_elev),
                 "Verify datum; negative ground is unusual.")

    def _junction_demand_info(self, j) -> Tuple[float, List[str]]:
        patterns: List[str] = []
        base_total = 0.0
        dlist = getattr(j, "demand_timeseries_list", None) or []
        for dt in dlist:
            base = getattr(dt, "base_value", 0.0) or 0.0
            base_total += float(base)
            p = getattr(dt, "pattern", None)
            pname = getattr(p, "name", None) if p is not None else None
            if pname:
                patterns.append(pname)
        if base_total == 0.0 and hasattr(j, "base_demand"):
            try:
                base_total = float(j.base_demand)
            except:  # noqa: E722
                pass
        if not patterns and hasattr(j, "demand_pattern_name"):
            if j.demand_pattern_name:
                patterns = [j.demand_pattern_name]
        return base_total, patterns

    def check_demands_and_patterns(self):
        all_patterns = set(self.wn.pattern_name_list or [])
        missing_pat_refs: Dict[str, List[str]] = {}
        neg_dem: List[Tuple[str, float]] = []
        huge_dem: List[Tuple[str, float]] = []

        for jn in self.wn.junction_name_list:
            j = self.wn.get_node(jn)
            base, pnames = self._junction_demand_info(j)
            if base is None:
                continue
            if base < 0:
                neg_dem.append((jn, base))
            if base > 1000:  # aggressive heuristic, but catches unit mistakes
                huge_dem.append((jn, base))
            for pn in pnames or []:
                if pn not in all_patterns:
                    missing_pat_refs.setdefault(jn, []).append(pn)

        if neg_dem:
            _add(self.issues, SEV_ERR, "Demands", "Negative base demands",
                 _fmt_list([f"{n}({v})" for n, v in neg_dem]),
                 "Fix sign; use negative emitters for sources, not negative demands.")
        if huge_dem:
            tops = sorted(huge_dem, key=lambda t: -t[1])[:10]
            _add(self.issues, SEV_WRN, "Demands", "Unusually large base demands",
                 _fmt_list([f"{n}({v:g})" for n, v in tops]),
                 "Check units and misplaced zeros (e.g., CMH vs L/s vs gpm).")
        if missing_pat_refs:
            ex = []
            for n, pats in list(missing_pat_refs.items())[:10]:
                ex.append(f"{n}:{'/'.join(pats)}")
            _add(self.issues, SEV_ERR, "Patterns", "Referenced patterns missing",
                 _fmt_list(ex), "Create patterns or fix names.")

        dmult = self.opt.get("Demand Multiplier", "")
        if dmult and dmult not in {"1", "1.0", "1.0000"}:
            _add(self.issues, SEV_INF, "Demands", "Demand Multiplier", dmult,
                 "Confirm multiplier isn’t double-scaling with patterns.")

    def check_pumps_and_curves(self):
        for pid, pinf in self.pumps_inp.items():
            if pinf.get("ParamType") == "HEAD":
                cid = pinf.get("CurveID", "")
                pts = self.curves.get(cid, [])
                if not pts:
                    _add(self.issues, SEV_ERR, "Pumps", pid, f"HEAD curve '{cid}' missing/empty.",
                         "Define curve in [CURVES].")
                    continue
                flows = [p[0] for p in pts]
                heads = [p[1] for p in pts]
                if any(flows[i] > flows[i+1] for i in range(len(flows)-1)):
                    _add(self.issues, SEV_ERR, "Pumps", f"{pid}/{cid}",
                         "Curve x (flow) not strictly ascending.", "Sort points, remove duplicates.")
                incr = [heads[i+1] - heads[i] for i in range(len(heads)-1)]
                if any(dh > 1e-6 for dh in incr):
                    _add(self.issues, SEV_WRN, "Pumps", f"{pid}/{cid}",
                         "Head increases with flow at some segment.", "Fix measured points or smooth physically.")
                if heads[0] <= 0:
                    _add(self.issues, SEV_WRN, "Pumps", f"{pid}/{cid}",
                         f"H(0)={heads[0]:.2f}", "Zero-flow head should be > 0.")
            elif pinf.get("ParamType") == "POWER":
                if not pinf.get("Power"):
                    _add(self.issues, SEV_ERR, "Pumps", pid, "POWER pump missing power value.", "Add kW.")

    def check_valves_prvs_fcv(self):
        for vname in self.wn.valve_name_list:
            v = self.wn.get_link(vname)
            vtype = str(getattr(v, "valve_type", getattr(v, "valve_type_name", "")))
            setting = getattr(v, "setting", None)
            n1, n2 = v.start_node_name, v.end_node_name
            if vtype.upper() == "PRV":
                if setting is None or not math.isfinite(setting) or setting < 0 or setting > 200:
                    _add(self.issues, SEV_WRN, "PRV", vname, f"Setting={setting}",
                         "Typical PRV head setpoint ~10–80 m (15–120 psi). Verify units.")
            if vtype.upper() == "FCV":
                if (setting is not None) and float(setting) <= 0:
                    _add(self.issues, SEV_INF, "FCV", vname, "Setting<=0 (likely CLOSED).",
                         "Ensure connectivity isn’t broken if this is the only supply path.")
            if vtype.upper() == "PBV":
                _add(self.issues, SEV_INF, "Valve", vname, "PBV present.",
                     "PBVs can mask headloss reality; confirm intent.")
            for nn in (n1, n2):
                if nn not in self.wn.node_name_list:
                    _add(self.issues, SEV_ERR, "Valves", vname, f"Endpoint node '{nn}' missing.", "Fix node IDs.")

    def check_hydrant_proxies(self):
        hydrant_like = []
        for jn in self.wn.junction_name_list:
            j = self.wn.get_node(jn)
            ec = getattr(j, "emitter_coefficient", None)
            if ec and float(ec) > 0:
                hydrant_like.append(jn)
        if not hydrant_like:
            _add(self.issues, SEV_INF, "Hydrants", "No emitter-based hydrants found",
                 "Use emitter coefficients or explicit laterals for hydrant tests.")
        else:
            _add(self.issues, SEV_INF, "Hydrants", "Emitter hydrants", f"{len(hydrant_like)} nodes",
                 "Good. Ensure K is calibrated to nozzle, μ & residual spec.")

    def check_connectivity_and_status(self):
        G = self.wn.get_graph().to_undirected()
        to_remove = []
        for lname in self.wn.link_name_list:
            l = self.wn.get_link(lname)
            st = str(getattr(l, "initial_status", getattr(l, "status", ""))).upper()
            if st in {"CLOSED", "CLOSE"}:
                if G.has_edge(l.start_node_name, l.end_node_name):
                    to_remove.append((l.start_node_name, l.end_node_name, lname))
        for u, v, _ in to_remove:
            if G.has_edge(u, v):
                try:
                    G.remove_edge(u, v)
                except:  # noqa: E722
                    pass

        sources = set(self.wn.tank_name_list) | set(self.wn.reservoir_name_list)
        if not sources:
            _add(self.issues, SEV_WRN, "Connectivity", "No tanks/reservoirs",
                 "Network has no source nodes.", "Add source or confirm boundary conditions.")
        comps = list(nx.connected_components(G))
        if len(comps) > 1:
            sizes = sorted([len(c) for c in comps], reverse=True)
            _add(self.issues, SEV_WRN, "Connectivity", "Multiple disconnected components", f"Sizes: {sizes}",
                 "Closed links or missing connections may isolate areas.")

        if sources:
            reachable = set()
            for s in sources:
                if s in G:
                    reachable |= nx.node_connected_component(G, s)
            unreachable = [n for n in self.wn.junction_name_list if n not in reachable]
            if unreachable:
                _add(self.issues, SEV_ERR, "Connectivity", "Junctions unreachable from sources",
                     _fmt_list(unreachable), "Open valves/pipes or add connections.")

        closed_links = []
        for lname in self.wn.link_name_list:
            l = self.wn.get_link(lname)
            st = str(getattr(l, "initial_status", getattr(l, "status", ""))).upper()
            if st in {"CLOSED", "CLOSE"}:
                closed_links.append(lname)
        if closed_links:
            _add(self.issues, SEV_INF, "Status", "Closed links at start", _fmt_list(closed_links),
                 "Confirm intentional closures; these can starve fire-flows.")

    def check_controls_vs_tanks(self):
        lines = _parse_simple_section(self.text, "CONTROLS")
        if not lines:
            return
        for ln in lines:
            m = re.search(r"IF\s+NODE\s+(\S+)\s+(BELOW|ABOVE)\s+([0-9.+-Ee]+)", ln, re.I)
            if m:
                tank = m.group(1); rel = m.group(2).upper(); th = float(m.group(3))
                if tank in self.wn.tank_name_list:
                    t = self.wn.get_node(tank)
                    minL = getattr(t, "min_level", None)
                    maxL = getattr(t, "max_level", None)
                    if None not in (minL, maxL):
                        if rel == "BELOW" and th < minL - 1e-6:
                            _add(self.issues, SEV_WRN, "Controls", f"{ln.strip()}",
                                 f"Threshold {th:g} below tank MinLevel {minL:g}.",
                                 "Adjust threshold within [MinLevel, MaxLevel].")
                        if rel == "ABOVE" and th > maxL + 1e-6:
                            _add(self.issues, SEV_WRN, "Controls", f"{ln.strip()}",
                                 f"Threshold {th:g} above tank MaxLevel {maxL:g}.",
                                 "Adjust threshold within [MinLevel, MaxLevel].")
                else:
                    if tank not in self.wn.node_name_list:
                        _add(self.issues, SEV_ERR, "Controls", f"{ln.strip()}",
                             f"Condition node '{tank}' not found.", "Fix ID in [CONTROLS].")
            m2 = re.match(r"(LINK|PUMP|VALVE)\s+(\S+)\s+(open|closed|OPEN|CLOSED)", ln)
            if m2:
                tgt = m2.group(2)
                if tgt not in self.wn.link_name_list:
                    _add(self.issues, SEV_ERR, "Controls", f"{ln.strip()}",
                         f"Target link '{tgt}' not found.", "Fix ID in [CONTROLS].")

    def check_sources(self):
        # --- Robust reservoir head getter (supports WNTR variants) ---
        def _res_head(res) -> Optional[float]:
            if hasattr(res, "base_head") and res.base_head is not None:
                return float(res.base_head)
            if hasattr(res, "head") and res.head is not None:
                try:
                    return float(res.head)
                except:  # noqa: E722
                    pass
            ts = getattr(res, "head_timeseries", None)
            if ts is not None:
                bv = getattr(ts, "base_value", None)
                if bv is not None:
                    try:
                        return float(bv)
                    except:  # noqa: E722
                        pass
            lines = _parse_simple_section(self.text, "RESERVOIRS")
            if lines:
                rid = res.name if hasattr(res, "name") else None
                for ln in lines:
                    toks = ln.split()
                    if toks and rid and toks[0] == rid and len(toks) >= 2:
                        try:
                            return float(toks[1])
                        except:  # noqa: E722
                            pass
            return None

        # Build connectivity (minus initially closed links)
        G = self.wn.get_graph().to_undirected()
        for lname in list(self.wn.link_name_list):
            l = self.wn.get_link(lname)
            st = str(getattr(l, "initial_status", getattr(l, "status", ""))).upper()
            if st in {"CLOSED", "CLOSE"} and G.has_edge(l.start_node_name, l.end_node_name):
                try:
                    G.remove_edge(l.start_node_name, l.end_node_name)
                except:  # noqa: E722
                    pass

        # Reservoir checks
        for rid in self.wn.reservoir_name_list:
            r = self.wn.get_node(rid)
            head = _res_head(r)
            deg = G.degree(rid) if rid in G else 0
            if head is None:
                if deg == 0:
                    _add(self.issues, SEV_WRN, "Reservoirs", rid,
                         "Head missing but reservoir is disconnected.",
                         "Either remove the placeholder reservoir or assign a head if you plan to use it.")
                else:
                    _add(self.issues, SEV_ERR, "Reservoirs", rid,
                         "Missing/NaN head.", "Set a numeric head.")
            else:
                elevs = [self.wn.get_node(j).elevation
                         for j in self.wn.junction_name_list
                         if hasattr(self.wn.get_node(j), "elevation")]
                if elevs:
                    med = float(pd.Series(elevs).median())
                    if head < med - 10:
                        _add(self.issues, SEV_WRN, "Reservoirs", rid,
                             f"Head {head:g} < median elevation {med:g}.",
                             "Unusual unless below-grade source or siphon; verify datum.")

        # Tank init within [min,max]
        for tid in self.wn.tank_name_list:
            t = self.wn.get_node(tid)
            minL = getattr(t, "min_level", None)
            initL = getattr(t, "init_level", None)
            maxL = getattr(t, "max_level", None)
            if None not in (minL, initL, maxL):
                if not (minL - 1e-6 <= initL <= maxL + 1e-6):
                    _add(self.issues, SEV_ERR, "Tanks", tid,
                         f"InitLevel {initL:g} outside [{minL:g}, {maxL:g}].",
                         "Fix initial water level.")

    # --------------------------- reporting ---------------------------

    def to_dataframe(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["severity", "category", "item", "detail", "suggestion"])
        df = pd.DataFrame(self.issues)
        sev_rank = {SEV_ERR: 0, SEV_WRN: 1, SEV_INF: 2}
        df["sev_rank"] = df["severity"].map(sev_rank).fillna(9)
        df = df.sort_values(["sev_rank", "category", "item"]).drop(columns="sev_rank")
        return df

    def print_report(self, max_rows: int = 200):
        df = self.to_dataframe()
        if df.empty:
            print("✅ No issues found. (Still run a sanity snapshot with PDD + hydrant tests!)")
            return
        counts = df["severity"].value_counts().to_dict()
        print(f"=== Fire-Flow Checker Report v{__version__} for {self.inp_path.name} ===")
        print(f"Totals: {counts.get(SEV_ERR,0)} ERRORS, {counts.get(SEV_WRN,0)} WARNINGS, {counts.get(SEV_INF,0)} INFOS")
        with pd.option_context("display.max_colwidth", 120, "display.width", 160):
            print(df.head(max_rows).to_string(index=False))


# --------------------------- CLI ---------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fireflow_checker.py path/to/model.inp")
        sys.exit(1)
    checker = FireFlowChecker(sys.argv[1]).run_all()
    checker.print_report()
