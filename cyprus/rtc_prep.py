# rtc_prep.py
# Prep + utilities for real-time control experiments (normal vs. fire)
# Assumes EPANET INP units are SI with CMH for flow and meters for head.
# WNTR results are SI base (m, m^3/s); helpers convert for convenience.

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd
import wntr
import networkx as nx


# --------------------------- Conversions ---------------------------

def gpm_to_cmh(q_gpm: float) -> float:
    return float(q_gpm) / 4.4028675

def cmh_to_gpm(q_cmh: float) -> float:
    return float(q_cmh) * 4.4028675

def psi_to_mhead(p_psi: float) -> float:
    return float(p_psi) / 1.422334

def mhead_to_psi(h_m: float) -> float:
    return float(h_m) * 1.422334


# --------------------------- Version-tolerant valve helpers ---------------------------

def _add_valve_generic(
    wn,
    valve_id: str,
    n1: str,
    n2: str,
    diameter_m: float,
    valve_type: str,
    setting_value=None,
    minor_loss: float = 0.0,
):
    """Create a valve across WNTR versions; apply .setting afterwards."""
    try:
        if setting_value is None:
            wn.add_valve(valve_id, n1, n2, diameter_m, valve_type, minor_loss=minor_loss)
        else:
            wn.add_valve(valve_id, n1, n2, diameter_m, valve_type, setting_value, minor_loss)
    except TypeError:
        if setting_value is None:
            wn.add_valve(valve_id, n1, n2, diameter_m, valve_type)
        else:
            wn.add_valve(valve_id, n1, n2, diameter_m, valve_type, setting_value)
    v = wn.get_link(valve_id)
    try:
        if setting_value is not None:
            v.setting = setting_value
    except Exception:
        pass
    try:
        if hasattr(v, "initial_status"):
            v.initial_status = "Open"
    except Exception:
        pass
    return v

def _add_valve_compat(wn, valve_id, n1, n2, diameter_m, valve_type, setting_m3s=None, minor_loss=0.0):
    """As above but with 'setting_m3s' naming for FCV flows."""
    return _add_valve_generic(
        wn, valve_id=valve_id, n1=n1, n2=n2, diameter_m=diameter_m,
        valve_type=valve_type, setting_value=setting_m3s, minor_loss=minor_loss
    )

def service_node_names(wn: wntr.network.WaterNetworkModel) -> List[str]:
    """All nodes except reservoirs; keeps junctions + tanks."""
    res = set(wn.reservoir_name_list or [])
    return [n for n in wn.node_name_list if n not in res]

def min_service_pressure_m(P_m: pd.Series, wn: wntr.network.WaterNetworkModel) -> float:
    """Min pressure among service nodes (excludes reservoirs)."""
    names = service_node_names(wn)
    return float(P_m.loc[names].min())

def count_below_floor(P_m: pd.Series, wn: wntr.network.WaterNetworkModel, floor_psi: float) -> int:
    """Count service nodes with pressure below a psi floor."""
    names = service_node_names(wn)
    floor_m = psi_to_mhead(float(floor_psi))
    return int((P_m.loc[names] < floor_m).sum())


# --------------------------- Model load & snapshot config ---------------------------

def load_model(inp_path: str) -> wntr.network.WaterNetworkModel:
    return wntr.network.WaterNetworkModel(str(inp_path))

def set_pdd_snapshot(
    wn: wntr.network.WaterNetworkModel,
    required_m: float = 15.0,
    pattern_hour: int = 8,
    minimum_m: float = 0.0,
    pressure_exponent: float = 0.5,
) -> None:
    """Switch network to PDD and configure a snapshot (Duration=0)."""
    wn.options.hydraulic.demand_model = "PDD"
    wn.options.hydraulic.minimum_pressure = float(minimum_m)
    wn.options.hydraulic.required_pressure = float(required_m)
    wn.options.hydraulic.pressure_exponent = float(pressure_exponent)
    wn.options.time.duration = 0
    wn.options.time.report_timestep = 3600
    wn.options.time.pattern_start = int(pattern_hour) * 3600


# --------------------------- Hydrant proxies ---------------------------

def attach_hydrant_emitter(
    wn: wntr.network.WaterNetworkModel,
    node_id: str,
    target_gpm: float = 1000.0,
    residual_psi: float = 20.0,
) -> float:
    """
    Attach an emitter so Q≈target_gpm at residual_psi.
    Emitter law: Q = K * sqrt(P). Returns K in CMH/√m; sets j.emitter_coefficient.
    """
    q_cmh = gpm_to_cmh(target_gpm)
    h_m = max(psi_to_mhead(residual_psi), 1e-6)
    K = q_cmh / math.sqrt(h_m)
    j = wn.get_node(node_id)
    j.emitter_coefficient = K
    return K

def remove_hydrant_emitter(wn: wntr.network.WaterNetworkModel, node_id: str) -> None:
    j = wn.get_node(node_id)
    j.emitter_coefficient = 0.0

def add_hydrant_fcv(
    wn: wntr.network.WaterNetworkModel,
    main_node: str,
    hydrant_node: Optional[str] = None,
    diam_mm: float = 100.0,
    fcv_id: Optional[str] = None,
    target_gpm: float = 1000.0,
) -> Tuple[str, str]:
    """
    Deterministic hydrant via FCV lateral:
      - Add hydrant node with base_demand == target flow
      - Connect main_node --[FCV(setting=target m^3/s)]--> hydrant_node
    Returns (hydrant_node_id, fcv_id).
    """
    if main_node not in wn.node_name_list:
        raise ValueError(f"main_node '{main_node}' not found")
    if hydrant_node is None:
        hydrant_node = f"HYD_{main_node}"
    if fcv_id is None:
        fcv_id = f"FCV_{hydrant_node}"

    j_main = wn.get_node(main_node)
    q_cmh = gpm_to_cmh(target_gpm)
    q_m3s = q_cmh / 3600.0

    if hydrant_node not in wn.node_name_list:
        wn.add_junction(
            hydrant_node,
            base_demand=float(q_m3s),
            demand_pattern=None,
            elevation=float(getattr(j_main, "elevation", 0.0)),
        )
    else:
        j_h = wn.get_node(hydrant_node)
        try:
            j_h.base_demand = float(q_m3s)
        except Exception:
            pass

    if fcv_id not in wn.link_name_list:
        _add_valve_compat(
            wn,
            valve_id=fcv_id,
            n1=main_node,
            n2=hydrant_node,
            diameter_m=float(diam_mm) / 1000.0,
            valve_type="FCV",
            setting_m3s=q_m3s,
            minor_loss=0.0,
        )
    else:
        v = wn.get_link(fcv_id)
        try:
            v.setting = q_m3s
        except Exception:
            pass
        try:
            if hasattr(v, "initial_status"):
                v.initial_status = "Open"
        except Exception:
            pass

    return hydrant_node, fcv_id


# --------------------------- Optional: editing links for PRV tests ---------------------------

def _add_junction_compat(wn, junc_id, elevation, base_demand=0.0):
    if junc_id not in wn.node_name_list:
        wn.add_junction(junc_id, base_demand=float(base_demand), demand_pattern=None, elevation=float(elevation))
    return wn.get_node(junc_id)

def insert_prv_in_series(
    wn,
    link_id: str,
    prv_id: str,
    setpoint_m: float,
    mid_node: str | None = None,
    diameter_m: float | None = None,
):
    """
    Split pipe 'link_id' and insert one upstream pipe + one PRV (no parallel).
    n1 --[pipe_A]--> mid_node --[PRV]--> n2
    """
    if link_id not in wn.link_name_list:
        raise ValueError(f"Link '{link_id}' not found")
    L = wn.get_link(link_id)
    link_type = str(getattr(L, "link_type", getattr(L, "link_type_name", ""))).upper()
    if link_type != "PIPE":
        raise ValueError("insert_prv_in_series expects a PIPE link")

    n1, n2 = L.start_node_name, L.end_node_name
    if mid_node is None:
        mid_node = f"J_{link_id}_PRV"
    elev = 0.5 * (wn.get_node(n1).elevation + wn.get_node(n2).elevation)
    _add_junction_compat(wn, mid_node, elevation=elev)

    length = float(getattr(L, "length", 1.0))
    dia = float(diameter_m if diameter_m is not None else getattr(L, "diameter", 0.1))
    rough = float(getattr(L, "roughness", 100.0))
    minor = float(getattr(L, "minor_loss", 0.0))

    wn.remove_link(link_id)

    pA = f"{link_id}_A"
    wn.add_pipe(pA, n1, mid_node, length=max(0.5 * length, 0.1), diameter=dia, roughness=rough, minor_loss=minor)

    v = _add_valve_generic(
        wn, valve_id=prv_id, n1=mid_node, n2=n2,
        diameter_m=dia, valve_type="PRV", setting_value=float(setpoint_m)
    )
    return v

def replace_pipe_with_prv(wn, pipe_id: str, prv_id: str, setpoint_m: float):
    """
    Replace an existing PIPE with a PRV between the SAME two nodes (no mid-node).
    """
    if pipe_id not in wn.link_name_list:
        raise ValueError(f"Pipe '{pipe_id}' not found")
    L = wn.get_link(pipe_id)
    lt = str(getattr(L, "link_type", getattr(L, "link_type_name", ""))).upper()
    if lt != "PIPE":
        raise ValueError(f"Link '{pipe_id}' is not a PIPE")
    n1, n2 = L.start_node_name, L.end_node_name
    dia = float(getattr(L, "diameter", 0.1))
    wn.remove_link(pipe_id)
    return _add_valve_generic(
        wn, valve_id=prv_id, n1=n1, n2=n2, diameter_m=dia, valve_type="PRV", setting_value=float(setpoint_m)
    )


# --------------------------- Control levers ---------------------------

def apply_mode(
    wn: wntr.network.WaterNetworkModel,
    mode: str,
    controls: Dict[str, float | str],
    prv_id: Optional[str] = "PRV_dist",
    fcv_id: Optional[str] = "FCV_Desalination",
) -> None:
    _ = mode  # reserved
    if prv_id and "PRV_set" in controls and prv_id in wn.valve_name_list:
        v = wn.get_link(prv_id)
        try:
            v.setting = float(controls["PRV_set"])
        except Exception:
            pass
    if fcv_id and "FCV_Desalination" in controls and fcv_id in wn.valve_name_list:
        fcv = wn.get_link(fcv_id)
        try:
            fcv.initial_status = str(controls["FCV_Desalination"]).capitalize()
        except Exception:
            pass
    if "pump_speed" in controls:
        for pid in wn.pump_name_list:
            p = wn.get_link(pid)
            try:
                p.speed = float(controls["pump_speed"])
            except Exception:
                pass


# --------------------------- Simulation & metrics ---------------------------

@dataclass
class ScenarioResult:
    mode: str
    maxP_m: float
    minP_m: float
    Q_hyd_cmh: Optional[float]
    maxP_exceed_m: float
    P_series: pd.Series
    D_series: pd.Series
    Q_series: pd.Series  # link flowrate in CMH
    results: wntr.sim.results.SimulationResults
    # --- new (service-node) fields ---
    min_service_m: Optional[float] = None
    nodes_below_floor: Optional[int] = None

def run_snapshot(wn: wntr.network.WaterNetworkModel) -> wntr.sim.results.SimulationResults:
    sim = wntr.sim.EpanetSimulator(wn)
    return sim.run_sim()

def scenario_metrics(
    wn: wntr.network.WaterNetworkModel,
    mode: str,
    hydrant_node: Optional[str] = None,
    pmax_psi: float = 100.0,
    *,
    service_only_min: bool = False,
    pressure_floor_psi: Optional[float] = None,
) -> ScenarioResult:
    """
    Compute snapshot metrics:
      - system max/min pressure (all nodes)
      - hydrant outflow (if hydrant_node given)
      - maximum pressure exceedance vs pmax_psi
    Optional:
      - service_only_min=True -> also compute min_service_m (exclude reservoirs)
      - pressure_floor_psi set -> also compute nodes_below_floor among service nodes
    """
    res = run_snapshot(wn)

    # Last (only) timestep for snapshot
    P_m = res.node["pressure"].iloc[-1]     # meters
    D = res.node["demand"].iloc[-1]         # m^3/s (WNTR SI)
    Q = res.link["flowrate"].iloc[-1]       # m^3/s

    D_cmh = D * 3600.0
    Q_cmh = Q * 3600.0
    Q_hyd_cmh = float(abs(D_cmh.get(hydrant_node, 0.0))) if hydrant_node else None

    Pmax_m = float(P_m.max())
    Pmin_m = float(P_m.min())
    Pmax_limit_m = psi_to_mhead(pmax_psi)
    exceed_m = max(0.0, Pmax_m - Pmax_limit_m)

    # Optional service-only stats
    min_serv = None
    n_below = None
    if service_only_min:
        try:
            min_serv = min_service_pressure_m(P_m, wn)
        except Exception:
            min_serv = None
    if pressure_floor_psi is not None:
        try:
            n_below = count_below_floor(P_m, wn, pressure_floor_psi)
        except Exception:
            n_below = None

    return ScenarioResult(
        mode=mode,
        maxP_m=Pmax_m,
        minP_m=Pmin_m,
        Q_hyd_cmh=Q_hyd_cmh,
        maxP_exceed_m=exceed_m,
        P_series=P_m,
        D_series=D_cmh,
        Q_series=Q_cmh,
        results=res,
        min_service_m=min_serv,
        nodes_below_floor=n_below,
    )


def estimate_energy_cost(
    wn: wntr.network.WaterNetworkModel,
    results: wntr.sim.results.SimulationResults,
    price_per_kWh: float = 0.12,
    eff: float = 0.75,
) -> float:
    """Very rough one-step energy estimate from hydraulic power."""
    rho_g = 9810.0       # N/m^3
    dt_hr = 1.0
    cost = 0.0
    link_flow = results.link["flowrate"].iloc[-1]   # m^3/s
    link_hloss = results.link["headloss"].iloc[-1]  # m
    for pid in wn.pump_name_list:
        q = float(link_flow.get(pid, 0.0))
        dh = float(link_hloss.get(pid, 0.0))
        P_watts = (q * dh * rho_g) / max(eff, 1e-3)
        cost += (P_watts / 1000.0) * dt_hr * price_per_kWh
    return cost


# --------------------------- Path diagnostics ---------------------------

def path_report(
    wn: wntr.network.WaterNetworkModel,
    dst_node: str,
    respect_closed: bool = True,
) -> Tuple[List[str], bool, Set[Tuple[str, str]]]:
    """Return (sources, any_path_exists, set of edges on some shortest path to dst_node)."""
    sources = list(set(wn.tank_name_list) | set(wn.reservoir_name_list))
    G = wn.get_graph().to_undirected()
    if respect_closed:
        for lname in wn.link_name_list:
            l = wn.get_link(lname)
            st = str(getattr(l, "initial_status", getattr(l, "status", ""))).upper()
            if st in {"CLOSED", "CLOSE"} and G.has_edge(l.start_node_name, l.end_node_name):
                try:
                    G.remove_edge(l.start_node_name, l.end_node_name)
                except Exception:
                    pass
    paths = []
    for s in sources:
        if s in G and dst_node in G:
            try:
                p = nx.shortest_path(G, s, dst_node)
                paths.append(p)
            except nx.NetworkXNoPath:
                pass
    linkset: Set[Tuple[str, str]] = set()
    for p in paths:
        linkset |= set(zip(p[:-1], p[1:]))
    return sources, bool(paths), linkset

def lever_is_on_paths(
    wn: wntr.network.WaterNetworkModel,
    lever_link_id: str,
    linkset: Set[Tuple[str, str]],
) -> bool:
    if lever_link_id not in wn.link_name_list:
        return False
    u = wn.get_link(lever_link_id).start_node_name
    v = wn.get_link(lever_link_id).end_node_name
    return (u, v) in linkset or (v, u) in linkset

def link_on_any_source_path(
    wn: wntr.network.WaterNetworkModel,
    link_id: str,
    dst_node: str,
    respect_closed: bool = True,
) -> bool:
    _, ok, linkset = path_report(wn, dst_node, respect_closed=respect_closed)
    if not ok:
        return False
    return lever_is_on_paths(wn, link_id, linkset)


# --------------------------- Workflows ---------------------------

def prepare_for_rtc(
    inp_path: str,
    hydrant_nodes: Optional[List[str]] = None,
    target_gpm: float = 1000.0,
    residual_psi: float = 20.0,
    required_m: float = 15.0,
    pattern_hour: int = 8,
) -> Tuple[wntr.network.WaterNetworkModel, Dict[str, float]]:
    """Load, switch to PDD snapshot, optionally attach emitter hydrants."""
    wn = load_model(inp_path)
    set_pdd_snapshot(wn, required_m=required_m, pattern_hour=pattern_hour)

    emitters: Dict[str, float] = {}
    if hydrant_nodes:
        for node in hydrant_nodes:
            K = attach_hydrant_emitter(wn, node, target_gpm=target_gpm, residual_psi=residual_psi)
            emitters[node] = K
    return wn, emitters

def run_two_modes_simple(
    inp_path: str,
    hydrant_node: str,
    prv_candidates_m: List[float] = (15, 20, 25, 30, 35),
    pmax_psi: float = 100.0,
    open_desal_in_fire: bool = True,
) -> pd.DataFrame:
    """
    Grid search PRV setpoint for 'normal' and 'fire' modes (emitter hydrant in fire).
    """
    rows: List[Dict] = []
    for mode in ("normal", "fire"):
        for prv in prv_candidates_m:
            wn = load_model(inp_path)
            set_pdd_snapshot(wn, required_m=15.0, pattern_hour=8)

            if mode == "fire":
                attach_hydrant_emitter(wn, hydrant_node, target_gpm=1000.0, residual_psi=20.0)
                apply_mode(
                    wn, mode,
                    {"PRV_set": prv,
                     "FCV_Desalination": "Open" if open_desal_in_fire else "Closed",
                     "pump_speed": 1.0}
                )
            else:
                apply_mode(wn, mode, {"PRV_set": prv, "FCV_Desalination": "Closed", "pump_speed": 1.0})

            met = scenario_metrics(wn, mode=mode, hydrant_node=(hydrant_node if mode == "fire" else None), pmax_psi=pmax_psi)
            rows.append({
                "mode": mode,
                "PRV_set_m": prv,
                "maxP_psi": mhead_to_psi(met.maxP_m),
                "minP_psi": mhead_to_psi(met.minP_m),
                "Q_hyd_gpm": (cmh_to_gpm(met.Q_hyd_cmh) if met.Q_hyd_cmh is not None else None),
                "maxP_exceed_psi": max(0.0, mhead_to_psi(met.maxP_exceed_m)),
            })
    return pd.DataFrame(rows)


# --------------------------- Targeted Fixes & Fire sweeps ---------------------------

def calibrate_emitter_to_target(
    inp_path: str,
    node_id: str,
    target_gpm: float = 1000.0,
    pattern_hour: int = 8,
    required_m: float = 15.0,
    tol_gpm: float = 5.0,
    max_iter: int = 6,
) -> Tuple[wntr.network.WaterNetworkModel, float, float]:
    """
    Iteratively adjust emitter K to achieve target_gpm at node_id for snapshot.
    Returns (wn, K, achieved_gpm).
    """
    wn_chk = load_model(inp_path)
    sources, ok, _ = path_report(wn_chk, node_id, respect_closed=True)
    if not ok:
        raise RuntimeError(f"Hydrant node '{node_id}' not connected to sources. Sources: {sources}")

    wn = wn_chk
    set_pdd_snapshot(wn, required_m=required_m, pattern_hour=pattern_hour)

    K = attach_hydrant_emitter(wn, node_id, target_gpm=target_gpm, residual_psi=20.0)

    for _ in range(max_iter):
        met = scenario_metrics(wn, mode="fire", hydrant_node=node_id, pmax_psi=999)
        q_now_gpm = cmh_to_gpm(met.Q_hyd_cmh or 0.0)
        p_node_m = float(met.P_series[node_id])
        if p_node_m < 0.5:
            break
        if abs(q_now_gpm - target_gpm) <= tol_gpm:
            return wn, K, q_now_gpm
        K = gpm_to_cmh(target_gpm) / (max(p_node_m, 1e-6) ** 0.5)
        wn.get_node(node_id).emitter_coefficient = K

    met = scenario_metrics(wn, mode="fire", hydrant_node=node_id, pmax_psi=999)
    return wn, K, cmh_to_gpm(met.Q_hyd_cmh or 0.0)

def sweep_fire_controls(
    inp_path: str,
    hydrant_node: str,
    prv_setpoints_m: List[float] = (15, 20, 25, 30, 35),
    pump_speeds: List[float] = (0.6, 0.8, 1.0),
    open_desal: bool = True,
    q_target_gpm: float = 1000.0,
    pmax_cap_psi: float = 100.0,
    pmin_floor_psi: float = 20.0,
    use_calibrated_K: bool = True,
    use_fcv_hydrant: bool = False,
    prv_id: str = "PRV_dist",
) -> pd.DataFrame:
    """
    Sweep PRV setpoint × pump speed for fire mode.
    Supports emitter hydrant (default) or FCV hydrant (use_fcv_hydrant=True).
    Skips PRV sweep if the PRV is off the source→hydrant path.
    """
    rows: List[Dict] = []

    wn_probe = load_model(inp_path)
    _, ok_path, linkset = path_report(wn_probe, hydrant_node, respect_closed=True)
    prv_on_path = ok_path and lever_is_on_paths(wn_probe, prv_id, linkset)

    if use_fcv_hydrant:
        K = None
    else:
        if use_calibrated_K:
            wn_base, K, _ = calibrate_emitter_to_target(inp_path, hydrant_node, target_gpm=q_target_gpm)
        else:
            wn_base = load_model(inp_path)
            set_pdd_snapshot(wn_base, required_m=15.0, pattern_hour=8)
            K = attach_hydrant_emitter(wn_base, hydrant_node, target_gpm=q_target_gpm, residual_psi=20.0)

    for sp in pump_speeds:
        for prv in prv_setpoints_m:
            if not prv_on_path and prv != prv_setpoints_m[0]:
                continue

            wn = load_model(inp_path)
            set_pdd_snapshot(wn, required_m=15.0, pattern_hour=8)

            if use_fcv_hydrant:
                hyd_node, _ = add_hydrant_fcv(wn, main_node=hydrant_node, target_gpm=q_target_gpm)
                hydrant_for_metric = hyd_node
            else:
                wn.get_node(hydrant_node).emitter_coefficient = K or 0.0
                hydrant_for_metric = hydrant_node

            apply_mode(
                wn, "fire",
                {"PRV_set": prv if prv_on_path else None,
                 "FCV_Desalination": ("Open" if open_desal else "Closed"),
                 "pump_speed": sp}
            )
            met = scenario_metrics(wn, mode="fire", hydrant_node=hydrant_for_metric, pmax_psi=pmax_cap_psi)

            maxP_psi = mhead_to_psi(met.maxP_m)
            minP_psi = mhead_to_psi(met.minP_m)
            q_gpm    = cmh_to_gpm(met.Q_hyd_cmh or 0.0)

            feasible = (q_gpm >= q_target_gpm) and (maxP_psi <= pmax_cap_psi) and (minP_psi >= pmin_floor_psi)

            rows.append({
                "PRV_id": prv_id,
                "PRV_on_path": prv_on_path,
                "PRV_m": (prv if prv_on_path else None),
                "pump_speed": sp,
                "maxP_psi": maxP_psi,
                "minP_psi": minP_psi,
                "Q_hyd_gpm": q_gpm,
                "feasible": feasible,
                "maxP_exceed_psi": max(0.0, maxP_psi - pmax_cap_psi),
                "used_fcv_hydrant": use_fcv_hydrant,
            })

    df = pd.DataFrame(rows).sort_values(
        by=["feasible", "maxP_exceed_psi", "Q_hyd_gpm"],
        ascending=[False, True, False]
    ).reset_index(drop=True)
    return df

def recommend_fire_settings(
    inp_path: str,
    hydrant_node: str,
    **kwargs
) -> pd.DataFrame:
    df = sweep_fire_controls(inp_path, hydrant_node, **kwargs)
    return df.head(5)


# --------------------------- Pump curve scaling & speed search ---------------------------

def scale_pump_head_curves_inplace(wn, speed: float) -> None:
    """
    Affinity scaling for HEAD curves (Q->Q*speed, H->H*speed^2).
    Modifies existing curve points in-place for this run. Reload wn to revert.
    """
    s = float(speed)
    s2 = s * s
    for pid in wn.pump_name_list:
        p = wn.get_link(pid)
        cid = getattr(p, "pump_curve_name", None) or getattr(p, "curve_name", None)
        if not cid:
            continue
        curve = wn.get_curve(cid)
        pts = list(getattr(curve, "points", []))
        if not pts:
            continue
        new_pts = [(q * s, h * s2) for (q, h) in pts]
        try:
            curve.points = new_pts
        except Exception:
            try:
                curve._points = new_pts
            except Exception:
                pass

def find_speed_for_pressure_cap(
    inp_path: str,
    hydrant_main_node: str,
    q_target_gpm: float = 1000.0,
    pmax_cap_psi: float = 100.0,
    required_m: float = 15.0,
    pattern_hour: int = 8,
    lo: float = 0.3,
    hi: float = 1.0,
    tol_psi: float = 0.5,
    max_iter: int = 12,
) -> Tuple[float, pd.Series]:
    """
    Bisection on pump speed to achieve system max pressure <= pmax_cap_psi
    with a deterministic FCV hydrant at q_target_gpm.
    Returns (best_speed, last_row_metrics).
    """
    target = float(pmax_cap_psi)

    def run_at(speed: float) -> Tuple[float, pd.Series]:
        wn = load_model(inp_path)
        set_pdd_snapshot(wn, required_m=required_m, pattern_hour=pattern_hour)
        hyd_node, _ = add_hydrant_fcv(wn, main_node=hydrant_main_node, target_gpm=q_target_gpm)
        scale_pump_head_curves_inplace(wn, speed)
        met = scenario_metrics(wn, mode="fire", hydrant_node=hyd_node, pmax_psi=target)
        row = pd.Series({
            "speed": speed,
            "maxP_psi": mhead_to_psi(met.maxP_m),
            "minP_psi": mhead_to_psi(met.minP_m),
            "Q_hyd_gpm": q_target_gpm,
            "maxP_exceed_psi": max(0.0, mhead_to_psi(met.maxP_exceed_m)),
        })
        return float(row["maxP_psi"]), row

    p_lo, r_lo = run_at(lo)
    p_hi, r_hi = run_at(hi)
    if p_lo > target + tol_psi:
        return lo, r_lo
    if p_hi <= target + tol_psi:
        return hi, r_hi

    best_speed, best_row = None, None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid, r_mid = run_at(mid)
        best_speed, best_row = mid, r_mid
        if abs(p_mid - target) <= tol_psi:
            break
        if p_mid > target:
            hi = mid
        else:
            lo = mid
    return float(best_speed), best_row

def recommend_fire_speed(
    inp_path: str,
    hydrant_main_node: str,
    q_target_gpm: float = 1000.0,
    pmax_cap_psi: float = 100.0,
) -> Tuple[float, pd.DataFrame]:
    spd, row = find_speed_for_pressure_cap(
        inp_path,
        hydrant_main_node=hydrant_main_node,
        q_target_gpm=q_target_gpm,
        pmax_cap_psi=pmax_cap_psi,
    )
    return spd, row.to_frame().T


# --------------------------- CLI (optional) ---------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="RTC prep for normal vs fire modes.")
    ap.add_argument("inp_path", help="Path to EPANET INP file")
    ap.add_argument("--hydrant", default=None, help="Node ID to anchor hydrant (emitter or FCV lateral)")
    ap.add_argument("--target_gpm", type=float, default=1000.0, help="Hydrant target flow in gpm")
    ap.add_argument("--residual_psi", type=float, default=20.0, help="Hydrant residual in psi (emitter seed)")
    ap.add_argument("--required_m", type=float, default=15.0, help="PDD required pressure (m)")
    ap.add_argument("--pattern_hour", type=int, default=8, help="Pattern start hour")
    ap.add_argument("--recommend", action="store_true", help="Run quick PRV×pump sweep recommender")
    ap.add_argument("--use_fcv_hydrant", action="store_true", help="Use FCV hydrant lateral instead of emitter")
    ap.add_argument("--pmax_cap_psi", type=float, default=100.0, help="System max pressure cap (psi)")
    ap.add_argument("--pmin_floor_psi", type=float, default=20.0, help="System minimum pressure floor (psi)")
    ap.add_argument("--prv_id", default="PRV_dist", help="PRV id to tune (must lie on path to be effective)")
    args = ap.parse_args()

    wn, emitters = prepare_for_rtc(
        args.inp_path,
        hydrant_nodes=([args.hydrant] if args.hydrant and not args.use_fcv_hydrant else None),
        target_gpm=args.target_gpm,
        residual_psi=args.residual_psi,
        required_m=args.required_m,
        pattern_hour=args.pattern_hour,
    )

    print("RTC prep complete.")
    if args.hydrant and not args.use_fcv_hydrant and emitters:
        for n, K in emitters.items():
            print(f"Hydrant emitter at {n}: K={K:.3f} (CMH/√m)")
    elif args.hydrant and args.use_fcv_hydrant:
        hyd_node, hyd_fcv = add_hydrant_fcv(wn, main_node=args.hydrant, target_gpm=args.target_gpm)
        print(f"FCV hydrant created: node={hyd_node}, valve={hyd_fcv}")

    if args.hydrant and args.recommend:
        print("Recommending fire settings...")
        df = recommend_fire_settings(
            args.inp_path, args.hydrant,
            q_target_gpm=args.target_gpm,
            pmax_cap_psi=args.pmax_cap_psi,
            pmin_floor_psi=args.pmin_floor_psi,
            use_fcv_hydrant=args.use_fcv_hydrant,
            prv_id=args.prv_id,
        )
        with pd.option_context("display.max_colwidth", 120, "display.width", 160):
            print(df.to_string(index=False))
