# =========================
# parser_inp.py — EPANET INP schema parser + large-object summarizer
# =========================
from __future__ import annotations

import sys, os, io, re, shlex, argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# ---------- Optional scientific stack (handled gracefully) ----------
try:
    import numpy as np  # type: ignore
except Exception:
    np = None
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None
try:
    import scipy.sparse as sp  # type: ignore
except Exception:
    sp = None
try:
    import networkx as nx  # type: ignore
except Exception:
    nx = None

# ================================================================
# Generic object "x-ray"
# ================================================================

def _sizeof(obj: Any) -> int:
    try:
        return sys.getsizeof(obj)
    except Exception:
        return 0

def _typename(obj: Any) -> str:
    t = type(obj)
    return f"{t.__module__}.{t.__name__}"

def _shape_and_len(obj: Any) -> str:
    parts = []
    if hasattr(obj, "__len__"):
        try:
            parts.append(f"len={len(obj)}")
        except Exception:
            pass
    if np is not None and isinstance(obj, np.ndarray):
        parts.append(f"shape={tuple(obj.shape)} dtype={obj.dtype}")
    if pd is not None and isinstance(obj, pd.DataFrame):
        parts.append(f"DataFrame[{obj.shape[0]} rows x {obj.shape[1]} cols]")
    if pd is not None and isinstance(obj, pd.Series):
        parts.append(f"Series[len={obj.shape[0]}] dtype={getattr(obj, 'dtype', '?')}")
    if sp is not None and hasattr(sp, "issparse") and sp.issparse(obj):  # type: ignore
        parts.append(f"sparse[{obj.shape[0]}x{obj.shape[1]}] nnz={obj.nnz}, fmt={obj.getformat()}")
    if nx is not None and isinstance(obj, nx.Graph):
        parts.append(f"Graph|nodes={obj.number_of_nodes()}, edges={obj.number_of_edges()}")
    return " ".join(parts)

def _head_tail(seq, max_items):
    n = len(seq)
    if n <= max_items:
        return list(range(n)), seq
    half = max_items // 2
    idxs = list(range(half)) + list(range(n - (max_items - half), n))
    vals = []
    for i in idxs:
        try:
            vals.append(seq[i])
        except Exception:
            vals.append("<err>")
    return idxs, vals

def summarize_object(
    obj: Any,
    *,
    depth: int = 2,
    max_children: int = 8,
    max_kv_pairs: int = 12,
    max_str: int = 120,
    stream: io.StringIO | None = None,
) -> str:
    """
    Print a compact tree summary of `obj` up to `depth` levels.
    Handles dict/list/tuple, NumPy, pandas, SciPy sparse, and NetworkX graphs.
    """
    out = stream or io.StringIO()
    seen = set()

    def trunc(s: str) -> str:
        return (s if len(s) <= max_str else s[: max_str - 3] + "...").replace("\n", "\\n")

    def line(level: int, text: str):
        out.write("  " * level + text + "\n")

    def visit(x: Any, level: int, path: str):
        oid = id(x)
        if oid in seen:
            line(level, f"↩ {path}: {_typename(x)} (cycle)")
            return
        seen.add(oid)

        tname = _typename(x)
        meta = _shape_and_len(x)
        size = _sizeof(x)
        header = f"{path}: {tname}"
        if meta:
            header += f" [{meta}]"
        if size:
            header += f" ~{size}B"
        line(level, header)

        if level >= depth:
            try:
                preview = trunc(repr(x))
                line(level, f"  └ preview: {preview}")
            except Exception:
                pass
            return

        # pandas
        if pd is not None and isinstance(x, pd.DataFrame):
            cols = list(x.columns)
            show = cols[:max_children]
            line(level, f"  ├ columns({len(cols)}): {show}{' ...' if len(cols)>len(show) else ''}")
            dtypes = {c: str(x[c].dtype) for c in show}
            line(level, f"  ├ dtypes(sample): {dtypes}")
            try:
                sample = x.head(min(3, len(x)))
                line(level, f"  └ head(3):")
                for r in sample.itertuples(index=False):
                    line(level+1, trunc(str(r)))
            except Exception:
                pass
            return

        if pd is not None and isinstance(x, pd.Series):
            line(level, f"  ├ dtype: {getattr(x,'dtype','?')}")
            try:
                line(level, f"  └ head(5): {trunc(str(x.head(5)))}")
            except Exception:
                pass
            return

        # numpy
        if np is not None and isinstance(x, np.ndarray):
            try:
                line(level, f"  └ sample: {trunc(np.array2string(x.take(range(min(6, x.size)), mode='raise')))}")
            except Exception:
                pass
            return

        # scipy sparse
        if sp is not None and hasattr(sp, "issparse") and sp.issparse(x):  # type: ignore
            try:
                line(level, f"  └ sample nnz (up to 10): {trunc(str(list(zip(x.nonzero()[0][:10], x.nonzero()[1][:10]))))}")
            except Exception:
                pass
            return

        # networkx
        if nx is not None and isinstance(x, nx.Graph):
            try:
                nodes = list(x.nodes())[:max_children]
                edges = list(x.edges())[:max_children]
                line(level, f"  ├ nodes(sample {len(nodes)}): {trunc(str(nodes))}{' ...' if x.number_of_nodes()>len(nodes) else ''}")
                line(level, f"  └ edges(sample {len(edges)}): {trunc(str(edges))}{' ...' if x.number_of_edges()>len(edges) else ''}")
            except Exception:
                pass
            return

        # mappings
        if isinstance(x, Mapping):
            keys = list(x.keys())
            n = len(keys)
            keys_sorted = sorted(map(str, keys)) if n<=2000 else list(map(str, keys))
            keys_show = keys_sorted[:max_kv_pairs]
            line(level, f"  ├ keys({n}): {keys_show}{' ...' if n>len(keys_show) else ''}")
            for k in keys_show:
                try:
                    v = x[k if k in x else next(orig for orig in x.keys() if str(orig)==k)]
                except Exception:
                    continue
                visit(v, level + 1, f"[{repr(k)}]")
            return

        # sequences (but not str/bytes)
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
            try:
                n = len(x)
                idxs, vals = _head_tail(x, max_children)
                line(level, f"  ├ seq({n}) showing {len(idxs)}")
                for i, v in zip(idxs, vals):
                    visit(v, level + 1, f"[{i}]")
            except Exception:
                pass
            return

        # dataclass-like (__dict__)
        if hasattr(x, "__dict__"):
            try:
                attrs = list(vars(x).keys())
                show = attrs[:max_children]
                line(level, f"  ├ attrs({len(attrs)}): {show}{' ...' if len(attrs)>len(show) else ''}")
                for name in show:
                    try:
                        visit(getattr(x, name), level + 1, f".{name}")
                    except Exception:
                        pass
            except Exception:
                pass
            return

        # fallback
        try:
            line(level, f"  └ value: {trunc(repr(x))}")
        except Exception:
            pass

    visit(obj, 0, path="root")
    return out.getvalue()

# ================================================================
# EPANET INP — section scanner (counts + line ranges)
# ================================================================

_INP_SECTION_RE = re.compile(r"^\s*\[(?P<name>[A-Za-z_]+)\]\s*$")

def scan_inp_sections(path: str | os.PathLike) -> dict[str, dict]:
    """
    Return {section: {'count': <data rows>, 'first_line': int, 'last_line': int}}
    counting non-empty, non-comment lines within each section.
    """
    info: dict[str, dict] = {}
    current: Optional[str] = None
    line_no = 0
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line_no += 1
            m = _INP_SECTION_RE.match(raw)
            if m:
                current = m.group("name").upper()
                info.setdefault(current, {"count": 0, "first_line": line_no + 1, "last_line": line_no})
                continue
            if current is None:
                continue
            s = raw.strip()
            if not s or s.startswith(";"):
                continue
            info[current]["count"] += 1
            info[current]["last_line"] = line_no
    return info

def pretty_print_inp_sections(info: dict[str, dict]) -> str:
    if not info:
        return "No sections detected."
    width = max(len(k) for k in info.keys())
    lines = ["EPANET .inp sections:"]
    for sec in sorted(info.keys()):
        d = info[sec]
        lines.append(f"  [{sec:<{width}}]  rows={d['count']:<8}  lines {d['first_line']}-{d['last_line']}")
    return "\n".join(lines)

# ================================================================
# EPANET INP — heuristics & helpers
# ================================================================

def _tokenize_inp_line(s: str) -> List[str]:
    """
    Tokenize a data line, preserving quoted strings and stripping inline comments.
    """
    if ';' in s:
        before, _, _ = s.partition(';')
        s = before
    s = s.strip()
    if not s:
        return []
    try:
        return shlex.split(s, posix=True)
    except Exception:
        return s.split()

def _normalize_header_name(h: str) -> str:
    h = h.strip()
    h = h.replace("&", " and ")
    h = re.sub(r"\s+", "_", h)
    h = re.sub(r"[^0-9A-Za-z_]+", "", h)
    replacements = {"XCoord": "X", "YCoord": "Y", "Label_and_Anchor_Node": "Label_Anchor_Node"}
    return replacements.get(h, h)

# Treat an "aligned gap" as either >=1 tab OR >=2 spaces
_HEADER_GAP_RE = re.compile(r"(?:\t+| {2,})")
_KV_GAP_RE     = re.compile(r"(?:\t+| {2,})")

def _split_header_comment(header_comment: str) -> List[str]:
    """
    Split comment header into column names by aligned gaps (tabs or >=2 spaces),
    keeping multi-word headers intact (e.g., 'Label & Anchor Node').
    """
    s = header_comment.lstrip(";").strip()
    parts = [p for p in _HEADER_GAP_RE.split(s) if p]
    if len(parts) >= 2:
        return [_normalize_header_name(p) for p in parts]
    # fallback: tokenization
    return [_normalize_header_name(h) for h in _tokenize_inp_line(s)]

def _split_aligned_kv_pair(raw: str) -> Optional[tuple[str, str]]:
    """
    Split a line into (key, rhs) at the first aligned gap (>=1 tab OR >=2 spaces).
    Removes inline comments. Returns None if no aligned gap is found.
    """
    if ';' in raw:
        raw = raw.split(';', 1)[0]
    s = raw.rstrip()
    if not s.strip():
        return None
    m = _KV_GAP_RE.search(s)
    if not m:
        return None
    key = s[:m.start()].strip()
    rhs = s[m.end():].strip()
    return key, rhs

def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False

def _infer_kind_and_columns(
    data_lines: List[Tuple[int, str]],
    header_comment: Optional[str]
) -> Tuple[str, List[str], List[List[str]], List[Tuple[int, List[str]]], str]:
    """
    Heuristically infer a section as a named table, inferred-width table, or key/value block.
    Returns (kind, columns, parsed_rows, samples, note)
    """
    note = ""
    parsed: List[Tuple[int, List[str]]] = []
    for ln, raw in data_lines:
        s = raw.strip()
        if s.startswith(";") or not s:
            continue
        toks = _tokenize_inp_line(raw)
        if toks:
            parsed.append((ln, toks))

    # Header-based table
    if header_comment:
        columns = _split_header_comment(header_comment)
        return "table", columns, [t for _, t in parsed], parsed[:3], "header-derived"

    if not parsed:
        return "unknown", [], [], [], note

    # Compute per-row stats
    lengths = [len(t) for _, t in parsed]
    unique_len = len(set(lengths))
    first_nonnum_ratio = sum(1 for _, t in parsed if len(t) >= 1 and not _is_number(t[0])) / len(parsed)

    # KEY–VALUE heuristic: non-numeric first token on most lines + variable row widths
    if first_nonnum_ratio >= 0.8 and unique_len >= 2:
        max_vals = max(0, max(lengths) - 1)
        columns = ["KEY"] + [f"VALUE{i}" for i in range(1, max_vals + 1)]
        parsed_rows = []
        for _, t in parsed:
            vals = t[1:]
            row = [t[0]] + vals[:max_vals] + [""] * max(0, max_vals - len(vals))
            parsed_rows.append(row)
        return "kv", columns, parsed_rows, parsed[:3], f"inferred kv (variable widths: {sorted(set(lengths))})"

    # TABLE-INFERRED heuristic: fixed width by most common token length
    lens = Counter(lengths)
    col_count, _ = max(lens.items(), key=lambda kv: (kv[1], kv[0]))
    id_like = sum(1 for _, t in parsed if len(t) >= 1 and not _is_number(t[0])) >= 0.8 * len(parsed)
    columns = (["ID"] if id_like else ["col1"]) + [f"col{i}" for i in range(2, col_count + 1)]
    parsed_rows = [t[:col_count] for _, t in parsed]
    return "table_inferred", columns, parsed_rows, parsed[:3], f"inferred width={col_count}"

def parse_inp_with_schema(path: str | os.PathLike, max_samples: int = 3) -> Dict[str, Dict[str, Any]]:
    """
    Parse an INP file at a schema level (without converting to DataFrames),
    returning per-section kind/columns/row counts and a few sample rows.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    result: Dict[str, Dict[str, Any]] = {}
    current: Optional[str] = None
    header_seen_for_current: Optional[str] = None
    buffer: List[Tuple[int, str]] = []
    line_no = 0

    def flush_section():
        nonlocal buffer, header_seen_for_current, current
        if current is None:
            buffer = []
            header_seen_for_current = None
            return
        kind, cols, parsed_rows, samples, note = _infer_kind_and_columns(buffer, header_seen_for_current)
        result[current] = {
            "kind": kind,
            "columns": cols,
            "rows": len(parsed_rows),
            "header_comment": header_seen_for_current,
            "samples": [{"lineno": ln, "values": vals} for ln, vals in samples[:max_samples]],
            "note": note,
        }
        buffer = []
        header_seen_for_current = None

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line_no += 1
            m = _INP_SECTION_RE.match(raw)
            if m:
                flush_section()
                current = m.group("name").upper()
                result.setdefault(current, {})
                buffer = []
                header_seen_for_current = None
                continue

            if current is None:
                continue

            s = raw.strip()
            if header_seen_for_current is None and s.startswith(";") and len(s) > 1:
                header_seen_for_current = s
                continue

            buffer.append((line_no, raw.rstrip("\n")))

    flush_section()
    return result

# ================================================================
# Filesystem conveniences
# ================================================================

def summarize_pathlike(pathlike: Any) -> str:
    """
    Directory: list entries (up to 100).
    File: if .inp -> section counts; else size + first KB preview.
    """
    try:
        p = Path(pathlike)
    except Exception:
        return f"(not a pathlike: {pathlike!r})"
    if not p.exists():
        return f"(path not found) {p}"
    if p.is_dir():
        entries = list(p.iterdir())
        parts = [f"Directory: {p} ({len(entries)} items)"]
        for e in entries[:100]:
            try:
                size = e.stat().st_size if e.is_file() else 0
                parts.append(f"  - {e.name} {'(dir)' if e.is_dir() else f'[{size} B]'}")
            except Exception:
                parts.append(f"  - {e.name} (stat error)")
        if len(entries) > 100:
            parts.append("  ...")
        return "\n".join(parts)
    # file
    if p.suffix.lower() == ".inp":
        try:
            info = scan_inp_sections(p)
            return pretty_print_inp_sections(info)
        except Exception as e:
            return f"Failed to scan INP: {e}"
    try:
        sz = p.stat().st_size
        with open(p, "rb") as f:
            preview = f.read(min(1024, sz)).decode("utf-8", errors="replace")
        return f"File: {p} [{sz} B]\n--- first {min(1024, sz)} bytes ---\n{preview}"
    except Exception as e:
        return f"File summary error: {e}"

# ================================================================
# Section helpers & parsers
# ================================================================

def _rows_for_section(path: str | os.PathLike, target_section: str) -> Tuple[Optional[str], List[List[str]]]:
    """
    Return (header_comment, rows_tokens) for a given section (tokenized rows).
    """
    target = target_section.upper()
    header_comment: Optional[str] = None
    rows: List[List[str]] = []
    current: Optional[str] = None

    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            m = _INP_SECTION_RE.match(raw)
            if m:
                current = m.group("name").upper()
                continue
            if current != target:
                continue
            s = raw.strip()
            if not s:
                continue
            if s.startswith(";"):
                if header_comment is None and len(s) > 1:
                    header_comment = s
                continue
            toks = _tokenize_inp_line(raw)
            if toks:
                rows.append(toks)
    return header_comment, rows

def _raw_rows_for_section(path: str | os.PathLike, target_section: str) -> List[str]:
    """
    Return raw (non-comment, non-empty) lines for a given section.
    """
    target = target_section.upper()
    out: List[str] = []
    current: Optional[str] = None
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            m = _INP_SECTION_RE.match(raw)
            if m:
                current = m.group("name").upper()
                continue
            if current != target:
                continue
            s = raw.strip("\n")
            if not s.strip():
                continue
            if s.lstrip().startswith(";"):
                continue
            out.append(s)
    return out

# ---------------- [CONTROLS] grammar parser ----------------

def _parse_controls_rows(rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """
    Parse EPANET [CONTROLS] into named columns.

    Recognized patterns (practical subset):
      - LINK <id> (open|closed) IF NODE <id> (ABOVE|BELOW) <val>
      - PUMP <id> setting <val> IF NODE <id> (ABOVE|BELOW) <val>
      - LINK <id> (open|closed) AT TIME <hours>
      - LINK <id> (open|closed) AT CLOCKTIME <HH:MM>
    """
    columns = [
        "TargetKind",   # LINK / PUMP / VALVE / TANK
        "TargetID",
        "Action",       # open / closed / setting / level / status
        "ActionValue",  # optional numeric after 'setting'/'level'
        "WhenType",     # IF / AT
        "CondKind",     # NODE / TANK / SYSTEM (for IF)
        "CondID",       # e.g., T_Zone
        "Comparator",   # ABOVE / BELOW
        "Threshold",    # numeric threshold
        "Time",         # hours for AT TIME
        "ClockTime",    # HH:MM for AT CLOCKTIME
    ]

    def rowdict_to_list(d: dict) -> List[str]:
        return [str(d.get(c, "")) if d.get(c, "") is not None else "" for c in columns]

    parsed_rows: List[List[str]] = []

    def is_num(s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False

    for toks in rows:
        d = {c: "" for c in columns}
        T = toks[:]
        U = [t.upper() for t in T]

        try:
            if len(T) >= 3 and U[0] in {"LINK", "PUMP", "VALVE", "TANK"}:
                d["TargetKind"] = U[0]
                d["TargetID"]   = T[1]
                # Action + optional value
                if U[2] in {"OPEN", "CLOSED"}:
                    d["Action"] = T[2].lower()
                    j = 3
                elif U[2] in {"SETTING", "LEVEL", "STATUS"}:
                    d["Action"] = T[2].lower()
                    if len(T) > 3 and is_num(T[3]):
                        d["ActionValue"] = T[3]
                        j = 4
                    else:
                        j = 3
                else:
                    d["Action"] = T[2]
                    j = 3

                # WHEN: IF ...
                if len(T) > j and U[j] == "IF":
                    d["WhenType"] = "IF"
                    if len(T) > j+1:
                        d["CondKind"] = U[j+1]
                    if len(T) > j+2:
                        d["CondID"] = T[j+2]
                    if len(T) > j+3 and U[j+3] in {"ABOVE", "BELOW"}:
                        d["Comparator"] = U[j+3]
                    if len(T) > j+4 and is_num(T[j+4]):
                        d["Threshold"] = T[j+4]
                    parsed_rows.append(rowdict_to_list(d))
                    continue

                # WHEN: AT TIME <hours>
                if len(T) > j+2 and U[j] == "AT" and U[j+1] == "TIME":
                    if len(T) > j+2 and is_num(T[j+2]):
                        d["WhenType"] = "AT"
                        d["Time"] = T[j+2]
                        parsed_rows.append(rowdict_to_list(d))
                        continue

                # WHEN: AT CLOCKTIME <HH:MM>
                if len(T) > j+2 and U[j] == "AT" and U[j+1] == "CLOCKTIME":
                    d["WhenType"] = "AT"
                    d["ClockTime"] = T[j+2] if len(T) > j+2 else ""
                    parsed_rows.append(rowdict_to_list(d))
                    continue

                # fallback: still record TargetKind/ID/Action
                parsed_rows.append(rowdict_to_list(d))
                continue

        except Exception:
            pass  # fall through to fallback

        # Generic fallback
        if T:
            d["TargetKind"] = U[0]
        if len(T) > 1:
            d["TargetID"] = T[1]
        if len(T) > 2:
            d["Action"] = T[2]
        if "IF" in U:
            d["WhenType"] = "IF"
            k = U.index("IF")
            if len(T) > k+1:
                d["CondKind"] = U[k+1]
            if len(T) > k+2:
                d["CondID"] = T[k+2]
            if len(T) > k+3 and U[k+3] in {"ABOVE", "BELOW"}:
                d["Comparator"] = U[k+3]
            if len(T) > k+4 and is_num(T[k+4]):
                d["Threshold"] = T[k+4]
        elif "AT" in U:
            d["WhenType"] = "AT"
            k = U.index("AT")
            if len(T) > k+2 and U[k+1] == "TIME":
                d["Time"] = T[k+2]
            elif len(T) > k+2 and U[k+1] == "CLOCKTIME":
                d["ClockTime"] = T[k+2]
        parsed_rows.append(rowdict_to_list(d))

    return columns, parsed_rows

# ---------------- [BACKDROP] row-by-row named columns ----------------

def _parse_backdrop_rows(rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """
    Parse EPANET [BACKDROP] into named columns.
    """
    columns = ["Key", "Xmin", "Ymin", "Xmax", "Ymax", "Units", "File", "Xoff", "Yoff"]
    parsed: List[List[str]] = []

    for toks in rows:
        if not toks:
            continue
        key = toks[0].upper()
        vals = toks[1:]

        row = {
            "Key": key, "Xmin": "", "Ymin": "", "Xmax": "", "Ymax": "",
            "Units": "", "File": "", "Xoff": "", "Yoff": ""
        }

        if key == "DIMENSIONS":
            if len(vals) > 0: row["Xmin"] = vals[0]
            if len(vals) > 1: row["Ymin"] = vals[1]
            if len(vals) > 2: row["Xmax"] = vals[2]
            if len(vals) > 3: row["Ymax"] = vals[3]
        elif key == "UNITS":
            if vals: row["Units"] = vals[0]
        elif key == "FILE":
            row["File"] = " ".join(vals).strip()
        elif key == "OFFSET":
            if len(vals) > 0: row["Xoff"] = vals[0]
            if len(vals) > 1: row["Yoff"] = vals[1]
        else:
            row["File"] = " ".join(vals).strip()

        parsed.append([row[c] for c in columns])

    return columns, parsed

# ---------------- Generic "aligned KV" parsers ----------------

def _parse_kv_aligned_rows(raw_lines: List[str]) -> Tuple[List[str], List[List[str]]]:
    """
    For sections where each line is "Key<aligned gap>Value(s)" and keys may contain spaces.
    We split on aligned gap (>=1 tab OR >=2 spaces). Values are tokenized via shlex.
    Returns (columns, rows) with columns = ["Key", "Value1"..].
    """
    rows_parsed: List[Tuple[str, List[str]]] = []
    max_vals = 0

    for raw in raw_lines:
        pair = _split_aligned_kv_pair(raw)
        if pair:
            key, rhs = pair
            try:
                toks = shlex.split(rhs, posix=True)
            except Exception:
                toks = rhs.split()
        else:
            s = raw.strip()
            if not s:
                continue
            parts = s.split()
            key = parts[0]
            toks = parts[1:]

        rows_parsed.append((key, toks))
        max_vals = max(max_vals, len(toks))

    columns = ["Key"] + [f"Value{i}" for i in range(1, max_vals + 1)]
    matrix: List[List[str]] = []
    for key, toks in rows_parsed:
        matrix.append([key] + toks + [""] * (max_vals - len(toks)))
    return columns, matrix

def _parse_kv_aligned_single_value_rows(raw_lines: List[str]) -> Tuple[List[str], List[List[str]]]:
    """
    Split on aligned gap (tabs or >=2 spaces), but keep the RHS as one string value.
    Useful for [OPTIONS]: 'Unbalanced  Continue 10' -> Value='Continue 10'
    """
    rows: List[List[str]] = []
    for raw in raw_lines:
        pair = _split_aligned_kv_pair(raw)
        if pair:
            key, val = pair
            rows.append([key, val])
        else:
            s = raw.strip()
            if not s:
                continue
            parts = s.split()
            key = parts[0]
            val = " ".join(parts[1:]) if len(parts) > 1 else ""
            rows.append([key, val])
    return ["Key", "Value"], rows

# ---------------- [PATTERNS] parser (ID + multipliers per line) ----------------

def _parse_patterns_rows(rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """
    Each line becomes: ID, M1..Mk (k inferred from widest line).
    """
    if not rows:
        return ["ID"], []

    max_vals = max(len(r) - 1 for r in rows if r)  # number of multipliers on the widest line
    max_vals = max(max_vals, 1)

    cols = ["ID"] + [f"M{i}" for i in range(1, max_vals + 1)]
    out: List[List[str]] = []
    for r in rows:
        pid = r[0] if r else ""
        vals = r[1:]
        out.append([pid] + vals[:max_vals] + [""] * (max_vals - len(vals)))
    return cols, out

# ---------------- [PUMPS] parser (keeps curve IDs etc.) ----------------

def _parse_pumps_rows(rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    """
    Parse EPANET [PUMPS] lines into:
      ID, Node1, Node2, ParamType, ParamArgs, CurveID, Power, Speed
    """
    cols = ["ID", "Node1", "Node2", "ParamType", "ParamArgs", "CurveID", "Power", "Speed"]
    out: List[List[str]] = []

    def is_num(s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False

    for r in rows:
        if not r:
            continue
        d = {c: "" for c in cols}
        d["ID"]    = r[0] if len(r) > 0 else ""
        d["Node1"] = r[1] if len(r) > 1 else ""
        d["Node2"] = r[2] if len(r) > 2 else ""
        rest = r[3:] if len(r) > 3 else []

        if rest:
            ptype = rest[0].upper()
            args  = rest[1:]
            d["ParamType"] = ptype
            d["ParamArgs"] = " ".join(args)

            if ptype == "HEAD":
                if args:
                    d["CurveID"] = args[0]
            elif ptype == "POWER":
                if args and is_num(args[0]):
                    d["Power"] = args[0]
            elif ptype == "SPEED":
                if args and is_num(args[0]):
                    d["Speed"] = args[0]
            # Other parameter styles will still be visible in ParamArgs

        out.append([d[c] for c in cols])

    return cols, out

# ================================================================
# Section -> DataFrame
# ================================================================

def section_to_dataframe(path: str | os.PathLike, section: str):
    """
    Convert a section to pandas DataFrame using header comment if present,
    or infer a schema. Special-cases:
      - [CONTROLS] : grammar-based named columns
      - [BACKDROP] : named columns (Xmin..)
      - [OPTIONS]  : aligned KV with a single Value string
      - [REACTIONS], [REPORT], [TIMES], [ENERGY] : aligned KV (Key, Value1..)
      - [PATTERNS] : ID + multipliers per line (M1..Mk)
      - [PUMPS]    : ID/Node1/Node2/ParamType/ParamArgs/CurveID/Power/Speed
    """
    if pd is None:
        raise ImportError("pandas is required for section_to_dataframe")

    sec = section.upper()

    # Special-cases first
    if sec == "CONTROLS":
        _, rows = _rows_for_section(path, sec)
        cols, norm_rows = _parse_controls_rows(rows)
        return pd.DataFrame(norm_rows, columns=cols)

    if sec == "BACKDROP":
        _, rows = _rows_for_section(path, sec)
        cols, norm_rows = _parse_backdrop_rows(rows)
        return pd.DataFrame(norm_rows, columns=cols)

    # OPTIONS -> single Value string (preserve 'Continue 10', 'NONE mg/L', etc.)
    if sec == "OPTIONS":
        raw_lines = _raw_rows_for_section(path, sec)
        cols, norm_rows = _parse_kv_aligned_single_value_rows(raw_lines)
        return pd.DataFrame(norm_rows, columns=cols)

    if sec in {"REACTIONS", "REPORT", "TIMES", "ENERGY"}:
        raw_lines = _raw_rows_for_section(path, sec)
        cols, norm_rows = _parse_kv_aligned_rows(raw_lines)
        return pd.DataFrame(norm_rows, columns=cols)

    if sec == "PATTERNS":
        _, rows = _rows_for_section(path, sec)
        cols, norm_rows = _parse_patterns_rows(rows)
        return pd.DataFrame(norm_rows, columns=cols)

    if sec == "PUMPS":
        _, rows = _rows_for_section(path, sec)
        cols, norm_rows = _parse_pumps_rows(rows)
        return pd.DataFrame(norm_rows, columns=cols)

    # Generic path
    header_comment, rows = _rows_for_section(path, sec)
    if not rows:
        return pd.DataFrame()

    # Header present -> table with named columns
    if header_comment:
        hdr = _split_header_comment(header_comment)
        width = len(hdr)
        norm_rows = [r[:width] + [""]*(width - len(r)) if len(r) < width else r[:width] for r in rows]
        return pd.DataFrame(norm_rows, columns=hdr)

    # No header -> decide kv vs table by heuristic
    lengths = [len(r) for r in rows if r]
    unique_len = len(set(lengths)) if lengths else 0
    first_nonnum_ratio = sum(1 for r in rows if r and not _is_number(r[0])) / len(rows) if rows else 0.0

    # kv if first token non-numeric in most rows AND variable lengths
    if first_nonnum_ratio >= 0.8 and unique_len >= 2:
        max_vals = max(0, max(lengths) - 1) if lengths else 0
        cols = ["KEY"] + [f"VALUE{i}" for i in range(1, max_vals + 1)]
        norm_rows = []
        for r in rows:
            key = r[0] if r else ""
            vals = r[1:] if len(r) > 1 else []
            norm_rows.append([key] + vals[:max_vals] + [""] * max(0, max_vals - len(vals)))
        return pd.DataFrame(norm_rows, columns=cols)

    # else treat as fixed-width table
    if lengths:
        col_count, _ = max(Counter(lengths).items(), key=lambda kv: (kv[1], kv[0]))
    else:
        col_count = 1
    id_like = sum(1 for r in rows if r and not _is_number(r[0])) >= 0.8 * len(rows)
    cols = (["ID"] if id_like else ["col1"]) + [f"col{i}" for i in range(2, col_count + 1)]
    norm_rows = [r[:col_count] + [""]*(col_count - len(r)) if len(r) < col_count else r[:col_count] for r in rows]
    return pd.DataFrame(norm_rows, columns=cols)

def save_section_csv(path: str | os.PathLike, section: str, out_csv: str | os.PathLike) -> str:
    """
    Save a given section to CSV. Returns output path as string.
    """
    if pd is None:
        raise ImportError("pandas is required for save_section_csv")
    df = section_to_dataframe(path, section)
    out_csv = str(out_csv)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv

# ================================================================
# Pretty schema summary
# ================================================================

def describe_inp_structure(path: str | os.PathLike, max_samples: int = 3) -> str:
    """
    Pretty-print per-section schema summary with sample rows.
    Upgrades [CONTROLS], [BACKDROP], [OPTIONS], [REACTIONS]/[REPORT]/[TIMES]/[ENERGY],
    [PATTERNS], and [PUMPS] to their richer kinds.
    """
    schema = parse_inp_with_schema(path, max_samples=max_samples)

    # CONTROLS
    try:
        hdr_c, rows_c = _rows_for_section(path, "CONTROLS")
        if rows_c:
            cols_c, norm_rows_c = _parse_controls_rows(rows_c)
            schema["CONTROLS"] = {
                "kind": "controls", "columns": cols_c, "rows": len(norm_rows_c),
                "header_comment": hdr_c, "samples": [{"lineno": None, "values": r} for r in norm_rows_c[:max_samples]],
                "note": "grammar-based",
            }
    except Exception:
        pass

    # BACKDROP
    try:
        hdr_b, rows_b = _rows_for_section(path, "BACKDROP")
        if rows_b:
            cols_b, norm_rows_b = _parse_backdrop_rows(rows_b)
            schema["BACKDROP"] = {
                "kind": "backdrop", "columns": cols_b, "rows": len(norm_rows_b),
                "header_comment": hdr_b, "samples": [{"lineno": None, "values": r} for r in norm_rows_b[:max_samples]],
                "note": "kv→named",
            }
    except Exception:
        pass

    # OPTIONS -> single Value column
    try:
        raw_opt = _raw_rows_for_section(path, "OPTIONS")
        if raw_opt:
            cols_o, rows_o = _parse_kv_aligned_single_value_rows(raw_opt)
            schema["OPTIONS"] = {
                "kind": "kv_aligned_single", "columns": cols_o, "rows": len(rows_o),
                "header_comment": None, "samples": [{"lineno": None, "values": r} for r in rows_o[:max_samples]],
                "note": "aligned key/value (single Value string)",
            }
    except Exception:
        pass

    # Other aligned KV: REACTIONS, REPORT, TIMES, ENERGY
    for sec in ("REACTIONS", "REPORT", "TIMES", "ENERGY"):
        try:
            raw_lines = _raw_rows_for_section(path, sec)
            if raw_lines:
                cols_kv, rows_kv = _parse_kv_aligned_rows(raw_lines)
                schema[sec] = {
                    "kind": "kv_aligned", "columns": cols_kv, "rows": len(rows_kv),
                    "header_comment": None, "samples": [{"lineno": None, "values": r} for r in rows_kv[:max_samples]],
                    "note": "aligned key/value",
                }
        except Exception:
            pass

    # PATTERNS
    try:
        hdr_p, rows_p = _rows_for_section(path, "PATTERNS")
        if rows_p:
            cols_p, norm_rows_p = _parse_patterns_rows(rows_p)
            schema["PATTERNS"] = {
                "kind": "patterns", "columns": cols_p, "rows": len(norm_rows_p),
                "header_comment": hdr_p, "samples": [{"lineno": None, "values": r} for r in norm_rows_p[:max_samples]],
                "note": "ID + multipliers per line",
            }
    except Exception:
        pass

    # PUMPS
    try:
        hdr_pm, rows_pm = _rows_for_section(path, "PUMPS")
        if rows_pm:
            cols_pm, norm_rows_pm = _parse_pumps_rows(rows_pm)
            schema["PUMPS"] = {
                "kind": "pumps", "columns": cols_pm, "rows": len(norm_rows_pm),
                "header_comment": hdr_pm, "samples": [{"lineno": None, "values": r} for r in norm_rows_pm[:max_samples]],
                "note": "ParamType + CurveID/Power/Speed",
            }
    except Exception:
        pass

    # Pretty print
    lines = [f"Schema for: {Path(path)}"]
    width = max(len(k) for k in schema.keys()) if schema else 0
    for sec in sorted(schema.keys()):
        meta = schema[sec]
        kind = meta.get("kind", "unknown")
        rows = meta.get("rows", 0)
        cols = meta.get("columns", [])
        note = meta.get("note", "")
        lines.append(f"\n[{sec:<{width}}]  kind={kind:<20} rows={rows:<7} columns={cols or '—'}{('  ('+note+')') if note else ''}")
        hc = meta.get("header_comment")
        if hc:
            lines.append(f"  header: {hc.lstrip(';').strip()}")
        samples = meta.get("samples", [])
        for s in samples[:max_samples]:
            ln = s.get("lineno", None)
            if ln is None:
                lines.append(f"  sample       : {s['values']}")
            else:
                lines.append(f"  sample L{str(ln).rjust(6)}: {s['values']}")
    return "\n".join(lines)

# ================================================================
# CLI
# ================================================================

def _cli():
    ap = argparse.ArgumentParser(description="EPANET INP schema parser & object summarizer")
    ap.add_argument("--inp", type=str, help="Path to .inp file")
    ap.add_argument("--show-sections", action="store_true", help="Show section counts and line ranges")
    ap.add_argument("--show-schema", action="store_true", help="Show per-section schema + samples")
    ap.add_argument("--export", nargs=2, metavar=("SECTION", "CSV"), help="Export a section to CSV (requires pandas)")
    args = ap.parse_args()

    if args.inp is None:
        ap.print_help(); sys.exit(0)

    if args.show_sections:
        print(pretty_print_inp_sections(scan_inp_sections(args.inp)))

    if args.show_schema:
        print(describe_inp_structure(args.inp, max_samples=3))

    if args.export:
        section, out_csv = args.export
        try:
            path = save_section_csv(args.inp, section, out_csv)
            print(f"Saved {section} -> {path}")
        except Exception as e:
            print(f"Export failed: {e}", file=sys.stderr); sys.exit(2)

if __name__ == "__main__":
    _cli()
