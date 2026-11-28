"""The parameter filter the Models and Backtests pages share (written once;
the old dashboard had two copies). Pure functions plus one widget."""

import math
from typing import Dict, Iterable, Mapping, Tuple

import streamlit as st

Ranges = Dict[str, Tuple[float, float]]


def _number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def numeric_fields(params: Iterable[Mapping]) -> Dict[str, Tuple[float, float]]:
    """Each numeric key's (min, max) over the finite values (NaN is skipped)."""
    seen: Dict[str, list] = {}
    for p in params:
        for key, value in p.items():
            if _number(value):
                seen.setdefault(key, []).append(float(value))
    return {key: (min(v), max(v)) for key, v in seen.items()}


def keep(params: Mapping, ranges: Ranges, text: str, haystack: str = "") -> bool:
    """True if every ranged key that ``params`` has lies inside its range, and
    ``text`` (lower case) is in ``haystack`` + the params. A NaN or non-number
    value of a ranged key fails (``not lo <= v <= hi``)."""
    for key, (lo, hi) in ranges.items():
        if key not in params:
            continue
        value = params[key]
        if not _number(value) or not lo <= value <= hi:
            return False
    return not text or text in f"{haystack} {params}".lower()


def parameter_filter(params: Iterable[Mapping], key: str) -> Tuple[Ranges, str]:
    """Sliders for every numeric parameter that varies, plus a text search.
    Returns the chosen ranges (only those narrowed) and the search text."""
    fields = numeric_fields(params)
    ranges: Ranges = {}
    with st.expander("Parameter filters", expanded=False):
        varying = {k: b for k, b in fields.items() if b[0] < b[1]}
        columns = st.columns(3)
        for i, (name, (lo, hi)) in enumerate(sorted(varying.items())):
            with columns[i % 3]:
                chosen = st.slider(name, lo, hi, (lo, hi), key=f"{key}_{name}")
            if chosen != (lo, hi):
                ranges[name] = chosen
        fixed = sorted(k for k, b in fields.items() if b[0] == b[1])
        if fixed:
            st.caption(
                "Same in every row: "
                + ", ".join(f"{k} = {fields[k][0]:g}" for k in fixed)
            )
        text = st.text_input("Search (id or parameters)", key=f"{key}_text")
    return ranges, text.strip().lower()
