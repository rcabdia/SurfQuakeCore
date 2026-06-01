# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# SurfQuakeCore – Seismic Trace Algebra
# ------------------------------------------------------------------
# Licence : GPL-3.0  (same as SurfQuakeCore)
# Author  : SurfQuakeCore contributors
# ------------------------------------------------------------------
"""
Provides :class:`TraceAlgebra`, a safe expression evaluator that maps
user-supplied algebraic strings onto ObsPy :class:`~obspy.core.stream.Stream`
traces and returns a new :class:`~obspy.core.stream.Stream` with the result.

Pre-processing pipeline (all opt-in via constructor keywords)
-------------------------------------------------------------
The three steps run in this fixed order on a **deep copy** of the stream,
leaving the original untouched:

1. **trim** (``trim=True``)
   Cuts every trace to the common time window
   [max(starttimes), min(endtimes)].  Raises :class:`TraceAlgebraError`
   if the traces do not overlap at all.

2. **fill_gaps** (``fill_gaps=True``)
   Merges the stream (``Stream.merge``) and fills any internal gap or
   overlap using linear interpolation (``fill_value="interpolate"``).

3. **resample** (``resample=True``)
   Resamples all traces to the **highest** sampling rate found in the
   stream using ``Trace.resample``.  Pass ``resample_to`` to pin a
   specific target rate instead.

Supported expression syntax
----------------------------
* Arithmetic operators  : ``+  -  *  /  **``
* Comparison operators  : ``>  <  >=  <=  ==  !=``  (return 0/1 arrays)
* Unary minus           : ``-tr1``
* Math functions        : ``abs, sqrt, exp, log, log2, log10,``
                          ``sin, cos, tan, arcsin, arccos, arctan, arctan2,``
                          ``sinh, cosh, tanh,``
                          ``ceil, floor, round,``
                          ``sign, diff, cumsum, gradient, real, imag``
* Constants             : ``pi, e``
* Parentheses           : fully supported

Trace naming
------------
Traces are referenced by positional alias ``tr1, tr2, …, trN``
(where ``tr1`` = ``stream[0]``, ``tr2`` = ``stream[1]``, etc.)
**or** by their SEED id ``NET.STA.LOC.CHA`` with dots replaced by
underscores, e.g. ``BW_RJOB__EHZ``.

Examples
--------
>>> from obspy import read
>>> from trace_algebra import TraceAlgebra
>>> st = read()          # three-component demo stream
>>>
>>> # all pre-processing steps enabled
>>> ta = TraceAlgebra(st, trim=True, fill_gaps=True, resample=True)
>>>
>>> # simple sum
>>> result = ta.evaluate("tr1 + tr2")
>>>
>>> # complex expression
>>> result = ta.evaluate("exp(tr1) + sin(tr2) * sqrt(abs(tr3))")
>>>
>>> # SEED-id reference
>>> result = ta.evaluate("BW_RJOB__EHZ * 1e3")
>>>
>>> # multi-output: comma-separated produces a multi-trace stream
>>> result = ta.evaluate("tr1 + tr2, tr1 - tr2")
>>>
>>> # force a specific target sampling rate
>>> ta2 = TraceAlgebra(st, resample=True, resample_to=50.0)
"""

from __future__ import annotations

import ast
import copy
import logging
import math
import re
from typing import Dict, List, Optional

import numpy as np
from obspy import Stream, Trace
from obspy.core import UTCDateTime

__all__ = ["TraceAlgebra", "TraceAlgebraError"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class TraceAlgebraError(Exception):
    """Raised for any error originating inside :class:`TraceAlgebra`."""


# ---------------------------------------------------------------------------
# Safe AST evaluator
# ---------------------------------------------------------------------------

_ALLOWED_FUNCTIONS: Dict[str, callable] = {
    # basic
    "abs":      np.abs,
    "sqrt":     np.sqrt,
    "exp":      np.exp,
    "log":      np.log,
    "log2":     np.log2,
    "log10":    np.log10,
    # trigonometry
    "sin":      np.sin,
    "cos":      np.cos,
    "tan":      np.tan,
    "arcsin":   np.arcsin,
    "arccos":   np.arccos,
    "arctan":   np.arctan,
    "arctan2":  np.arctan2,
    "sinh":     np.sinh,
    "cosh":     np.cosh,
    "tanh":     np.tanh,
    # rounding
    "ceil":     np.ceil,
    "floor":    np.floor,
    "round":    np.round,
    # signal helpers
    "sign":     np.sign,
    "diff":     np.diff,
    "cumsum":   np.cumsum,
    "gradient": np.gradient,
    # complex helpers
    "real":     np.real,
    "imag":     np.imag,
}

_ALLOWED_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e":  math.e,
}

_SAFE_NODES = (
    ast.Expression,
    ast.BoolOp, ast.And, ast.Or,
    ast.BinOp,
    ast.UnaryOp, ast.UAdd, ast.USub, ast.Not,
    ast.Compare,
    ast.Call,
    ast.Constant,
    ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Tuple,
)


class _SafeEval(ast.NodeVisitor):
    def generic_visit(self, node: ast.AST) -> None:  # type: ignore[override]
        if not isinstance(node, _SAFE_NODES):
            raise TraceAlgebraError(
                f"Forbidden expression node: {type(node).__name__}. "
                "Only arithmetic / math expressions are allowed."
            )
        super().generic_visit(node)


def _check_ast(tree: ast.Expression) -> None:
    _SafeEval().visit(tree)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class TraceAlgebra:
    """Evaluate algebraic expressions over ObsPy :class:`~obspy.core.stream.Stream` traces.

    Parameters
    ----------
    stream : obspy.Stream
        Input stream.  A deep copy is made internally; the original is
        never modified.
    output_label : str, optional
        Prefix for the channel code of output trace(s).  Default ``"ALG"``.
    trim : bool, optional
        If ``True`` (default ``False``), trim all traces to the common
        time window [max(starttimes), min(endtimes)] before building the
        evaluation namespace.
    fill_gaps : bool, optional
        If ``True`` (default ``False``), merge the stream and fill any
        gap or overlap using linear interpolation (``merge(fill_value=
        "interpolate")``).  Applied **after** trimming (if enabled).
    resample : bool, optional
        If ``True`` (default ``False``), resample all traces to a common
        sampling rate.  Applied **after** gap-filling (if enabled).
    resample_to : float, optional
        Target sampling rate in Hz.  When ``resample=True`` and this is
        ``None`` (the default), the highest sampling rate present in the
        stream is used.  Ignored when ``resample=False``.

    Raises
    ------
    TraceAlgebraError
        If the stream is empty, not an ObsPy Stream, or if trimming
        produces an empty / non-overlapping result.
    """

    def __init__(self, stream: Stream, *, output_label: str = "ALG", trim: bool = False,
                 fill_gaps: bool = False, resample: bool = False, resample_to: Optional[float] = None) -> None:
        if not isinstance(stream, Stream):
            raise TraceAlgebraError(
                f"Expected an obspy.Stream, got {type(stream).__name__}."
            )
        if len(stream) == 0:
            raise TraceAlgebraError("The input stream contains no traces.")

        self._output_label = output_label

        # Work on a deep copy so the caller's stream is never touched
        working = copy.deepcopy(stream)

        # --- pre-processing pipeline (order is fixed: trim → gaps → resample)
        if trim:
            working = self._apply_trim(working)
        if fill_gaps:
            working = self._apply_fill_gaps(working)
        if resample:
            working = self._apply_resample(working, resample_to)

        self._stream = working

        # Build evaluation namespace
        self._namespace: Dict[str, np.ndarray] = {}
        self._meta: List[dict] = []

        for idx, tr in enumerate(self._stream):
            alias      = f"tr{idx + 1}"
            seed_alias = tr.id.replace(".", "_")
            data       = tr.data.astype(float)
            self._namespace[alias]      = data
            self._namespace[seed_alias] = data
            self._meta.append({
                "network":       tr.stats.network,
                "station":       tr.stats.station,
                "location":      tr.stats.location,
                "channel":       tr.stats.channel,
                "sampling_rate": tr.stats.sampling_rate,
                "starttime":     tr.stats.starttime,
            })

        self._namespace.update(_ALLOWED_CONSTANTS)

    # ------------------------------------------------------------------
    # Pre-processing steps
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_trim(stream: Stream) -> Stream:
        """Trim to [max(starttime), min(endtime)] across all traces."""
        t_start = max(tr.stats.starttime for tr in stream)
        t_end   = min(tr.stats.endtime   for tr in stream)

        if t_start >= t_end:
            raise TraceAlgebraError(
                f"Traces do not overlap: latest starttime {t_start} is not "
                f"before earliest endtime {t_end}.  Cannot trim."
            )

        logger.debug("TraceAlgebra trim: [%s, %s]", t_start, t_end)
        stream.trim(starttime=t_start, endtime=t_end)
        return stream

    @staticmethod
    def _apply_fill_gaps(stream: Stream) -> Stream:
        """Merge traces and fill gaps / overlaps with linear interpolation."""
        logger.debug("TraceAlgebra fill_gaps: merging with interpolation")
        stream.merge(method=1, fill_value="interpolate", interpolation_samples=-1)
        return stream

    @staticmethod
    def _apply_resample(stream: Stream, target_fs: Optional[float]) -> Stream:
        """Resample all traces to *target_fs* (or the highest fs if None)."""
        if target_fs is None:
            target_fs = max(tr.stats.sampling_rate for tr in stream)

        logger.debug("TraceAlgebra resample: target fs = %s Hz", target_fs)
        for tr in stream:
            if tr.stats.sampling_rate != target_fs:
                tr.resample(sampling_rate=target_fs)
        return stream

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        expression: str,
        output_channel: Optional[str] = None,
    ) -> Stream:
        """Parse and evaluate *expression*, returning a new :class:`~obspy.core.stream.Stream`.

        Parameters
        ----------
        expression : str
            Algebraic expression referencing traces by ``tr1``, ``tr2``, …
            or by SEED id (dots replaced by underscores).
            Multiple comma-separated sub-expressions produce a multi-trace
            stream, e.g. ``"tr1+tr2, tr1-tr2"``.
        output_channel : str, optional
            Force a specific channel code on the output trace.  Only
            meaningful when the expression produces a single trace.

        Returns
        -------
        obspy.Stream
            New stream with one trace per sub-expression.

        Raises
        ------
        TraceAlgebraError
            On any parsing, safety, or evaluation error.
        """
        expression = expression.strip()
        if not expression:
            raise TraceAlgebraError("Expression string is empty.")

        sub_expressions = self._split_top_level(expression)

        out_stream = Stream()
        for i, sub_expr in enumerate(sub_expressions):
            sub_expr     = sub_expr.strip()
            result_array = self._eval_single(sub_expr)
            tr = self._make_trace(
                result_array,
                sub_expression=sub_expr,
                output_channel=output_channel if len(sub_expressions) == 1 else None,
                index=i,
            )
            out_stream.append(tr)

        return out_stream

    def list_trace_names(self) -> List[str]:
        """Return positional aliases and SEED aliases for all traces in the stream."""
        return [f"tr{idx + 1}  →  {tr.id}" for idx, tr in enumerate(self._stream)]

    def preprocessing_summary(self) -> dict:
        """Return a summary of the (post-preprocessing) stream for quick inspection.

        Returns
        -------
        dict
            Keys: ``n_traces``, ``common_starttime``, ``common_endtime``,
            ``sampling_rates``, ``npts``.
        """
        return {
            "n_traces":        len(self._stream),
            "common_starttime": str(min(tr.stats.starttime for tr in self._stream)),
            "common_endtime":   str(max(tr.stats.endtime   for tr in self._stream)),
            "sampling_rates":  [tr.stats.sampling_rate for tr in self._stream],
            "npts":            [tr.stats.npts           for tr in self._stream],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_single(self, expression: str) -> np.ndarray:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise TraceAlgebraError(
                f"Syntax error in expression '{expression}': {exc}"
            ) from exc

        _check_ast(tree)

        eval_ns = dict(self._namespace)
        eval_ns.update(_ALLOWED_FUNCTIONS)

        try:
            code   = compile(tree, filename="<trace_algebra>", mode="eval")
            result = eval(code, {"__builtins__": {}}, eval_ns)  # noqa: S307
        except Exception as exc:
            raise TraceAlgebraError(
                f"Error evaluating expression '{expression}': {exc}"
            ) from exc

        result = np.asarray(result, dtype=float)
        if result.ndim == 0:
            n      = len(next(iter(self._namespace.values())))
            result = np.full(n, float(result))

        return result

    def _make_trace(
        self,
        data: np.ndarray,
        sub_expression: str,
        output_channel: Optional[str],
        index: int,
    ) -> Trace:
        ref_meta = self._infer_reference_meta(sub_expression)
        channel  = output_channel or f"{self._output_label}{index if index > 0 else ''}"
        stats = {
            "network":       ref_meta["network"],
            "station":       ref_meta["station"],
            "location":      ref_meta["location"],
            "channel":       channel,
            "sampling_rate": ref_meta["sampling_rate"],
            "starttime":     ref_meta["starttime"],
        }
        return Trace(data=data, header=stats)

    def _infer_reference_meta(self, expression: str) -> dict:
        positional = re.findall(r"\btr(\d+)\b", expression)
        if positional:
            idx = int(positional[0]) - 1
            if 0 <= idx < len(self._meta):
                return self._meta[idx]
        for idx, tr in enumerate(self._stream):
            if tr.id.replace(".", "_") in expression:
                return self._meta[idx]
        return self._meta[0]

    @staticmethod
    def _split_top_level(expression: str) -> List[str]:
        """Split on commas that are NOT inside parentheses."""
        parts: List[str] = []
        depth   = 0
        current: List[str] = []
        for ch in expression:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current))
        return parts


