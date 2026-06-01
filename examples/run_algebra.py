from obspy import read
from surfquakecore.algebra.trace_algebra import TraceAlgebra

path = "/Users/roberto/Desktop/test_input/*"
st = read(path)
ta = TraceAlgebra(st, output_label = "ALG",
    trim         = True,
    fill_gaps    = True,
    resample     = True,
    resample_to  = None)

result = ta.evaluate("tr1 + tr2")

# complex expression
result = ta.evaluate("exp(tr1) + sin(tr2) * sqrt(abs(tr3))")
# SEED-id reference  (BW_RJOB__EHZ -> BW_RJOB__EHZ)
result = ta.evaluate("tr1 * 1e3")
# multi-output: comma-separated produces a multi-trace stream
result = ta.evaluate("tr1 + tr2, tr1 - tr2")
result.plot()