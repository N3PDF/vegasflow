# Place your function here
import time
import numpy as np
from vegasflow.configflow import DTYPE, DTYPEINT
from vegasflow.vflow import VegasFlow
import tensorflow as tf

from cffi import FFI
ffibuilder = FFI()


# MC integration setup
dim = 4
ncalls = np.int32(1e5)
n_iter = 5

if DTYPE is tf.float64:
    C_type = "double"
elif DTYPE is tf.float32:
    C_type = "float"
else:
    raise TypeError(f"Datatype {DTYPE} not understood")


ffibuilder.cdef(f"""
    void lepage({C_type}*, int, int, {C_type}*);
""")

ffibuilder.set_source("_lepage_cffi", f"""
    void lepage({C_type} *x, int n, int evts, {C_type}* out)
    {{
        for (int e = 0; e < evts; e++)
        {{
            {C_type} a = 0.1;
            {C_type} pref = pow(1.0/a/sqrt(M_PI), n);
            {C_type} coef = 0.0;
            for (int i = 1; i <= 100*n; i++) {{
                coef += ({C_type}) i;
            }}
            for (int i = 0; i < n; i++) {{
                coef += pow((x[i+e*n] - 1.0/2.0)/a, 2);
            }}
            coef -= 100.0*n*(100.0*n+1.0)/2.0;
            out[e] = pref*exp(-coef);
        }}
    }}
""")
ffibuilder.compile(verbose=True)

from _lepage_cffi import ffi, lib

def lepage(xarr, n_dim=None, **kwargs):
    if n_dim is None:
        n_dim = xarr.shape[-1]
    n_events = xarr.shape[0]

    res = np.empty(n_events, dtype = DTYPE.as_numpy_dtype)
    x_flat = xarr.numpy().flatten()

    pinput = ffi.cast(f'{C_type}*', ffi.from_buffer(x_flat))
    pres = ffi.cast(f'{C_type}*', ffi.from_buffer(res))
    lib.lepage(pinput, n_dim, n_events, pres)
    return res

if __name__ == "__main__":
    """Testing a basic integration"""

    print(f"VEGAS MC, ncalls={ncalls}:")
    start = time.time()
    vegas_instance = VegasFlow(dim, ncalls)
    vegas_instance.compile(lepage, compilable = False)
    r = vegas_instance.run_integration(n_iter)
    end = time.time()
    print(f"time (s): {end-start}")
