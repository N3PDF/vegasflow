# Place your function here
import time
import numpy as np
from vegasflow.vegas import vegas, DTYPE, DTYPEINT

from cffi import FFI
ffibuilder = FFI()


# MC integration setup
dim = 4
ncalls = np.int32(1e3)
n_iter = 5

ffibuilder.cdef("""
    void lepage(double*, int, int, double*);
""")

ffibuilder.set_source("_lepage_cffi",
    """
    void lepage(double *x, int n, int evts, double* out)
    {
        for (int e = 0; e < evts; e++)
        {
            double a = 0.1;
            double pref = pow(1.0/a/sqrt(M_PI), n);
            double coef = 0.0;
            for (int i = 1; i <= 100*n; i++) {
                coef += (float) i;
            }
            for (int i = 0; i < n; i++) {
                coef += pow((x[i+e*n] - 1.0/2.0)/a, 2);
            }
            coef -= 100.0*n*(100.0*n+1.0)/2.0;
            out[e] = pref*exp(-coef);
        }
    }
""")
ffibuilder.compile(verbose=True)

from _lepage_cffi import ffi, lib

def lepage(xarr, n_dim=None):
    res = np.zeros(xarr.shape[0])
    pinput = ffi.cast('double*', ffi.from_buffer(xarr.numpy().flatten()))
    pres = ffi.cast('double*', res.ctypes.data)
    lib.lepage(pinput, n_dim, xarr.shape[0], pres)
    return res

if __name__ == "__main__":
    """Testing a basic integration"""

    print(f"VEGAS MC, ncalls={ncalls}:")
    start = time.time()
    r = vegas(lepage, dim, n_iter, ncalls)
    end = time.time()
    print(f"time (s): {end-start}")