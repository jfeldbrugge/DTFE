# ~\~ language=Python filename=dtfe/impl2d.py
# ~\~ begin <<lit/implementation.md|dtfe/impl2d.py>>[0]
import numba
from numba import float32, float64, int64
import numpy as np
from typing import Union

# ~\~ begin <<lit/implementation.md|triangle-area>>[0]
@numba.jit(nopython=True, nogil=True)
def triangle_area(sim: int64[:], points: float64[:,:]):
    return abs(np.linalg.det(np.stack((points[sim[1]] - points[sim[0]], 
                                       points[sim[2]] - points[sim[0]])))) / 2
# ~\~ end
# ~\~ begin <<lit/implementation.md|compute-densities>>[0]
@numba.jit(nopython=True, nogil=True)
def compute_densities(pts: float64[:,:], cells: float64[:,:],
                      m: Union[float64, float64[:]]) -> np.ndarray:
    M = len(pts)
    area = np.zeros(M, dtype='float64')
    for cell in cells:
        area[cell] += triangle_area(cell, pts)
    return 3 * m / area
# ~\~ end
# ~\~ begin <<lit/implementation.md|compute-gradient-scalar>>[0]
@numba.jit(nopython=True, nogil=True)
def compute_gradient_scalar(
        pts: float64[:,:], simps: int64[:,:], rho: float64[:]
        ) -> float64[:,:]:
    N = len(simps)
    result = np.zeros((N, 2), dtype='float64')

    for i, s in enumerate(simps):
        [p0, p1, p2] = pts[s]
        [r0, r1, r2] = rho[s]
        # ~\~ begin <<lit/implementation.md|invert-2x2-matrix>>[0]
        A: float64[:,:] = np.stack((p1 - p0, p2 - p0))
        det = A[0,0] * A[1,1] - A[1,0] * A[0,1]
        Ainv = np.array([[A[1,1] / det, -A[0,1] / det],
                         [-A[1,0] / det, A[0,0] / det]])
        # ~\~ end
        result[i] = Ainv @ np.array([r1 - r0, r2 - r0])

    return result
# ~\~ end
# ~\~ begin <<lit/implementation.md|compute-gradient-vector>>[0]
@numba.jit(nopython=True, nogil=True)
def compute_gradient_vector(
        pts: float64[:,:], simps: float64[:,:], v: float64[:,:]
        ) -> float64[:,:,:]:
    N = len(simps)
    result = np.zeros((N, 2, 2), dtype='float64')

    for i, s in enumerate(simps):
        [p0, p1, p2] = pts[s]
        [v0, v1, v2] = v[s]
        # ~\~ begin <<lit/implementation.md|invert-2x2-matrix>>[0]
        A: float64[:,:] = np.stack((p1 - p0, p2 - p0))
        det = A[0,0] * A[1,1] - A[1,0] * A[0,1]
        Ainv = np.array([[A[1,1] / det, -A[0,1] / det],
                         [-A[1,0] / det, A[0,0] / det]])
        # ~\~ end
        result[i] = Ainv @ np.stack((v1 - v0, v2 - v0))

    return result
# ~\~ end
# ~\~ end
