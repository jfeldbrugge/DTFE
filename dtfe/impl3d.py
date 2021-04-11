# ~\~ language=Python filename=dtfe/impl3d.py
# ~\~ begin <<lit/implementation.md|dtfe/impl3d.py>>[0]
import numba
from numba import float32, float64, int64
import numpy as np
from typing import Union

@numba.jit(nopython=True, nogil=True)
def tetrahedron_volume(sim: int64[:], points: float64[:,:]):
    return abs(np.linalg.det(np.stack((points[sim[1]] - points[sim[0]],
                                       points[sim[2]] - points[sim[0]],
                                       points[sim[3]] - points[sim[0]])))) / 6

@numba.jit(nopython=True, nogil=True)
def compute_densities(pts: float64[:,:], cells: float64[:,:],
                      m: Union[float64, float64[:]]) -> np.ndarray:
    M = len(pts)
    rho = np.zeros(M, dtype='float64')
    for cell in cells:
        rho[cell] += tetrahedron_volume(cell, pts)
    return 4 * m / rho

@numba.jit(nopython=True, nogil=True)
def compute_gradient_scalar(
        pts: float64[:,:], simps: int64[:,:], rho: float64[:],
        ) -> np.ndarray:
    N = len(simps)
    gradient = np.zeros((N, 3), dtype='float64')

    for i, s in enumerate(simps):
        [p0, p1, p2, p3] = pts[s]
        [r0, r1, r2, r3] = rho[s]

        A = np.stack((p1 - p0, p2 - p0, p3 - p0))
        gradient[i] = np.linalg.solve(A, np.array([r1 - r0, r2 - r0, r3 - r0]))
    return gradient

@numba.jit(nopython=True, nogil=True)
def compute_gradient_vector(
        pts: float64[:,:], simps: int64[:,:], v: float64[:,:]
        ) -> np.ndarray:
    N = len(simps)
    gradient = np.zeros((N, 3, 3), dtype='float64')

    for i, s in enumerate(simps):
        [p0, p1, p2, p3] = pts[s]
        [v0, v1, v2, v3] = v[s]

        A = np.stack((p1 - p0, p2 - p0, p3 - p0))
        gradient[i] = np.linalg.solve(A, np.stack((v1 - v0, v2 - v0, v3 - v0)))
    return gradient
# ~\~ end
