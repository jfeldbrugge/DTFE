# Implementation

``` {.python file=dtfe/__init__.py}

```

## Helper functions

``` {.python #map-affine}
@numba.jit(nopython=True, nogil=True)
def map_affine(a, b, c):
    assert(len(a) == len(b) == len(c))
    result = np.zeros_like(a)
    for i in range(len(a)):
        result[i] = a[i] + b[i] @ c[i]
    return result
```

The area of a triangle or volume of a tetrahedron can be computed using the determinant method.

``` {.python #triangle-area}
@numba.jit(nopython=True, nogil=True)
def triangle_area(sim: int64[:], points: float64[:,:]):
    return abs(np.linalg.det(np.stack((points[sim[1]] - points[sim[0]], 
                                       points[sim[2]] - points[sim[0]])))) / 2
```

The density is defined on each vertex as one over the volume of the star of that vertex.

``` {.python #compute-densities}
@numba.jit(nopython=True, nogil=True)
def compute_densities(pts: float64[:,:], cells: float64[:,:],
                      m: Union[float64, float64[:]]) -> np.ndarray:
    M = len(pts)
    area = np.zeros(M, dtype='float64')
    for cell in cells:
        area[cell] += triangle_area(sim, pts)
    return 3 * m / area
```

To do the interpolation we need the derivatives of the quantities.

``` {.python #invert-2x2-matrix}
A: float64[:,:] = np.stack((p1 - p0, p2 - p0))
det = A[0,0] * A[1,1] - A[1,0] * A[0,1]
Ainv = np.array([[A[1,1] / det, -A[0,1] / det],
                 [-A[1,0] / det, A[0,0] / det]])
```

``` {.python #compute-gradient-scalar}
@numba.jit(nopython=True, nogil=True)
def compute_gradient_scalar(
        pts: float64[:,:], simps: int64[:,:], rho: float64[:]
        ) -> float64[:,:]:
    N = len(simps)
    result = np.zeros((N, 2), dtype='float64')

    for i, s in enumerate(simps):
        [p0, p1, p2] = pts[s]
        [r0, r1, r2] = rho[s]
        <<invert-2x2-matrix>>
        result[i] = Ainv @ np.array([r1 - r0, r2 - r0])

    return result
```

``` {.python #compute-gradient-vector}
@numba.jit(nopython=True, nogil=True)
def compute_gradient_vector(
        pts: float64[:,:], simps: float64[:,:], v: float64[:,:]
        ) -> float64[:,:,:]:
    N = len(simps)
    result = np.zeros((N, 2, 2), dtype='float64')

    for i, s in enumerate(simps):
        [p0, p1, p2] = pts[s]
        [v0, v1, v2] = v[s]
        <<invert-2x2-matrix>>
        result[i] = Ainv @ np.stack((v1 - v0, v2 - v0))

    return result
```

``` {.python file=dtfe/dtfe.py}
#Load the numpy and scipy libraries
import numpy as np
from scipy.spatial import Delaunay
import numba
from numba import float32, float64, int64
from typing import Union

<<triangle-area>>
<<compute-densities>>
<<compute-gradient-scalar>>
<<compute-gradient-vector>>
<<map-affine>>

class Triangulation(Delaunay):
    def density(self, bias=1.0):
        return compute_densities(self.points, self.simplices, bias)

    def dtfe(self, bias: Union[float, np.ndarray] = 1.0) -> Interpolation:
        return self.interpolate(self.density(bias))

    def interpolate(self, field) -> Interpolation:
        """Interpolates given field for which values are known at the
        points of the triangulation. If no `field` is given, this function
        uses the `density` method, resulting in the DTFE algorithm."""
        return Interpolation(self, field)

class Interpolation:
    def __init__(self, t: Triangulation, f: np.ndarray):
        self.triangulation = t
        self.field = f
        if f.ndim == 2:
            self.gradient = compute_gradient_scalar(t.points, t.simplices, f)
        elif f.ndim == 3:
            self.gradient = compute_gradient_vector(t.points, t.simplices, f)
        else:
            raise ValueError("Interpolation only supported for scalar and vector values.")

    def __call__(x: np.ndarray, y: np.ndarray):
        pts = np.c_[x.flat, y.flat]
        simplexIndex = self.triangulation.find_simplex(pts)
        pointIndex = self.triangualtion.simplices[simplexIndex][...,0]
        f = map_affine(self.field[pointIndex], self.gradient[simplexIndex],
                       pts - self.delaunay.points[pointIndex])
        return f.reshape(x.shape + (-1,))

    # def theta(self, x, y):
    #     simplexIndex = self.delaunay.find_simplex(np.c_[x, y])
    #     return self.Dv[simplexIndex][...,0,0] + self.Dv[simplexIndex][...,1,1]

    # def omega(self, x, y):
    #     simplexIndex = self.delaunay.find_simplex(np.c_[x, y])
    #     return self.Dv[simplexIndex][...,1,0] - self.Dv[simplexIndex][...,0,1]
```

