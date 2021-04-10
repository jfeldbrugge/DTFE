# ~\~ language=Python filename=dtfe/dtfe.py
# ~\~ begin <<lit/implementation.md|dtfe/dtfe.py>>[0]
#Load the numpy and scipy libraries
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay
import numba
from typing import Union

from . import impl2d
from . import impl3d

# ~\~ begin <<lit/implementation.md|map-affine>>[0]
@numba.jit(nopython=True, nogil=True)
def map_affine(a, b, c):
    assert(len(a) == len(b) == len(c))
    result = np.zeros_like(a)
    for i in range(len(a)):
        result[i] = a[i] + b[i] @ c[i]
    return result
# ~\~ end

class Triangulation(Delaunay):
    @property
    def _ndim(self):
        return self.points.shape[-1]

    @property
    def _impl(self):
        if self._ndim == 2:
            return impl2d
        elif self._ndim == 3:
            return impl3d
        else:
            raise("Only implemented in 2d and 3d.")

    def density(self, bias=1.0):
        return self._impl.compute_densities(self.points, self.simplices, bias)

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
        if f.ndim == 1:
            self.gradient = self.triangulation._impl.compute_gradient_scalar(
                t.points, t.simplices, f)
        elif f.ndim == 2:
            self.gradient = self.triangulation._impl.compute_gradient_vector(
                t.points, t.simplices, f)
        else:
            raise ValueError("Interpolation only supported for scalar and vector values.")

    def __call__(self, *mesh):
        pts = np.stack([axis.flat for axis in mesh], axis=1)
        simplexIndex = self.triangulation.find_simplex(pts)
        pointIndex = self.triangulation.simplices[simplexIndex][...,0]
        f = map_affine(self.field[pointIndex], self.gradient[simplexIndex],
                       pts - self.triangulation.points[pointIndex])
        return f.reshape(mesh[0].shape + (-1,))

    # def theta(self, x, y):
    #     simplexIndex = self.delaunay.find_simplex(np.c_[x, y])
    #     return self.Dv[simplexIndex][...,0,0] + self.Dv[simplexIndex][...,1,1]

    # def omega(self, x, y):
    #     simplexIndex = self.delaunay.find_simplex(np.c_[x, y])
    #     return self.Dv[simplexIndex][...,1,0] - self.Dv[simplexIndex][...,0,1]
# ~\~ end
