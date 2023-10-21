import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.sparse.linalg import factorized

from numerical import difference, operator

class Fluid:
    def __init__(self, shape, *quantites, pressure_order=1, advect_order=3):
        self.shape = shape
        self.dimensions = len(shape)

        self.quantities = quantites
        for q in quantites:
            setattr(self, q, np.zeros(shape))

        self.indices = np.indices(shape)
        self.velocity = np.zeros((self.dimensions, *shape))

        laplacian = operator(shape, difference(2, pressure_order))
        self.pressure_solver = factorized(laplacian)

        self.advect_order = advect_order

    def step(self):

        advection_map = self.indices - self.velocity

        def advect(field, filter_epsilon=10e-2, mode='constant'):
            filtered = spline_filter(field, order=self.advect_order, mode=mode)
            field = filtered * (1 - filter_epsilon) + field * filter_epsilon
            return map_coordinates(field, advection_map, prefilter=False, order=self.advect_order, mode=mode)


        for d in range(self.dimensions):
            self.velocity[d] = advect(self.velocity[d])

        for q in self.quantities:
            setattr(self, q, advect(getattr(self, q)))

        jacobian_shape = (self.dimensions,) * 2
        partials = tuple(np.gradient(d) for d in self.velocity)
        jacobian = np.stack(partials).reshape(*jacobian_shape, *self.shape)

        divergance = jacobian.trace()

        curl_mask = np.triu(np.ones(jacobian_shape, dtype=bool), k=1)
        curl = (jacobian[curl_mask] - jacobian[curl_mask.T]).squeeze()

        pressure = self.pressure_solver(divergance.flatten()).reshape(self.shape)
        self.velocity -= np.gradient(pressure)
        return divergance, curl, pressure