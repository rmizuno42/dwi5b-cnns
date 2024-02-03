import torch
import numpy as np
import elasticdeform.torch as etorch


class RandomElasticTransforms:
    def __init__(self, deform_scale=5, control_points=(3, 3), axis=None) -> None:
        self.deform_scale = deform_scale
        self.control_points = control_points
        self.axis = axis

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.randint(2) == 0:
            return x
        # we assume x.size() == (c, ... )

        if self.axis is None:
            axis = tuple([i + 1 for i in range(len(x.size()[1:]))])
        else:
            axis = self.axis
        data_dim = len(axis)
        d = np.random.randn(data_dim, *(self.control_points)) * self.deform_scale
        x = etorch.deform_grid(x, torch.tensor(d), axis=axis, mode="constant")
        x[x < 0] = 0
        return x
