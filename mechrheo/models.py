import numpy as _np

__all__ = ['CrossWLF']


class Model:
    def __init__(self):
        for fp in self._fitparams:
            setattr(self, fp, None)
        self.R_squared = 0.

    _fitparams = []


    def _kwargs(self, *args, **kwargs):
        if len(args) == len(self._fitparams):
            return dict(zip(self._fitparams, args))
        if len(kwargs) == len(self._fitparams):
            return kwargs
        else:
            values = []
            for fp in self._fitparams:
                values.append(getattr(self, fp))
            return dict(zip(self._fitparams, values))

class CrossWLF(Model):
    def __init__(self):
        self._fitparams = ['tau_star', 'A_1', 'A_2', 'D_1', 'D_2', 'n']
        super(CrossWLF, self).__init__()

    def __call__(self, x, *args, **kwargs):
        T = x[0]
        gamma_dot = x[1]
        p = self._kwargs(*args, **kwargs)
        eta_0 = p['D_1'] * _np.exp(- p['A_1'] * (T - p['D_2']) / (p['A_2'] + T - p['D_2']))
        eta = eta_0 / (1. + (eta_0 * gamma_dot / p['tau_star']) ** (1. - p['n']))
        return eta
