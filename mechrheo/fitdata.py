import numpy as _np
import scipy.optimize as _optimize


class FitData:
    def __init__(self, **kwargs):
        self.frequency = _np.array([])
        self.temperature = _np.array([])
        self.viscosity = _np.array([])
        if 'data' in kwargs:
            self.frequency = kwargs['data'][1:, 0]
            self.temperature = kwargs['data'][0, 1:]
            self.viscosity = kwargs['data'][1:, 1:]
        elif 'filename' in kwargs:
            delimiter = kwargs.setdefault('delimiter', ',')
            data = _np.loadtxt(kwargs['filename'], delimiter=delimiter)
            self.frequency = data[1:, 0]
            self.temperature = data[0, 1:]
            self.viscosity = data[1:, 1:]

    def fit(self, model, log=True, **kwargs):
        TT, SS = _np.meshgrid(self.temperature, self.frequency)

        if log:
            def func(x, *args, **kwargs):
                return _np.log10(model(x, *args, **kwargs)).flatten()

            visc = _np.log10(self.viscosity).flatten()
        else:
            def func(x, *args, **kwargs):
                return model(x, *args, **kwargs).flatten()

            visc = self.viscosity.flatten()

        popt, pcov = _optimize.curve_fit(func, (TT, SS), visc, **kwargs)
        for fp, value in zip(model._fitparams, popt):
            setattr(model, fp, value)

        model.R_squared = self.calc_r_squared(model, (TT, SS))
        return model

    def calc_r_squared(self, model, x):
        residuals = self.viscosity - model(x)
        ss_res = _np.sum(residuals ** 2)
        ss_tot = _np.sum(self.viscosity - _np.mean(self.viscosity) ** 2)
        return 1 - (ss_res / ss_tot)
