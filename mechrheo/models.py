import numpy as _np

__all__ = ['CrossWLF']


class Model:
    def __init__(self):
        for fp in self._fitparams:
            setattr(self, fp, None)
        for std_kw in self._std_fit_kw:
            setattr(self, std_kw, None)
        self._frequency = None
        self._temperature = None
        self._viscosity = None
        self.model = ""
        self.name = ""
        self.date = ""
        self.id = ""
        self.R_squared = 0.

    _fitparams = []
    _md_fitparams = []
    _std_fit_kw = ['p0', 'bounds', 'maxfev']
    _func_str = ''

    def __repr__(self):
        repr_str = "Material: {}\nDate: {}\nID: {}\n".format(self.name, self.date, self.id)
        for fp in self._fitparams:
            repr_str += "{} = {}\n".format(fp, getattr(self, fp))
        repr_str += "R**2 = {}".format(self.R_squared)
        return repr_str

    def _repr_markdown_(self):
        md = self._func_str
        for fp, fpp in zip(self._fitparams, self._md_fitparams):
            md += '{} = {}\n'.format(fpp, round(getattr(self, fp), 3))
        md += '\n\n'
        md += '$R^{2}='
        md += '{}$'.format(round(self.R_squared, 4))
        return md

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
        self._md_fitparams = [r'$\tau^{*}$', r'$A_1$', r'$A_2$', r'$D_1$', r'$D_2$', r'$n$']
        self._func_str = r'$\eta = \frac{\eta_0}{1+\left(\frac{\eta_0 \dot{\gamma}}{\tau^{*}}\right)^{1-n}}$'
        self._func_str += '\n'
        self._func_str += r'$\eta_0=e^{-\frac{A_1 (T - D_2)}{A_2 + T - D_2}}$'
        self._func_str += '\n\n'
        super(CrossWLF, self).__init__()
        self.model = "Cross-WLF (pressure independant)"
        self.p0 = (2.e4, 5., 1., 1.e5, 430., 0.1)
        self.bounds = ([0., 0., 0., 200., 0., 0.], [1.e6, 1.e6, 1.e4, 2.e20, 5.e2, 1.])
        self.maxfev = 4000

    def __call__(self, x, *args, **kwargs):
        T = x[0]
        gamma_dot = x[1]
        p = self._kwargs(*args, **kwargs)
        eta_0 = p['D_1'] * _np.exp(- p['A_1'] * (T - p['D_2']) / (p['A_2'] + T - p['D_2']))
        eta = eta_0 / (1. + (eta_0 * gamma_dot / p['tau_star']) ** (1. - p['n']))
        return eta
