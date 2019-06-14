import numpy as _np
import scipy.optimize as _optimize

from mechrheo import ureg


class FitData:
    def __init__(self, **kwargs):
        r"""
        Data fitter

        First load the data, then fit it according to a model such as Cross-WLF

        Usage:

        .. code-block::

           from mechrheo.models import CrossWLF
           from mechrheo.fitdata import FitData

           model = CrossWLF()
           fitter = FitData(filename='visc_data.csv')
           fitter.fit(model)
           print(model)

        Args:
            **kwargs:

              * frequency (ndarray) - shear rate with unit in [Hz] (optinal)
              * temperature (ndarray) - temperature rate with unit [K] (optional)
              * viscosity (ndarray) = viscosity table of size (temperaturesize, frequency.size) with unit [Pa*s]
              * filename (str) Viscosity table as csv file with header (optional) according to the following structure:
                .. code-block::

                    # name: <name tag>
                    # date: <date tag>
                    # id: <id tag>
                    # row: <frequency | temperature> [<Hz or s**-1 | degC or K>]
                    # column: <temperature | frequency> [<degC or K | Hz or s**-1>]
                    # data: viscosity [<Pa*s>]
                        ,<T1>   ,<T2>   ,...    ,<Tn>
                    <F1>,<V11>  ,<V12>  ,<V...1>,<Vn1>
                    <F2>,<V12>  ,<V22>  ,<V...2>,<Vn2>
                    ..., <V1...>,<V2...>,<...>  ,<Vn....>
                    <Fm>,<V1m>  ,<V2m>  ,<V...m>,<Vnm>

              * delimiter (str) - Table delimiter value (optional)
        """
        self.frequency = _np.array([]) * ureg.Hz
        self.temperature = _np.array([]) * ureg.K
        self.viscosity = _np.array([]) * ureg.Pa * ureg.s
        self.name = kwargs.setdefault('name', '')
        self.date = kwargs.setdefault('date', '')
        self.id = kwargs.setdefault('id', '')
        if 'filename' in kwargs:
            delimiter = kwargs.setdefault('delimiter', ',')
            _data = _np.genfromtxt(kwargs['filename'], delimiter=delimiter, comments="#")
            transpose = False
            with open(kwargs['filename'], "rt") as f:
                for line in f:
                    if line[0] == "#":
                        comment = line[1:].split(":")
                        if len(comment) != 2:
                            continue
                        else:
                            attr = comment[0].strip().lower()
                            value = comment[1].strip()
                            if attr in ['name', 'date', 'id']:
                                setattr(self, attr, value)
                            elif attr == 'row':
                                val_attr = value.split()
                                unit = ureg.parse_expression(val_attr[1])
                                if val_attr[0] == 'temperature':
                                    transpose = True
                                    self.temperature = _data[1:, 0] * unit
                                else:
                                    self.frequency = _data[1:, 0] * unit
                            elif attr == 'column':
                                val_attr = value.split()
                                unit = ureg.parse_expression(val_attr[1])
                                if val_attr[0] == 'frequency':
                                    transpose = True
                                    self.frequency = _data[0, 1:] * unit
                                else:
                                    self.temperature = _data[0, 1:] * unit
                            elif attr == 'data':
                                val_attr = value.split()
                                unit = ureg.parse_expression(val_attr[1])
                                if transpose:
                                    self.viscosity = _data[1:, 1:].T * unit
                                else:
                                    self.viscosity = _data[1:, 1:] * unit
                    else:
                        break

    def fit(self, model, log=True, **kwargs):
        r"""
        Fit the loaded data to a chosen model

        Args:
            model (mechrheo.models.Model): The model object
            log (bool): Fit using logarithmic scales default=True
            **kwargs:

              * p0 (tuple): inital guess for the model fit parameters
              * bounds (tuple(lower bounds, higher bounds)): bounds
              * maxfev (int): maximum number of iterations

            If no additional keywords are used the default parameters from the model object are used, if these are not specified
            the default scipy.fit_curve values are used.

        Returns:
            Fitted model
        """
        TT, SS = _np.meshgrid(self.temperature.to('K').m, self.frequency.to('Hz').m)

        if log:
            def func(x, *args, **kwargs):
                return _np.log10(model(x, *args, **kwargs)).flatten()

            visc = _np.log10(self.viscosity.to('Pa*s').m).flatten()
        else:
            def func(x, *args, **kwargs):
                return model(x, *args, **kwargs).flatten()

            visc = self.viscosity.to('Pa*s').m.flatten()

        for kw in model._std_fit_kw:
            if kw not in kwargs and getattr(model, kw):
                kwargs[kw] = getattr(model, kw)

        popt, pcov = _optimize.curve_fit(func, (TT, SS), visc, **kwargs)
        for fp, value in zip(model._fitparams, popt):
            setattr(model, fp, value)

        model.R_squared = self.calc_r_squared(model, (TT, SS))
        model.name = self.name
        model.id = self.id
        model.date = self.date
        model._temperature = self.temperature
        model._frequency = self.frequency
        model._viscosity = self.viscosity
        return model

    def calc_r_squared(self, model, x):
        r"""
        Calculates the R**2 for the fit parameters and the measured data

        Args:
            model: (mechrheo.models.Model): The model object
            x (ndarray): tuple with meshgrid (temperature, frequency)

        Returns:
            R**2
        """
        y = self.viscosity.to('Pa*s').m
        y_hat = model(x)
        y_bar = _np.mean(y)
        ss_reg = _np.sum((y - y_bar) ** 2)
        ss_tot = _np.sum((y_hat - y_bar) ** 2)
        return ss_reg / ss_tot
