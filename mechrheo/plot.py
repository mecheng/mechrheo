import matplotlib.pyplot as _plt
import matplotlib as _mpl
import numpy as _np

from mechrheo import ureg


def plot(model, **kwargs):
    fig, ax = _plt.subplots()
    for i, T in enumerate(model._temperature):
        eta = []
        for gamma_dot in model._frequency:
            eta.append(model((T.to('K').m, gamma_dot.to('Hz').m)))
        eta = _np.array(eta)
        ax.loglog(model._frequency, eta * ureg.Pa * ureg.s, label=r'$T = {}$'.format(T))
        ax.loglog(model._frequency, model._viscosity[:, i], color='black', marker='.', linestyle='')
    ax.set_title("{} Model - {}".format(model.model, model.name))
    xlabel = r'$\dot{\gamma}~[Hz]$'
    ax.set_xlabel(xlabel)
    ylabel = r'$\eta~[Pa \cdot s]$'
    ax.set_ylabel(ylabel)
    ax.grid(True, which='both')
    if 'xlim' in kwargs:
        ax.set_xlim(*kwargs['xlim'])
    if 'ylim' in kwargs:
        ax.set_ylim(*kwargs['ylim'])
    if 'figsize' in kwargs:
        fig.set_size_inches(*kwargs['figsize'])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.05, model._repr_markdown_(), transform=ax.transAxes,
            verticalalignment='bottom', bbox=props)
    ax.legend()
    if 'filename' in kwargs:
        _plt.savefig(fname=kwargs['filename'], format=kwargs['filename'].split('.')[-1])
    _plt.show()
