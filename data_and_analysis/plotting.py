import matplotlib
from matplotlib import cm, colors, pyplot
import numpy
from operator import itemgetter
import pathlib
from numpy.lib.arraysetops import isin
import scipy
import string
import typing
import arviz
import pymc3


import calibr8
import murefi

import models


params = {
    "text.latex.preamble": "\\usepackage{gensymb}",
    "image.origin": "lower",
    "image.interpolation": "nearest",
    "axes.grid": False,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "font.size": 16,
    "legend.fontsize": 11,
    "legend.frameon": False,
    "legend.fancybox": False,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Noteworthy", "DejaVu Sans", "Lucida Grande", "Verdana"],
    "lines.markersize": 3,
}

DP_FIGURES = pathlib.Path(__file__).parent / "figures"

matplotlib.rcParams.update(params)


def savefig(fig, name: str, **kwargs):
    """Saves a bitmapped and vector version of the figure.

    Parameters
    ----------
    fig
        The figure object.
    name : str
        Filename without extension.
    **kwargs
        Additional kwargs for `pyplot.savefig`.
    """
    fig.savefig(DP_FIGURES / f"{name}.png", **kwargs)
    fig.savefig(DP_FIGURES / f"{name}.pdf", **kwargs)
    return


def to_colormap(dark):
    N = 256
    dark = numpy.array((*dark[:3], 1))
    white = numpy.ones(4)
    cvals = numpy.array([
        (1 - n) * white + n * dark
        for n in numpy.linspace(0, 1, N)
    ])
    # add transparency
    cvals[:, 3] = numpy.linspace(0, 1, N)
    return colors.ListedColormap(cvals)


def transparentify(cmap: colors.Colormap) -> colors.ListedColormap:
    """Creates a transparent->color version from a standard colormap.
    
    Stolen from https://stackoverflow.com/a/37334212/4473230
    
    Testing
    -------
    x = numpy.arange(256)
    fig, ax = pyplot.subplots(figsize=(12,1))
    ax.scatter(x, numpy.ones_like(x) - 0.01, s=100, c=[
        cm.Reds(v)
        for v in x
    ])
    ax.scatter(x, numpy.ones_like(x) + 0.01, s=100, c=[
        redsT(v)
        for v in x
    ])
    ax.set_ylim(0.9, 1.1)
    pyplot.show()
    """
    # Get the colormap colors
    #cm_new = numpy.zeros((256, 4))
    #cm_new[:, :3] = numpy.array(cmap(cmap.N))[:3]
    cm_new = numpy.array(cmap(numpy.arange(cmap.N)))
    cm_new[:, 3] = numpy.linspace(0, 1, cmap.N)
    return colors.ListedColormap(cm_new)


redsT = transparentify(cm.Reds)
greensT = transparentify(cm.Greens)
bluesT = transparentify(cm.Blues)
orangesT = transparentify(cm.Oranges)
greysT = transparentify(cm.Greys)


class FZcolors:
    red = numpy.array((191, 21, 33)) / 255
    green = numpy.array((0, 153, 102)) / 255
    blue = numpy.array((2, 61, 107)) / 255
    orange = numpy.array((220, 110, 0)) / 255


class FZcmaps:
    red = to_colormap(FZcolors.red)
    green = to_colormap(FZcolors.green)
    blue = to_colormap(FZcolors.blue)
    orange = to_colormap(FZcolors.orange)
    black = transparentify(cm.Greys)


def plot_glucose_cmodels(fn_out=None, *, residual_type="relative"):
    model_lin = models.get_glucose_model_linear()
    model_asym = models.get_glucose_model()
    X = model_asym.cal_independent
    Y = model_asym.cal_dependent
    fig, axs = pyplot.subplots(
        nrows=2, ncols=3, figsize=(16, 10), dpi=120, sharex="col", sharey="col"
    )
    calibr8.plot_model(model_lin, fig=fig, axs=axs[0, :], residual_type=residual_type)
    calibr8.plot_model(model_asym, fig=fig, axs=axs[1, :], residual_type=residual_type)
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            if i == 1:
                ax.set_xlabel("glucose concentration [g/L]")
            else:
                ax.set_xlabel("")
            ax.scatter([], [], label="calibration data", color="#1f77b4")
            ax.plot([], [], label="$\mu_\mathrm{dependent}$", color="green")
            if j in [0, 1]:
                ax.set_ylabel("absorbance$_{365 \mathrm{nm}}$ [a.u.]")
                if i == 0:
                    ax.scatter(
                        X[X > 20],
                        Y[X > 20],
                        color="#1f77b4",
                        marker="x",
                        label="calibration data (ignored)",
                        s=25,
                    )
            ax.text(
                0.03,
                0.93,
                string.ascii_uppercase[j + i * len(row)],
                transform=ax.transAxes,
                size=20,
                weight="bold",
            )
    handles, labels = axs[0, 0].get_legend_handles_labels()
    handles2, labels2 = (
        (itemgetter(1, 2, 3, 0, -2, -1)(handles)),
        (itemgetter(1, 2, 3, 0, -2, -1)(labels)),
    )
    axs[0, 0].legend(handles2, labels2, loc="lower right")
    if residual_type == "relative":
        axs[0, 2].set_ylim(-0.1, 0.2)
        axs[1, 2].set_ylim(-0.1, 0.2)
    pyplot.tight_layout()
    if fn_out:
        savefig(fig, fn_out)


def plot_biomass_cmodel(fn_out=None, *, residual_type="relative"):
    btm_model = models.get_biomass_model()
    fig, axs = pyplot.subplots(nrows=1, ncols=3, figsize=(16, 6.5), dpi=120)
    calibr8.plot_model(btm_model, fig=fig, axs=axs, residual_type=residual_type)
    for i, ax in enumerate(axs):
        ax.set_xlabel("biomass concentration [g/L]")
        ax.scatter([], [], label="calibration data", color="#1f77b4")
        if i in [0, 1]:
            ax.set_ylabel("backscatter [a.u.]")
        ax.plot([], [], label="$\mu_\mathrm{dependent}$", color="green")
        # if i in [0,1]:
        #   ax.set_ylabel('absorbance$_{365 \mathrm{nm}}$ [a.u.]')
        ax.text(
            0.03,
            0.93,
            string.ascii_uppercase[i],
            transform=ax.transAxes,
            size=20,
            weight="bold",
        )
    handles, labels = axs[0].get_legend_handles_labels()
    handles2, labels2 = (
        (itemgetter(1, 2, 3, 0, -1)(handles)),
        (itemgetter(1, 2, 3, 0, -1)(labels)),
    )
    axs[0].legend(handles2, labels2, loc="lower right")
    pyplot.tight_layout()
    if fn_out:
        savefig(fig, fn_out)


def extract_parameters(
    idata: arviz.InferenceData,
    theta_mapping: murefi.ParameterMapping,
    nmax: int = 1_000,
):
    posterior = idata.posterior.stack(sample=("chain", "draw"))
    parameters = {}
    nsamples = idata.posterior.dims["chain"] * idata.posterior.dims["draw"]
    nmax = min(nmax, nsamples)
    # randomly shuffle the samples using the following indices
    idxrnd = numpy.random.permutation(numpy.arange(nmax))
    for pname, pkind in theta_mapping.parameters.items():
        with_coords = pname != pkind
        if with_coords:
            coord = tuple(posterior[pkind].coords.keys())[0]
            pvals = posterior[pkind].sel({coord: pname.replace(f"{pkind}_", "")}).values
        else:
            pvals = posterior[pname].values
        parameters[pname] = pvals[idxrnd]
    return parameters


def plot_residuals_pp(
    ax,
    cmodel,
    tsobs: murefi.Timeseries,
    tspred: murefi.Timeseries,
    *,
    color,
    palette,
    tspred_extra: typing.Optional[murefi.Timeseries] = None,
):
    assert isinstance(cmodel, calibr8.BaseModelT)
    numpy.testing.assert_array_equal(tspred.t, tsobs.t)

    # for each of the 9000 posterior samples, draw 1 observation
    mu, scale, df = cmodel.predict_dependent(tspred.y)
    ppbs = scipy.stats.t.rvs(loc=mu, scale=scale, df=df)
    median = numpy.median(ppbs, axis=0)

    if tspred_extra is not None:
        # tspred_extra may be used to plot a higher resolution or extrapolated density
        mu, scale, df = cmodel.predict_dependent(tspred_extra.y)
        ppbs_extra = scipy.stats.t.rvs(loc=mu, scale=scale, df=df)
        pymc3.gp.util.plot_gp_dist(
            ax=ax,
            x=tspred_extra.t,
            samples=ppbs_extra - numpy.median(ppbs_extra, axis=0),
            palette=palette,
            plot_samples=False,
        )
    else:
        # plot the density from the data-like prediction
        pymc3.gp.util.plot_gp_dist(
            ax=ax, x=tsobs.t, samples=ppbs - median, palette=palette, plot_samples=False
        )
    yres = tsobs.y - median
    ax.scatter(
        tsobs.t,
        yres,
        marker="x",
        color=color,
    )
    return numpy.abs(yres).max()
