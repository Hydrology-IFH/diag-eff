# Getting started


## Diagnostic Efficiency
Load the main module de. It can directly be imported
from the package, called de. The calculation of the diagnostic efficiency can be
easily demonstrated on the provided example dataset.

.. ipython:: python
    :okwarning:

    from de import de
    from de import util

    # path to example data
    path = '.../obs_sim.csv'
    # import observed time series
    df_ts = util.import_ts(path, sep=';')

    # make numpy arrays
    obs_arr = df_ts['Qobs'].values
    sim_arr = df_ts['Qsim'].values

    # calculate the diagnostic efficiency
    sig_de = de.calc_de(obs_arr, sim_arr)


## Diagnostic plot

.. ipython:: python
    :okwarning:
    @savefig default_diagnostic_plot.png width=7in
    de.vis2d_de(obs_arr, sim_arr)
