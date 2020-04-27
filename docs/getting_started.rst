===============
Getting started
===============

Diagnostic Efficiency
---------------------

Load the package `de`. The calculation of the diagnostic efficiency
can be easily demonstrated on the provided example dataset.

.. ipython:: python
    :okwarning:

    from pathlib import Path  # OS-independent path handling
    from de import de
    from de import util

    # path to example data
    path = Path('./examples/13331500_94_model_output.txt')
    # import observed time series
    df_ts = util.import_camels_obs_sim(path)

    # make numpy arrays
    obs_arr = df_ts['Qobs'].values
    sim_arr = df_ts['Qsim'].values

    # calculate the diagnostic efficiency
    de.calc_de(obs_arr, sim_arr)


Diagnostic polar plot
---------------------

.. ipython:: python
    :okwarning:

    from pathlib import Path  # OS-independent path handling
    from de import de
    from de import util

    # path to example data
    path = Path('./examples/13331500_94_model_output.txt')
    # import observed time series
    df_ts = util.import_camels_obs_sim(path)

    # make numpy arrays
    obs_arr = df_ts['Qobs'].values
    sim_arr = df_ts['Qsim'].values

    # display diagnostic polar plots
    @savefig default_diagnostic_plot.png width=7in
    de.diag_polar_plot(obs_arr, sim_arr)
