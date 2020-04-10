# install.packages("reticulate")
library(reticulate)

# pip installation
# py_install("numpy")
# py_install("pandas")
# py_install("scipy")
# py_install("matplotlib")
# py_install("seaborn")
# py_install("diag-eff")

# conda installation (requires anaconda. if conda is not installed on your
# computer use pip installation instead)
conda_create("de-reticulate")
use_condaenv("de-reticulate")

# install numpy
conda_install("de-reticulate", "numpy")
# install numpy
conda_install("de-reticulate", "pandas")
# install SciPy
conda_install("de-reticulate", "scipy")
# install matplotlib
conda_install("de-reticulate", "matplotlib")
# install seaborn
conda_install("de-reticulate", "seaborn")
# install tk
conda_install("de-reticulate", "tk")

# install de
# conda_install("de-reticulate", "diag-eff")

use_condaenv("de-reticulate")
# import Python modules
os <- import("os")
np <- import("numpy")
pd <- import("pandas")
sp <- import("scipy")
mpl <- import("matplotlib")
mpl$use("Agg", force = TRUE)
plt <- import("matplotlib.pyplot")
sns <- import("seaborn")
de <- import("de")

path_wd <- "./Desktop/PhD/diagnostic_efficiency"
setwd(path_wd)
source_python('de/de.py')
source_python('de/util.py')

# set path to example data
path_cam <- file.path(path_wd, 'examples/13331500_94_model_output.txt')

# import example data as dataframe
df_cam <- import_camels_obs_sim(path_cam)

# calculate diagnostic efficiency
sig_de <- calc_de(df_cam$Qobs, df_cam$Qsim)

# diagnostic polar plot
fig <- diag_polar_plot(df_cam$Qobs, df_cam$Qsim)
fig$show()
# currently figures cannot be displayed
fig$savefig('diagnostic_polar_plot.png')

repl_python()
# copy+paste the lines below to the interpreter
import os
PATH = './Desktop/PhD/diagnostic_efficiency/diag-eff'
os.chdir(PATH)
from de import de
from de import util
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# set path to example data
path = os.path.join(os.getcwd(), 'examples/13331500_94_model_output.txt')

# import example data as dataframe
df_cam = util.import_camels_obs_sim(path)

# make arrays
obs_arr = df_cam['Qobs'].values
sim_arr = df_cam['Qsim'].values

plt.plot(obs_arr)
plt.show()

# calculate diagnostic efficiency
sig_de = de.calc_de(obs_arr, sim_arr)

# diagnostic polar plot
de.diag_polar_plot(obs_arr, sim_arr)
plt.show()

# quit the interpreter
exit
