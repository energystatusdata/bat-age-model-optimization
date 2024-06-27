# Manual or semi-automatic optimization of battery aging models according to aging data, as described in:
# "Robust electricity grids through intelligent, highly efficient bidirectional charging systems for electric vehicles"
# (Dissertation of Matthias Luh, 2024)
#
# For use with the data set:
#   Matthias Luh and Thomas Blank. Comprehensive battery aging dataset: capacity and impedance fade measurements of a
#   lithium-ion NMC/C-SiO cell [dataset]. RADAR4KIT, 2024. DOI: 10.35097/1947
# ... however, you might adopt this script to fit other data, or adopt the data format to use this script.
#
# Feel free to implement new aging model functions.
# The model and settings that worked best with the data are:
#   model_f056
#   s0: 1.49e-9, s1: -2375, s2: 1.2, s3: 1.78e-8,
#   w0: 2.67e-7, w1: 2.25, w2: 0.14, w3: 9.5e-7,
#   p0: 0.07, p1: 0.029, p2: 314.65, p3: 3.5, p4: 0.33, p5: 5.3e-8, p6: 2.15,
#   c0: 3.60e-5, c1: 1050, c2: 3.2, c3: 2.47e-4, Vm: 3.73, Ci: 3.09
# The model and the variables are explained in Chapter 7.2 of the dissertation.
#
# ToDo: to start, you need to download the CFG and the LOG_AGE .csv files from RADAR4KIT (see above), e.g.:
#   cell_log_age_30s_P012_3_S14_C11.csv --> found in the "cell_log_age.7z" (e.g., use 7-Zip to unzip)
#   cell_cfg_P012_3_S14_C11.csv --> found in the "cfg.zip"

# import gc
import math
import time
import traceback
import pandas as pd
import config_main as cfg
import helper_tools as ht
import multiprocessing
from datetime import datetime
import os
import re
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from lmfit import Model
# from scipy import optimize
# import pyarrow as pa
from enum import IntEnum
import config_labels as csv_label
import config_logging  # as logging
import optimize_models_manual_stepping_list as optimizer_list


# Note: all "RMSE" values are relative to the nominal cell capacity ("RMSE%")!
# How to calculate the model error? This makes a difference if there are different numbers of CU's for the cells or
# different numbers of cells per parameter set. E.g., if RMSE_PER_CU instead of RMSE_PER_CELL / RMSE_PER_PARAMETER is
# used, the model might not be fitted well to cells that only have 2-3 CU's. If RMSE_PER_CELL is used instead of
# RMSE_PER_PARAMETER, the model might be fitted better to parameter sets that are tested with more cells.
class error_calculation_method(IntEnum):  # how to calculate the model error?
    RMSE_PER_CU = 0  # add error of each check-up (i.e., for each cell's measurement point) -> optimize CU fits
    RMSE_PER_CELL = 1  # calculate RMSE for each cell and add all values -> optimize cells
    RMSE_PER_PARAMETER = 2  # calculate RMSE for each parameter and add all values -> optimize parameters


# === ToDo: adjust these settings if you want ==========================================================================
STEPPING_MODE = 0  # 0: interactive mode, 1: stepping matrix-wise, 2: stepping list-wise
error_calc_method = error_calculation_method.RMSE_PER_PARAMETER  # Recommended: RMSE_PER_PARAMETER

# RMSE: x % / y % - x uses all points, y only the points for which the remaining cap > rmse_2_threshold
rmse_2_threshold = 0.7 * cfg.CELL_CAPACITY_NOMINAL

# show "error range" in the plot
# only works with model_f055 (and higher?) - it is recommended to set show_error_range to False when optimizing!
show_error_range = False  # show error range (start model with initial capacity +/- error_delta) - takes longer to plot!
error_delta = 0.02  # 0.03  # 0.05  # in 100%, i.e. 1 = 100%
error_delta_param = 0.02  # 0.03  # 0.05  # in 100%, i.e. 1 = 100%
P = +1.0  # more aging if the variable is larger in amplitude
N = -1.0  # less aging if the variable is larger in amplitude
Z = 0.0  # neither more nor less aging if the variable is larger in amplitude (or more complex dependency)
model_f056_error_var_dependency = [P, Z, Z, N,   P, P, N, N,   P, P, P, P, P, P, Z,   P, Z, Z, N,   Z, N]
# model_f056_error_var_dependency = [P, Z, Z, N,   P, P, N, N,   P, P, Z, Z, P, P, Z,   P, Z, Z, N,   Z, N]
# s0, s1, s2, s3,   w0, w1, w2, w3,   p0, p1, p2, p3, p4, p5, p6,   c0, c1, c2, c3,   Vm, Ci
# s0: 1.49e-9, s1: -2375, s2: 1.2, s3: 1.78e-8,
# w0: 2.67e-7, w1: 2.25, w2: 0.14, w3: 9.5e-7,
# p0: 0.07, p1: 0.029, p2: 314.65, p3: 3.5, p4: 0.33, p5: 5.3e-8, p6: 2.15,
# c0: 3.60e-5, c1: 1050, c2: 3.2, c3: 2.47e-4, Vm: 3.73, Ci: 3.09

# which aging types from the experiment shall be considered in optimization?
# CELL_PARAMETERS_USE = list(range(0, 17))  # all calendar aging cells
# CELL_PARAMETERS_USE = list(range(0, 65))  # all calendar + cyclic aging cells
# CELL_PARAMETERS_USE = list(range(17, 65))  # all cyclic aging cells
# CELL_PARAMETERS_USE = list(range(65, 77))  # all profile aging cells
CELL_PARAMETERS_USE = list(range(0, 77))  # all calendar + cyclic + profile aging cells

# RESOLUTION_USE = [2, 30]  # if None, use log_gae file with any time resolution. If >0, only use the one with specified
#   resolution. helpful if multiple resolutions in the same folder, e.g., RESOLUTION_USE = [30] -> only use log_age_30s
# RESOLUTION_USE = [4, 30]
# RESOLUTION_USE = [10, 30]
RESOLUTION_USE = [30]
# RESOLUTION_USE = [60]

# see prepare_cell_log_df(..., use_cap_nom, use_t_prod, ...) for the following settings
USE_NOMINAL_CAPACITY_DEFAULT = 0  # 0...1
# use_cap_nom = USE_NOMINAL_CAPACITY_DEFAULT
# if USE_NOMINAL_CAPACITY_DEFAULT >= 1:
#     C_0 = cfg.CELL_CAPACITY_NOMINAL
# else:
#     C_0 = use_cap_nom * cfg.CELL_CAPACITY_NOMINAL + (1.0 - use_cap_nom) * cap_aged.iloc[0]
USE_T_PRODUCTION_DEFAULT = True
# use_t_prod = USE_T_PRODUCTION_DEFAULT
# t = log_df.loc[:, DF_COL_TIME_USE]
# if use_t_prod, t = 0 at cfg.CELL_PRODUCTION_TIMESTAMP
# else, t = 0 at cfg.EXPERIMENT_START_TIMESTAMP

USE_ABS_TEMPERATURE = True
# if True, set 0°C to 273.15 K, i.e., add cfg.T0 (log_df.loc[:, csv_label.T_CELL] = log_df[csv_label.T_CELL] + cfg.T0)

N_PLOT_DATA_DOWNSAMPLE = 2880  # down-sampling for plots, 2880 is the factor to reduce 30-second data to 1-day data
CHUNK_DURATION = 3 * 24 * 60 * 60  # 3 days, process aging data in chunks of this duration if aging is iterative
# for some models, some data is processed in "chunks" -> speeds up simulation. Too large chunks reduce model accuracy.

MANUAL_CAL_AGING_EXPERIMENT_SHOW = False  # True  # if True, plot results of the manual calendar aging experiment
MANUAL_CAL_AGING_EXPERIMENT_SHOW_COL = 0  # show in first column
MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC = 18.0  # in °C, temperatures, at which the cells rested (weighted average?)
MANUAL_CAL_AGING_EXPERIMENT_V_USE_V = 3.5558  # in V, voltage, at which the cells rested (weighted average?)
MANUAL_CAL_AGING_EXPERIMENT_SOC_USE_PERCENT = 26.35  # in percent, 100.0 = 100 %, SoC, at which the cells rested
MANUAL_CAL_AGING_EXPERIMENT_TIME = 1707159600 - cfg.EXPERIMENT_START_TIMESTAMP  # 1707159600: 2024-02-05 20:00 GMT+1
MANUAL_CAL_AGING_EXPERIMENT_CAP_REMAINING = [2.8979, 2.9115, 2.9087]
MANUAL_CAL_AGING_EXPERIMENT_TIME_RESOLUTION = 30  # in seconds, resolution to use for modeling

# --- plot output format ---
USE_COMPACT_PLOT = True
PLOT_AGING_TYPE = False  # True
SHOW_IN_BROWSER_IF_STEPPING = False  # if True, open interactive plots in the browser when STEPPING_MODE == 1
SHOW_IN_BROWSER_IF_LIST = True  # if True, open interactive plots in the browser when STEPPING_MODE == 2
SHOW_IN_BROWSER_IF_MANUAL = True  # if True, open interactive plots in the browser when STEPPING_MODE == 0
EXPORT_HTML = True  # save interactive plots as html
EXPORT_IMAGE = False  # static plot images instead of interactive plots
REQUIRE_CFG = True  # probably can also be set to False if neither SHOW_IN_BROWSER / EXPORT_HTML / EXPORT_IMAGE enabled.
#                     if you use this script to fit other data: generate a CFG file for them, or set REQUIRE_CFG = False
# IMAGE_FORMAT = "jpg"  # -> tends to be larger than png
IMAGE_FORMAT = "png"  # -> easiest to view
# IMAGE_FORMAT = "svg"  # for EOC plots, this is smaller than png, and crisper of course
# IMAGE_FORMAT = "pdf"  # for EOC plots, this is by far the smallest, HOWEVER, there are tiny but annoying grid lines
USE_MANUAL_PLOT_LIMITS_CAL = True  # False
USE_MANUAL_PLOT_LIMITS_CYC = True
USE_MANUAL_PLOT_LIMITS_PRF = True  # False
MANUAL_PLOT_LIMITS_Y_CAL = [2.2, 3.1]  # [1.42, 3.02]  # [2.25, 3.0]
MANUAL_PLOT_LIMITS_Y_CYC = [1.4, 3.1]  # [1.42, 3.02]
MANUAL_PLOT_LIMITS_Y_PRF = [1.4, 3.1]  # [1.42, 3.02]

PLOT_FILENAME_BASE = f"%s_%s_plot_%s_%s"  # model name, model id, age mode, timestamp

PLOT_SCALE_FACTOR = 1.0  # for image export (to have better/higher resolution than specified in PLOT_WIDTH/PLOT_HEIGHT)
if (IMAGE_FORMAT == "jpg") or (IMAGE_FORMAT == "png"):
    PLOT_SCALE_FACTOR = 3.0  # scale up image dimensions by a factor of 3

IMAGE_EXPORT_ENGINE = 'kaleido'
# IMAGE_EXPORT_ENGINE = 'orca'  # you might try this engine if you get warnings in kaleido (need to install orca, see
# #                               config_main.py)

# ======================================================================================================================
# the following settings can usually stay like this but feel free to adjust them if needed
# you might want to adjust NUMBER_OF_PROCESSORS_TO_USE though (below)

# --- plot formatting --------------------------------------------------------------------------------------------------
SUBPLOT_LR_MARGIN = 30
SUBPLOT_TOP_MARGIN = 130  # 120  # 0
SUBPLOT_BOT_MARGIN = 0
SUBPLOT_PADDING = 0

if USE_COMPACT_PLOT:
    SUBPLOT_H_SPACING_REL = 0.155
    SUBPLOT_V_SPACING_REL = 0.35
    HEIGHT_PER_ROW = 250  # in px
    PLOT_WIDTH = 1175  # in px
else:
    SUBPLOT_H_SPACING_REL = 0.25  # 0.25  # 0.2  # 0.12  # 0.03, was 0.04
    SUBPLOT_V_SPACING_REL = 0.35  # 0.3  # 0.35  # 0.21  # was 0.035
    HEIGHT_PER_ROW = 300  # in px
    PLOT_WIDTH = 1350  # 1850  # in px

# PLOT_HEIGHT = HEIGHT_PER_ROW * SUBPLOT_ROWS -> we need to figure this out dynamically for each plot

# Title: model, RMSE, aging type, var string
PLOT_TITLE_RE = "<b>Usable discharge capacity [Ah] over time [days] – %s – %s - %s</b><br>%s"
PLOT_TITLE_Y_POS_REL = 20.0  # 30.0
AGE_TYPE_TITLES = {cfg.age_type.CALENDAR: "calendar aging",
                   cfg.age_type.CYCLIC: "cyclic aging",
                   cfg.age_type.PROFILE: "profile aging"
                   }

TIME_DIV = 24.0 * 60.0 * 60.0
TIME_UNIT = "days"
X_AXIS_TITLE = 'Time [' + TIME_UNIT + ']'
Y_AXIS_TITLE = "Capacity [Ah]"  # "usable dischg. capacity [Ah]"
TRACE_NAME_CU = f"P%03u-%01u (S%02u:C%02u)"
TRACE_NAME_MDL = f"%s for P%03u-%01u"

TITLE_FONT_SIZE = 17
SUBPLOT_TITLE_FONT_SIZE = 16
AXIS_FONT_SIZE = 16
AXIS_TICK_FONT_SIZE = 14
ANNOTATION_FONT_SIZE = 12
ANNOTATION_OPACITY = 0.8

# figure settings
FIGURE_TEMPLATE = "custom_theme"  # "custom_theme" "plotly_white" "plotly" "none"
# PLOT_LINE_OPACITY = 0.8
# PLOT_HOVER_TEMPLATE_1 = "<b>%s</b><br>"
# PLOT_HOVER_TEMPLATE_2 = "X %{x:.2f}, Y: %{y:.2f}<br><extra></extra>"
PLOT_HOVER_TEMPLATE_CU = "<b>%{text}</b><br>Remaining usable discharge capacity: %{y:.4f} Ah<br><extra></extra>"
PLOT_HOVER_TEMPLATE_MDL = "<b>after %{x:.1f} " + TIME_UNIT + ("</b><br>Remaining usable discharge capacity:"
                                                              " %{y:.4f} Ah<br><extra></extra>")
PLOT_HOVER_TEMPLATE_LOSS_TYPE_1 = "<b>after %{x:.1f} " + TIME_UNIT + "</b><br> only "
PLOT_HOVER_TEMPLATE_LOSS_TYPE_2 = " losses : %{y:.4f} Ah<br><extra></extra>"
PLOT_TEXT_CU = f"CU #%u after %.1f " + TIME_UNIT + ", %.2f EFCs"
PLOT_TEXT_MDL = f"after %.1f " + TIME_UNIT + ", %.2f EFCs"

# TEXT_RMSE = f"%.4f %%"
# TEXT_RMSE_CELL = f"RMSE cell %u: " + TEXT_RMSE
# TEXT_RMSE_PARAM = f"RMSE parameter: " + TEXT_RMSE
# TEXT_RMSE_TOTAL = f"RMSE total: " + TEXT_RMSE
# TEXT_POS_RMSE_X = 0.97  # 0.02
# TEXT_POS_RMSE_Y_BASE = 0.95  # 0.02
# TEXT_POS_RMSE_DY = 0.08  # 0.08
# TEXT_POS_RMSE_DY_OFFSET = -1  # 3
# TEXT_POS_RMSE_DY_FACTOR = -1  # ?

# TEXT_RMSE = f"%.4f %%"
TEXT_RMSE = f"%.2f%%/%.2f%%"  # f"%.3f %% / %.3f %%"
TEXT_RMSE_TITLE = f"%.4f %% / %.4f %%"  # f"%.3f %% / %.3f %%"
TEXT_RMSE_CELL = f"%u: " + TEXT_RMSE
TEXT_RMSE_PARAM = f"P: " + TEXT_RMSE
# TEXT_RMSE_EXPLAIN_DOUBLE = f"all / only ≥.7"
TEXT_RMSE_TOTAL = f"RMSE%% total: " + TEXT_RMSE_TITLE

TEXT_POS_RMSE_X = [0.99, 0.01]  # [0.985, 0.015]  # 0.97  # 0.02
TEXT_POS_RMSE_Y_BASE = [0.99, 0.01]  # [0.985, 0.015]  # 0.95  # 0.02
TEXT_POS_RMSE_DY = 0.105  # 0.115  # 0.08
TEXT_POS_RMSE_DY_OFFSET = [-1, 2]
TEXT_POS_RMSE_DY_FACTOR = [-1, -1]
TEXT_POS_RMSE_PARAM_OFFSET = [0, 3]

# TEXT_POS_RMSE_DY_OFFSET = [-2, 2]  # 3
# TEXT_POS_RMSE_DY_FACTOR = [-1, -1]  # ?
# TEXT_POS_RMSE_PARAM_OFFSET = [-1, 3]
# TEXT_POS_RMSE_EXPLANATION_OFFSET = [0, 4]

# TEXT_POS_RMSE_INDEX[i_fig][i_row][i_col]: how is the RMSE text aligned? Default: first item in TEXT_POS_RMSE_...
# # if only USE_MANUAL_PLOT_LIMITS_CYC is True:
# TEXT_POS_RMSE_INDEX = [  # 0: north-east, 1: south-west
#     [],  # manual
#     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]],  # calendar
#     [[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
#      [0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1],
#      [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1],
#      [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],  # cyclic
#     [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],  # profile
# ]
# if all USE_MANUAL_PLOT_LIMITS_... are True:
TEXT_POS_RMSE_INDEX = [  # 0: north-east, 1: south-west
    [],  # manual
    [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],  # calendar
    [[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1],
     [0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1],
     [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1],
     [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],  # cyclic
    [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 1]],  # profile
]


BG_COLOR = 'rgba(255, 255, 255, 127)'  # '#fff'
MAJOR_GRID_COLOR = '#bbb'
MINOR_GRID_COLOR = '#e8e8e8'  # '#ddd'

TRACE_OPACITY = 0.4
TRACE_LINE_WIDTH = 1.5
TRACE_COLORS = ['rgb(239,65,54)', 'rgb(1,147,70)', 'rgb(43,111,183)']  # for parameter_nr 1, 2, 3 (here: R, G, B)
MARKER_OPACITY = 0.8  # 75
MARKER_STYLE = dict(size=5, opacity=MARKER_OPACITY, line=None, symbol='circle')
PARAM_RMSE_COLOR = 'rgb(63, 63, 63)'

LOSS_TYPE_SEI = "SEI"
LOSS_TYPE_CYC1 = "cyc,I"
LOSS_TYPE_CYC2 = "cyc,II"
LOSS_TYPE_PLATING = "pl"
# LOSS_TYPES = [LOSS_TYPE_SEI, LOSS_TYPE_CYC1, LOSS_TYPE_CYC2, LOSS_TYPE_PLATING]
# LOSS_TYPE_COLORS = ["rgba(217,149,143,%.3f)", "rgba(249,196,153,%.3f)",
#                     "rgba(255,217,101,%.3f)", "rgba(146,205,220,%.3f)"]
LOSS_TYPES = [LOSS_TYPE_PLATING, LOSS_TYPE_CYC2, LOSS_TYPE_CYC1, LOSS_TYPE_SEI]
LOSS_TYPE_COLORS = ["rgba(146,205,220,%.3f)", "rgba(196,214,160,%.3f)",
                    "rgba(255,217,101,%.3f)", "rgba(217,149,143,%.3f)"]
# LOSS_TYPE_COLORS = ["rgba(146,205,220,%.3f)", "rgba(255,217,101,%.3f)",
#                     "rgba(249,196,153,%.3f)", "rgba(217,149,143,%.3f)"]
TRACE_OPACITY_AGE_TYPE = 0.24
FILL_OPACITY_AGE_TYPE = 0.06

FILL_COLOR_EDGE = 'rgba(127, 127, 127, 0.3)'
FILL_COLOR = 'rgba(127, 127, 127, 0.1)'  # 'rgba(127, 127, 127, 63)'

# create custom theme from default plotly theme
pio.templates["custom_theme"] = pio.templates["plotly"]
pio.templates["custom_theme"]['layout']['paper_bgcolor'] = BG_COLOR
pio.templates["custom_theme"]['layout']['plot_bgcolor'] = BG_COLOR
pio.templates["custom_theme"]['layout']['hoverlabel']['namelength'] = -1
pio.templates['custom_theme']['layout']['xaxis']['gridcolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['yaxis']['gridcolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['yaxis']['zerolinecolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['xaxis']['linecolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['yaxis']['linecolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['yaxis']['zerolinecolor'] = MAJOR_GRID_COLOR
pio.templates['custom_theme']['layout']['xaxis']['title']['standoff'] = 10
pio.templates['custom_theme']['layout']['yaxis']['title']['standoff'] = 12
pio.templates['custom_theme']['layout']['xaxis']['title']['font']['size'] = AXIS_FONT_SIZE
pio.templates['custom_theme']['layout']['yaxis']['title']['font']['size'] = AXIS_FONT_SIZE
pio.templates['custom_theme']['layout']['xaxis']['tickfont']['size'] = AXIS_TICK_FONT_SIZE
pio.templates['custom_theme']['layout']['yaxis']['tickfont']['size'] = AXIS_TICK_FONT_SIZE
pio.templates['custom_theme']['layout']['annotationdefaults']['font']['size'] = ANNOTATION_FONT_SIZE


# --- log_df reading ---------------------------------------------------------------------------------------------------
LOAD_COLUMNS = [csv_label.TIMESTAMP, csv_label.V_CELL, csv_label.OCV_EST, csv_label.I_CELL, csv_label.T_CELL,
                csv_label.SOC_EST, csv_label.DELTA_Q, csv_label.EFC,
                csv_label.CAP_CHARGED_EST]  # , csv_label.R0, csv_label.R1

COLUMN_DTYPES = {
    csv_label.TIMESTAMP: np.float64,
    csv_label.V_CELL: np.float32,
    csv_label.OCV_EST: np.float32,
    csv_label.I_CELL: np.float32,
    csv_label.T_CELL: np.float32,
    csv_label.SOC_EST: np.float32,
    csv_label.DELTA_Q: np.float32,
    csv_label.EFC: np.float32,
    csv_label.CAP_CHARGED_EST: np.float32,
    # csv_label.R0: np.float32,
    # csv_label.R1: np.float32
}

# --- other constants --------------------------------------------------------------------------------------------------
# ToDo: select numbers of CPU cors used for optimization (to run aging modeling for cells in parallel)
# NUMBER_OF_PROCESSORS_TO_USE = 1  # only use one core
NUMBER_OF_PROCESSORS_TO_USE = math.ceil(multiprocessing.cpu_count() / 2)  # use 50% of the cores
# NUMBER_OF_PROCESSORS_TO_USE = max(multiprocessing.cpu_count() - 1, 1)  # leave one free
# NUMBER_OF_PROCESSORS_TO_USE = multiprocessing.cpu_count()  # use all cores

OPT_FUNC_ID = "function_id"
OPT_FUNC_NAME = "function_name"
OPT_VARS = "variables"
OPT_VAR_LIMS = "variable_limits"
OPT_METHOD = "optimizer_method"
OPT_USE_CAP_NOM = "optimizer_use_cap_nom"
OPT_USE_T_PRODUCTION = "optimizer_use_t_production"

DF_COL_AGE_CAP_DELTA_APPENDIX = "_delta_cap_age"
DF_COL_AGE_MODEL_APPENDIX = "_age"
DF_COL_TIME_USE = csv_label.TIMESTAMP_ORIGIN + "_use"
DF_COL_AGE_CAP_DELTA_FILL = 0.0
DF_COL_AGE_MODEL_FILL = np.nan

# run_timestring = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

AGE_TYPE_FILENAMES = {cfg.age_type.CALENDAR: 'CAL',
                      cfg.age_type.CYCLIC: 'CYC',
                      cfg.age_type.PROFILE: 'PRF'}

LABEL_C_CHG = csv_label.I_CELL + "_chg_rel"
# LABEL_Q_LOSS_SEI = "Q_loss_SEI"
# LABEL_Q_LOSS_wearout = "Q_loss_wearout"
# LABEL_Q_LOSS_PLATING = "Q_loss_plating"

logging_filename = "09_log.txt"
logging = config_logging.bat_data_logger(cfg.LOG_DIR + logging_filename)

# tweaks to make it more likely that large image exports work:
pio.orca.config.executable = cfg.ORCA_PATH
# pio.kaleido.scope.mathjax = None
# pio.kaleido.scope.chromium_args += ("--single-process",)
pio.orca.config.timeout = 600  # increase timeout from 30 seconds to 10 minutes

modeling_task_queue = multiprocessing.Queue()

if STEPPING_MODE == 1:
    SHOW_IN_BROWSER = SHOW_IN_BROWSER_IF_STEPPING
elif STEPPING_MODE == 2:
    SHOW_IN_BROWSER = SHOW_IN_BROWSER_IF_LIST
else:
    SHOW_IN_BROWSER = SHOW_IN_BROWSER_IF_MANUAL

# global variables
iteration = 0


# exceptions
class ProcessingFailure(Exception):
    pass


# main function
def run():
    start_timestamp = datetime.now()
    logging.log.info(os.path.basename(__file__))

    optimizer_main()  # <-- this is the "actual" function doing the work

    # logging.log.info("\n\n========== All tasks ended - summary ==========\n")
    logging.log.info("\n\n========== All tasks ended ==========\n")

    stop_timestamp = datetime.now()
    logging.log.info("\nScript runtime: %s h:mm:ss.ms" % str(stop_timestamp - start_timestamp))


# main optimization function
def optimizer_main():
    global iteration
    global STEPPING_MODE
    global SHOW_IN_BROWSER
    # === Load log_age csv files to RAM ================================================================================
    cell_log_csv = []  # find .csv files: cell_log_age_30s_P012_3_S14_C11.csv
    cell_cfg_csv = []  # find .csv files: cell_cfg_P012_3_S14_C11.csv
    slave_cell_found = [[" "] * cfg.NUM_CELLS_PER_SLAVE for _ in range(cfg.NUM_SLAVES_MAX)]
    with os.scandir(cfg.CSV_RESULT_DIR) as iterator:
        re_str_log_csv = cfg.CSV_FILENAME_05_RESULT_BASE_CELL_RE.replace("(\w)", cfg.CSV_FILENAME_08_TYPE_LOG_AGE)
        re_str_log_csv = re_str_log_csv.replace("%u", "(\\d+)")
        re_pat_log_csv = re.compile(re_str_log_csv)
        re_str_cfg_csv = cfg.CSV_FILENAME_05_RESULT_BASE_CELL_RE.replace("(\w)", cfg.CSV_FILENAME_05_TYPE_CONFIG)
        re_pat_cfg_csv = re.compile(re_str_cfg_csv)
        for entry in iterator:  # walk through files
            re_match_log_csv = re_pat_log_csv.fullmatch(entry.name)
            re_match_cfg_csv = re_pat_cfg_csv.fullmatch(entry.name)
            if re_match_log_csv:  # is this a LOG_AGE file?
                resolution = int(re_match_log_csv.group(1))
                if RESOLUTION_USE is not None:
                    if not (resolution in RESOLUTION_USE):
                        logging.log.debug("Skipping '%s' because it doesn't match RESOLUTION_USE (%s)"
                                          % (entry.name, str(RESOLUTION_USE)))
                        continue
                param_id = int(re_match_log_csv.group(2))
                param_nr = int(re_match_log_csv.group(3))
                slave_id = int(re_match_log_csv.group(4))
                cell_id = int(re_match_log_csv.group(5))
                cell_csv = {"resolution_s": resolution,
                            csv_label.PARAMETER_ID: param_id, csv_label.PARAMETER_NR: param_nr,
                            csv_label.SLAVE_ID: slave_id, csv_label.CELL_ID: cell_id, "log_filename": entry.name}
                cell_log_csv.append(cell_csv)
                if (slave_id < 0) or (slave_id >= cfg.NUM_SLAVES_MAX):
                    logging.log.warning("Found unusual slave_id: %u" % slave_id)
                    num_warnings = num_warnings + 1
                else:
                    if (cell_id < 0) or (cell_id >= cfg.NUM_CELLS_PER_SLAVE):
                        logging.log.warning("Found unusual cell_id: %u" % cell_id)
                        num_warnings = num_warnings + 1
                    else:
                        if (slave_cell_found[slave_id][cell_id] == "l") or (slave_cell_found[slave_id][cell_id] == "b"):
                            logging.log.warning("Found more than one entry for S%02u:C%02u" % (slave_id, cell_id))
                            num_warnings = num_warnings + 1
                        elif slave_cell_found[slave_id][cell_id] == "c":
                            slave_cell_found[slave_id][cell_id] = "b"
                        else:
                            slave_cell_found[slave_id][cell_id] = "l"
            elif re_match_cfg_csv:  # or is this a CFG file?
                param_id = int(re_match_cfg_csv.group(1))
                param_nr = int(re_match_cfg_csv.group(2))
                slave_id = int(re_match_cfg_csv.group(3))
                cell_id = int(re_match_cfg_csv.group(4))
                cell_csv = {csv_label.PARAMETER_ID: param_id, csv_label.PARAMETER_NR: param_nr,
                            csv_label.SLAVE_ID: slave_id, csv_label.CELL_ID: cell_id, "cfg_filename": entry.name}
                cell_cfg_csv.append(cell_csv)
                if (slave_id < 0) or (slave_id >= cfg.NUM_SLAVES_MAX):
                    logging.log.warning("Found unusual slave_id: %u" % slave_id)
                    num_warnings = num_warnings + 1
                else:
                    if (cell_id < 0) or (cell_id >= cfg.NUM_CELLS_PER_SLAVE):
                        logging.log.warning("Found unusual cell_id: %u" % cell_id)
                        num_warnings = num_warnings + 1
                    else:
                        if (slave_cell_found[slave_id][cell_id] == "c") or (slave_cell_found[slave_id][cell_id] == "b"):
                            logging.log.warning("Found more than one entry for S%02u:C%02u" % (slave_id, cell_id))
                            num_warnings = num_warnings + 1
                        elif slave_cell_found[slave_id][cell_id] == "l":
                            slave_cell_found[slave_id][cell_id] = "b"
                        else:
                            slave_cell_found[slave_id][cell_id] = "c"

    num_parameters = 0
    num_cells_per_parameter = 0
    for cell_log in cell_log_csv:
        found_cfg = False
        for cell_cfg in cell_cfg_csv:
            if ((cell_log[csv_label.PARAMETER_ID] == cell_cfg[csv_label.PARAMETER_ID])
                    and (cell_log[csv_label.PARAMETER_NR] == cell_cfg[csv_label.PARAMETER_NR])
                    and (cell_log[csv_label.SLAVE_ID] == cell_cfg[csv_label.SLAVE_ID])
                    and (cell_log[csv_label.CELL_ID] == cell_cfg[csv_label.CELL_ID])):
                cell_log["cfg_filename"] = cell_cfg["cfg_filename"]
                found_cfg = True  # both LOG_AGE and CFG was found
                break
        if not found_cfg:
            if REQUIRE_CFG:
                continue  # skip cell
            cell_log["cfg_filename"] = ""  # else -> set empty

        slave_id = cell_log[csv_label.SLAVE_ID]
        cell_id = cell_log[csv_label.CELL_ID]
        if ((slave_id >= 0) and (slave_id < cfg.NUM_SLAVES_MAX)
                and (cell_id >= 0) and (cell_id < cfg.NUM_CELLS_PER_SLAVE)):
            param_id = cell_log[csv_label.PARAMETER_ID]
            if param_id in CELL_PARAMETERS_USE:
                slave_cell_found[slave_id][cell_id] = "X"
                param_nr = cell_log[csv_label.PARAMETER_NR]
                if param_id > num_parameters:
                    num_parameters = param_id
                if param_nr > num_cells_per_parameter:
                    num_cells_per_parameter = param_nr

    if MANUAL_CAL_AGING_EXPERIMENT_SHOW:
        num_parameters = num_parameters + 1
        num_manual_cal_aging_cells = len(MANUAL_CAL_AGING_EXPERIMENT_CAP_REMAINING)
        if num_manual_cal_aging_cells > num_cells_per_parameter:
            num_cells_per_parameter = num_manual_cal_aging_cells

    # List found slaves/cells for user
    pre_text = ("Found the following files:\n"
                "' ' = no file found, 'l' = LOG_AGE found, 'c' = config found, 'b' = both found,\n"
                "'X' = LOG_AGE and config of interest found & matching -> added\n")
    logging.log.info(ht.get_found_cells_text(slave_cell_found, pre_text))

    # read files and store in df
    logging.log.info("Reading LOG_AGE files...")
    log_dfs = [[pd.DataFrame() for _ in range(num_cells_per_parameter)] for _ in range(num_parameters)]
    cap_initial_avg = 0.0
    t_initial_avg = 0.0
    cap_initial_num = 0
    for cell_log in cell_log_csv:
        param_id = cell_log[csv_label.PARAMETER_ID]
        if param_id in CELL_PARAMETERS_USE:
            param_nr = cell_log[csv_label.PARAMETER_NR]
            logging.log.debug("P%03u-%u - reading LOG_AGE file" % (param_id, param_nr))
            log_fullpath = cfg.CSV_RESULT_DIR + cell_log["log_filename"]
            log_df = pd.read_csv(log_fullpath, header=0, sep=cfg.CSV_SEP, engine="pyarrow",
                                 usecols=LOAD_COLUMNS, dtype=COLUMN_DTYPES)

            if USE_ABS_TEMPERATURE:
                log_df.loc[:, csv_label.T_CELL] = log_df[csv_label.T_CELL] + cfg.T0

            log_df_with_cap_aged = log_df[~pd.isna(log_df[csv_label.CAP_CHARGED_EST])]
            if log_df_with_cap_aged.shape[0] > 0:
                cap_initial_avg = cap_initial_avg + log_df_with_cap_aged[csv_label.CAP_CHARGED_EST].iloc[0]
                t_initial_avg = t_initial_avg + log_df_with_cap_aged[csv_label.TIMESTAMP].iloc[0]
                cap_initial_num = cap_initial_num + 1

            log_dfs[param_id - 1][param_nr - 1] = log_df

    if cap_initial_num > 0:
        cap_initial_avg = cap_initial_avg / cap_initial_num
        t_initial_avg = t_initial_avg / cap_initial_num
        t_initial_hours = t_initial_avg / 3600.0
        t_initial_days = math.floor(t_initial_hours / 24.0)
        t_initial_hours = t_initial_hours - t_initial_days * 24.0
        logging.log.info("Average initial capacity measurement: %.5f Ah (at t = ca. %.0f s = %.0f d, %.1f h)"
                         % (cap_initial_avg, t_initial_avg, t_initial_days, t_initial_hours))
    else:
        cap_initial_avg = cfg.CELL_CAPACITY_NOMINAL

    if MANUAL_CAL_AGING_EXPERIMENT_SHOW:
        # df_dtypes = []
        # for col in LOAD_COLUMNS:
        #     df_dtypes.append(COLUMN_DTYPES.get(col))
        # log_manual_cal_aging_df = pd.DataFrame(columns=LOAD_COLUMNS, dtype=df_dtypes)
        # log_manual_cal_aging_df = log_dfs[CELL_PARAMETERS_USE[0]][0].head(0).copy()

        num_timestamps = math.ceil(MANUAL_CAL_AGING_EXPERIMENT_TIME / MANUAL_CAL_AGING_EXPERIMENT_TIME_RESOLUTION)
        log_manual_cal_aging_df = pd.DataFrame(index=range(0, num_timestamps), columns=LOAD_COLUMNS, dtype="float64")
        log_manual_cal_aging_df.loc[:, csv_label.TIMESTAMP] = (np.array(range(0, num_timestamps)).astype(np.float64)
                                                               * MANUAL_CAL_AGING_EXPERIMENT_TIME_RESOLUTION)
        log_manual_cal_aging_df.loc[:, csv_label.V_CELL] = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        log_manual_cal_aging_df.loc[:, csv_label.OCV_EST] = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        log_manual_cal_aging_df.loc[:, csv_label.I_CELL] = 0.0
        if USE_ABS_TEMPERATURE:
            log_manual_cal_aging_df.loc[:, csv_label.T_CELL] = MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC + cfg.T0
        else:
            log_manual_cal_aging_df.loc[:, csv_label.T_CELL] = MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC
        log_manual_cal_aging_df.loc[:, csv_label.SOC_EST] = MANUAL_CAL_AGING_EXPERIMENT_SOC_USE_PERCENT
        log_manual_cal_aging_df.loc[:, csv_label.DELTA_Q] = 0.0
        log_manual_cal_aging_df.loc[:, csv_label.EFC] = 0.0
        log_manual_cal_aging_df.loc[:, csv_label.CAP_CHARGED_EST] = np.nan
        tx = log_manual_cal_aging_df[csv_label.TIMESTAMP][log_manual_cal_aging_df[csv_label.TIMESTAMP] > t_initial_avg]
        i = 0
        if tx.shape[0] > 0:
            i = tx.index[0]
        log_manual_cal_aging_df.loc[i, csv_label.CAP_CHARGED_EST] = cap_initial_avg

        for col in LOAD_COLUMNS:
            if col in COLUMN_DTYPES:
                dty = COLUMN_DTYPES.get(col)
                # log_manual_cal_aging_df.loc[:, col] = log_manual_cal_aging_df.loc[:, col].astype(dty)
                log_manual_cal_aging_df[col] = log_manual_cal_aging_df[col].astype(dty)

        for i in range(len(MANUAL_CAL_AGING_EXPERIMENT_CAP_REMAINING)):
            log_manual_cal_aging_df.loc[num_timestamps - 1, csv_label.CAP_CHARGED_EST] = (
                MANUAL_CAL_AGING_EXPERIMENT_CAP_REMAINING[i])
            log_dfs[num_parameters - 1][i] = log_manual_cal_aging_df.copy()

    logging.log.info("...done reading LOG_AGE files")

    # generate base plots
    logging.log.info("generating plot templates")
    fig_list, fig_and_sp_from_pid_arr, param_df = generate_base_figures(
        cell_log_csv, slave_cell_found, num_parameters, num_cells_per_parameter)
    fig_list = fill_figures_with_measurements(fig_list, fig_and_sp_from_pid_arr, log_dfs, param_df)

    # ToDo: if you add new models, please adjust the following accordingly:
    # dictionary: model string -> model function
    model_function_dict = {'model_f001': model_f001, 'model_f002': model_f002, 'model_f003': model_f003,
                           'model_f004': model_f004, 'model_f005': model_f005, 'model_f006': model_f006,
                           'model_f007': model_f007, 'model_f008': model_f008, 'model_f009': model_f009,
                           'model_f010': model_f010, 'model_f011': model_f011, 'model_f012': model_f012,
                           'model_f013': model_f013, 'model_f013p': model_f013p, 'model_f014': model_f014,
                           'model_f015': model_f015, 'model_f016': model_f016, 'model_f017': model_f017,
                           'model_f018': model_f018, 'model_f019': model_f019, 'model_f020': model_f020,
                           'model_f021': model_f021, 'model_f022': model_f022, 'model_f023': model_f023,
                           'model_f024': model_f024, 'model_f025': model_f025, 'model_f026': model_f026,
                           'model_f027': model_f027, 'model_f028': model_f028, 'model_f029': model_f029,
                           'model_f030': model_f030, 'model_f031': model_f031, 'model_f032': model_f032,
                           'model_f033': model_f033, 'model_f034': model_f034, 'model_f035': model_f035,
                           'model_f036': model_f036, 'model_f037': model_f037, 'model_f038': model_f038,
                           'model_f039': model_f039, 'model_f040': model_f040, 'model_f041': model_f041,
                           'model_f042': model_f042, 'model_f043': model_f043, 'model_f044': model_f044,
                           'model_f045': model_f045, 'model_f046': model_f046, 'model_f047': model_f047,
                           'model_f048': model_f048, 'model_f049': model_f049, 'model_f050': model_f050,
                           'model_f051': model_f051, 'model_f052': model_f052, 'model_f053': model_f053,
                           'model_f054': model_f054, 'model_f055': model_f055, 'model_f056': model_f056}
    # dictionary: model string -> unique model (base) ID
    model_id_dict = {'model_f001': 1000, 'model_f002': 2000, 'model_f003': 3000,
                     'model_f004': 4000, 'model_f005': 5000, 'model_f006': 6000,
                     'model_f007': 7000, 'model_f008': 8000, 'model_f009': 9000,
                     'model_f010': 10000, 'model_f011': 11000, 'model_f012': 12000,
                     'model_f013': 13000, 'model_f013p': 13100, 'model_f014': 14000,
                     'model_f015': 15000, 'model_f016': 16000, 'model_f017': 17000,
                     'model_f018': 18000, 'model_f019': 19000, 'model_f020': 20000,
                     'model_f021': 21000, 'model_f022': 22000, 'model_f023': 23000,
                     'model_f024': 24000, 'model_f025': 25000, 'model_f026': 26000,
                     'model_f027': 27000, 'model_f028': 28000, 'model_f029': 29000,
                     'model_f030': 30000, 'model_f031': 31000, 'model_f032': 32000,
                     'model_f033': 33000, 'model_f034': 34000, 'model_f035': 35000,
                     'model_f036': 36000, 'model_f037': 37000, 'model_f038': 38000,
                     'model_f039': 39000, 'model_f040': 40000, 'model_f041': 41000,
                     'model_f042': 42000, 'model_f043': 43000, 'model_f044': 44000,
                     'model_f045': 45000, 'model_f046': 46000, 'model_f047': 47000,
                     'model_f048': 48000, 'model_f049': 49000, 'model_f050': 50000,
                     'model_f051': 51000, 'model_f052': 52000, 'model_f053': 53000,
                     'model_f054': 54000, 'model_f055': 55000, 'model_f056': 56000}
    # opt_method_list = ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr' 'CG', 'BFGS', 'Newton-CG',
    #                    'COBYLA', 'trust-ncg', 'dogleg', 'trust-exact', 'trust-krylov']
    # dictionary: model string -> default start values proposed in user prompt
    opt_start_value_list = {
     'model_f001': "a0: 0.0001, a1: 1.068, a2: 0.991",
     'model_f002': "a0: 0.00005, a1: 1.046, a2: 1.134",
     'model_f003': "a0: 0.00005, a1: 0.3, a2: 0.0001",
     'model_f004': "a0: 0.00000001, a1: 0.01, a2: 0.5, a3: 0.0001, a4: 0.004",
     'model_f005': "a00: 0.000000017, a4: 0.00000006",
     'model_f006': "a0: 1.8e-8, a1: -7500, a2: 1.25, a3: 1.5e-8, a4: 6.0e-8",
     # 'model_f007': "a0: 0.74e-8, a1: -13500, a2: 0.12, a3: 1.95e-8, a4: 2.4e-6, a5: 1.0e-5",
     'model_f007': "a0: 0.74e-8, a1: -13500, a2: 0.12, a3: 0.5e-8, a4: 2.4e-6, a5: 1.0e-5",
     # 'model_f008': "a0: 2.47e-9, a1: -13500, a2: 0.12, a3: 1.95e-8, a4: 0.5, a5: 7.0e-7",
     # 'model_f008': "a0: 2.47e-9, a1: -13500, a2: 0.12, a3: 9.75e-8, a4: 0.0, a5: 7.0e-7",
     'model_f009': "a0: 2.47e-9, a1: -13500, a2: 0.12, a3: 9.75e-8, a4: 7.0e-7, a5: 7.0e-7",
     'model_f010': "a0: 0.75e-9, a1: -10000, a2: 0.12, a3: 1.2e-7, a4: 1.0e-7, a5: 35.0e-7, a6: 1.5e-8",
     'model_f011': "a0: 1.0e-9, a1: -2500, a2: 2.0, a3: 3.0e-7, a4: 0.5e-7, a5: 4.0e-6, a6: 1.8e-8",
     'model_f012': "a0: 1.0e-9, a1: -2500, a2: 2.0, a3: 0.5e-7, a4: 0.5, a5: 3.0e-7, a6: 4.0e-6, a7: 1.8e-8",
     'model_f013': "a0: 1.0e-9, a1: -2500, a2: 2.0, a3: 0.5e-7, a4: 3.0e-7, a5: 4.0e-6, a6: 0.05, a7: 0.1, a8: 1.0e-5",
     'model_f013p': "a0: 1.0e-9, a1: -2500, a2: 2.0, a3: 0.5e-7, a4: 3.0e-7, a5: 4.0e-6, a6: 0.05, a7: 0.1, a8: 1.0e-5",
     'model_f014': "a0: 1.0e-9, a1: -2500, a2: 2.0, a3: 0.5e-7, a4: 3.0e-7, "
                   "a5: 4.0e-6, a6: 0.16, a7: 0.0225, a8: 6.0e-7",
     'model_f015': "a0: 1.0e-9, a1: -2500, a2: 2.0, a3: 0.5e-7, a4: 3.0e-7, "
                   "a5: 4.0e-6, a6: 0.12, a7: 0.052, a8: 3.0e-7, a9: 2.0",
     'model_f016': "a0: 1.0e-9, a1: -2500, a2: 2.0, a3: 0.5e-7, a4: 3.0e-7, "
                   "a5: 4.0e-6, a6: 0.12, a7: 0.052, a8: 3.0e-7, a9: 2.0",
     'model_f017': "a0: 3.9e-9, a1: -1100, a2: 0.6, a3: 0.5e-7, a4: 0.75e-7, "
                   "a5: 1.28e-6, a6: 0.13, a7: 0.05, a8: 2.0e-7, a9: 1.9",
     'model_f018': "s0: 3.74e-9, s1: -1100, s2: 0.6, s3: 0.5e-7, "
                   "w0: 0.75e-7, w1: 3.0, w2: 1.28e-6, "
                   "p0: 0.45, p1: 0.05, p2: 0.30e-7, p3: 2.0, Vmid: 3.8",
     'model_f019': "s0: 3.74e-9, s1: -1100, s2: 0.6, s3: 0.5e-7, "
                   "w0: 0.75e-7, w1: 3.0, w2: 1.28e-6, "
                   "p0: 0.45, p1: 0.05, p2: 0.30e-7, p3: 2.0, Vmid: 3.8",
     'model_f020': "s0: 2.8e-9, s1: -1700, s2: 0.63, s3: 4.0e-8, "
                   "w0: 8.5e-8, w1: 2.0, w2: 1.2e-6, "
                   "p0: 0.15, p1: 0.095, p2: 5.0e-8, p3: 1.7, Vmid: 3.73",
     'model_f021': "s0: 2.8e-9, s1: -1700, s2: 0.63, s3: 4.0e-8, "
                   "w0: 8.5e-8, w1: 2.0, w2: 1.2e-6, "
                   "p0: 0.6, p1: 0.08, p2: 5.5e-9, p3: 3.0, "
                   "c0: 1.0e-10, c1: 2.5, c2: 1.0e-7, "
                   "Vmid: 3.73",
     'model_f022': "s0: 2.8e-9, s1: -1700, s2: 0.63, s3: 4.0e-8, "
                   "w0: 8.5e-8, w1: 2.0, w2: 1.2e-6, "
                   "p0: 0.6, p1: 0.08, p2: 5.5e-9, p3: 3.0, "
                   "c0: 2.5e-10, c1: 1.0e-7, "
                   "Vmid: 3.73",
     'model_f023': "s0: 2.8e-9, s1: -1700, s2: 0.63, s3: 4.0e-8, "
                   "w0: 8.5e-8, "
                   "p0: 0.6, p1: 0.08, p2: 5.5e-9, p3: 3.0, "
                   "c0: 2.5e-10, c1: 1.0e-7, "
                   "Vmid: 3.73",
     'model_f024': "s0: 1.35e-9, s1: -3700, s2: 1.2, s3: 4.3e-8, "
                   "w0: 0.55e-6, w1: 1.0, w2: 2.0e-6, "
                   "p0: 0.40, p1: 3000, p2: 0.8e-8, p3: 3.5, "
                   "c0: 1.1e-6, c1: 0.65e-5, "
                   "Vmid: 3.73",
     'model_f025': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 1.0, w2: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, "
                   "c0: 1.15e-6, c1: 1.05e-5, "
                   "Vmid: 3.73",
     'model_f026': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 1.0, w2: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, "
                   "c0: 1.15e-6, c1: 1.05e-5, "
                   "Vmid: 3.73",
     'model_f027': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 1.0, w2: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, "
                   "c0: 1.15e-6, c1: 1.05e-5, "
                   "Vmid: 3.73",
     'model_f028': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 1.0, w2: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, "
                   "c0: 1.15e-6, c1: 1.05e-5, "
                   "Vmid: 3.73",
     'model_f029': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, "
                   "c0: 1.15e-6, c1: 1.05e-5, "
                   "Vmid: 3.73",
     'model_f030': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, "
                   "c0: 1.15e-6, c1: 900, c2: 1.05e-5, "
                   "Vmid: 3.73",
     'model_f031': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, "
                   "c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, "
                   "Vmid: 3.73",
     'model_f032': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, "
                   "c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, "
                   "Vmid: 3.73",
     'model_f033': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, "
                   "c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, "
                   "Vmid: 3.73",
     'model_f034': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, "
                   "c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, "
                   "Vmid: 3.73",
     'model_f035': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, "
                   "c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, "
                   "Vmid: 3.73",
     'model_f036': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, "
                   "c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, "
                   "Vmid: 3.73",
     'model_f037': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, "
                   "c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, c4: 3.2, "
                   "Vmid: 3.73",
     'model_f038': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, p8: 1.0, "
                   "c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, c4: 3.2, "
                   "Vmid: 3.73",
     'model_f039': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, p8: 1.0, p9: 1.0, "
                   "c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, c4: 3.2, "
                   "Vmid: 3.73",
     'model_f040': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, p8: 1.0, p9: 1.0, "
                   "c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, c4: 3.2, "
                   "Vmid: 3.73",
     'model_f041': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, p8: 1.0, p9: 1.0, "
                   "c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, c4: 3.2, "
                   "Vmid: 3.73, Cn: 3.03",
     'model_f042': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, p8: 1.0, p9: 1.0, "
                   "c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, c4: 3.2, f0: 0.9,"
                   "Vmid: 3.73, Cn: 3.03",
     'model_f043': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, p8: 1.0, p9: 1.0, "
                   "c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, c4: 3.2, f0: 0.9,"
                   "Vmid: 3.73, Cn: 3.03",
     'model_f044': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 0.5, w4: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, p8: 1.0, p9: 1.0, "
                   "c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, c4: 3.2, f0: 0.9,"
                   "Vmid: 3.73, Cn: 3.03",
     'model_f045': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 0.5, w4: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, p8: 1.0, p9: 1.0, "
                   "p10: 1.0, p11: 3800, c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, c4: 3.2, f0: 0.9,"
                   "Vmid: 3.73, Cn: 3.03",
     'model_f046': "s0: 1.35e-9, s1: -3700, s2: 1.45, s3: 4.0e-8, "
                   "w0: 0.70e-6, w1: 2000, w2: 1.0, w3: 0.5, w4: 2.4e-6, "
                   "p0: 0.096, p1: 3800, p2: 0.45e-6, p3: 1.0, p4: 2.0, p5: 1.5, p6: 1.0, p7: 1.0, p8: 1.0, p9: 1.0, "
                   "p10: 1.0, p11: 3800, c0: 1.15e-6, c1: 900, c2: 0.2, c3: 1.05e-5, c4: 3.2, f0: 0.9,"
                   "Vmid: 3.73, Cn: 3.03",
     'model_f047': "s0: 2.85e-9, s1: -2475, s2: 1.2, s3: 3.783e-8, "
                   "w0: 9.2e-7, w1: -1500, w2: 1.0, w3: 0.333, w4: 8.8e-6, "
                   "p0: 0.16, p1: 1750, p2: 0.3, p3: 2.55e-7, p4: 1.0, p5: 5.0, p6: 0.38, p7: 5.0, "
                   "c0: 10.2e-6, c1: 1425, c2: 0, c3: 6.6e-5, c4: 3.2, f0: 0.3,"
                   "Vmid: 3.73, Cn: 3.06",
     'model_f048': "s0: 2.85e-9, s1: -2475, s2: 1.2, s3: 3.783e-8, "
                   "w0: 9.2e-7, w1: -1500, w2: 1.0, w3: 0.333, w4: 8.8e-6, "
                   "p0: 0.16, p1: 1750, p2: 0.3, p3: 2.55e-7, p4: 45.0, p5: 2.0, p6: 2.0, p7: 5.0, p8: 0.38, p9: 5.0, "
                   "c0: 10.2e-6, c1: 1425, c2: 0, c3: 6.6e-5, c4: 3.2, f0: 0.3,"
                   "Vmid: 3.73, Cn: 3.06",
     'model_f049': "s0: 2.85e-9, s1: -2475, s2: 1.2, s3: 3.783e-8, "
                   "w0: 9.2e-7, w1: -1500, w2: 1.0, w3: 0.333, w4: 8.8e-6, "
                   "p0: 0.155, p1: 1750, p2: 0.3, p3: 2.55e-7, p4: 45.0, p5: 2.0, p6: 2.0, p7: 5.0, p8: 0.38, p9: 5.0, "
                   "p10: 0.05, p11: 0.0099, p12: 70.0, c0: 10.2e-6, c1: 1425, c2: 0, c3: 6.6e-5, c4: 3.2, f0: 0.3,"
                   "Vmid: 3.73, Cn: 3.06",
     'model_f050': "s0:3.45e-9, s1:-2275, s2:1.0, s3:4.8e-8, "
                   "w0:12.4e-7, w1:-1300, w2:1, w3:0.5, w4:8.9e-6, "
                   # "p0:0.16, p1:0.0099, p2:0.3, p3:3.2e-4, p4:100.0, p5:4.0, p6:1.0, "
                   # "p7:5.5, p8:0.20, p9:5.0, p10:0.05, p11:70, p12:4, p13: 0.0, "  # add standard values for plating
                   "c0:11.5e-6, c1:2275, c2:0.0, c3:8.95e-5, c4:3.2, "
                   "f0:0.3, Vmid:3.73, Cn:3.075",
     'model_f051': "s0:3.45e-9, s1:-2275, s2:1.0, s3:4.8e-8, "
                   "w0:12.4e-7, w1:-1300, w2:1, w3:0.5, w4:8.9e-6, "
                   # "p0:0.16, p1:0.0099, p2:0.3, p3:3.2e-4, p4:100.0, p5:4.0, p6:1.0, "
                   # "p7:5.5, p8:0.20, p9:5.0, p10:0.05, p11:70, p12:4, p13: 0.0, "  # add standard values for plating
                   "c0:11.5e-6, c1:2275, c2:0.0, c3:8.95e-5, c4:3.2, "
                   "f0:0.3, Vmid:3.73, Cn:3.075",
     'model_f052': "s0:3.45e-9, s1:-2275, s2:1.0, s3:4.8e-8, "
                   "w0:12.4e-7, w1:-1300, w2:1, w3:0.5, w4: 5.0, w5: 0.25, w6:8.9e-6, "
                   # "p0:0.16, p1:0.0099, p2:0.3, p3:3.2e-4, p4:100.0, p5:4.0, p6:1.0, "
                   # "p7:5.5, p8:0.20, p9:5.0, p10:0.05, p11:70, p12:4, p13: 0.0, "  # add standard values for plating
                   "c0:11.5e-6, c1:2275, c2:0.0, c3:8.95e-5, c4:3.2, "
                   "f0:0.3, Vmid:3.73, Cn:3.075",
     'model_f053': "s0:3.45e-9, s1:-2275, s2:1.0, s3:4.8e-8, "
                   "w0:12.4e-7, w1:-1300, w2:1, w3:0.5, w4: 5.0, w5: 0.25, w6:8.9e-6, "
                   # "p0:0.16, p1:0.0099, p2:0.3, p3:3.2e-4, p4:100.0, p5:4.0, p6:1.0, "
                   # "p7:5.5, p8:0.20, p9:5.0, p10:0.05, p11:70, p12:4, p13: 0.0, "  # add standard values for plating
                   "c0:11.5e-6, c1:2275, c2:0.0, c3:8.95e-5, c4:3.2, "
                   "f0:0.3, Vmid:3.73, Cn:3.075",
     'model_f054': "s0:3.45e-9, s1:-2275, s2:1.0, s3:4.8e-8, "
                   "w0:12.4e-7, w1:-1300, w2:1, w3:0.5, w4: 5.0, w5: 0.25, w6:8.9e-6, "
                   # "p0:0.16, p1:0.0099, p2:0.3, p3:3.2e-4, p4:100.0, p5:4.0, p6:1.0, "
                   # "p7:5.5, p8:0.20, p9:5.0, p10:0.05, p11:70, p12:4"  # add standard values for plating
                   "c0:11.5e-6, c1:2275, c2:0.0, c3:8.95e-5, c4:3.2, "
                   "f0:0.3, Vmid:3.73, Cn:3.075",
     'model_f055': "s0: 4.82e-9, s1: -1975, s2: 1.0, s3: 5.05e-8, "
                   "w0: 7.3e-7, w1: 2.55, w2: 0.163, w3: 2.45e-6, "
                   "p0: 0.078, p1: 0.0320, p2: 40, p3: 3.79, p4: 0.37, p5: 5.4e-8, p6: 2, "
                   "c0: 1.25e-4, c1: 950, c2: 3.2, c3: 8.95e-4, "
                   "f0: 0.3, Vm: 3.73, Ci: 3.075",
     'model_f056': "s0: 1.49e-9, s1: -2375, s2: 1.2, s3: 1.78e-8, "
                   "w0: 2.67e-7, w1: 2.25, w2: 0.14, w3: 9.5e-7, "
                   "p0: 0.07, p1: 0.029, p2: 314.65, p3: 3.5, p4: 0.33, p5: 5.3e-8, p6: 2.15, "
                   "c0: 3.60e-5, c1: 1050, c2: 3.2, c3: 2.47e-4, "
                   "Vm: 3.73, Ci: 3.09",
    }
    # ToDo: depending on which "STEPPING_MODE" you selected, adjust the following if necessary
    if STEPPING_MODE == 1:  # stepping matrix-wise (brute force / exhaustive testing using selected variables)
        model_name = "model_f054"  # in the dictionaries above
        use_cap_nom = 1.0  # see notes for USE_NOMINAL_CAPACITY_DEFAULT
        use_t_prod = True  # see notes for USE_T_PRODUCTION_DEFAULT
        kvarr_dict = {"s0": [3.45e-9],  # model variables (depending on the model!) and (list of) values to test
                      "s1": [-1975],
                      "s2": [0.85],
                      "s3": [5.1e-8],
                      "w0": [6.5e-7, 7.3e-7],  # 6.9e-7
                      "w1": [0.0],
                      "w2": [1.0],
                      "w3": [1.0],
                      "w4": [2.55, 3.05],  # 2.8
                      "w5": [0.165, 0.195],  # 0.18
                      "w6": [2.4e-6, 3.0e-6],  # 2.7e-6
                      "p0": [0.068, 0.078],  # 0.073
                      "p1": [0.0313, 0.0359],  # 0.0336
                      "p2": [40, 45],  # 40
                      "p3": [3.54, 3.86],  # 3.7
                      "p4": [0.37, 0.43],  # 0.4
                      "p5": [0.0],
                      "p6": [4.9e-8],  # 4.9e-8, 5.4e-8, 4.4e-8
                      "p7": [0.0],
                      "p8": [1.85, 2.0],
                      "p9": [0.0],
                      "p10": [0.0],
                      "p11": [0.0],
                      "p12": [0.0],
                      "c0": [1.3e-4],
                      "c1": [950.0],
                      "c2": [1.0],
                      "c3": [8.95e-4],
                      "c4": [3.2],
                      "f0": [0.3],
                      "Vmid": [3.73],
                      "Cn": [3.09],
                      }
        # create empty dict (this can also be done nicer for sure)
        kv_dict = {"s0": 0.0, "s1": 0.0, "s2": 0.0, "s3": 0.0, "w0": 0.0, "w1": 0.0, "w2": 0.0, "w3": 0.0, "w4": 0.0,
                   "w5": 0.0, "w6": 0.0, "p0": 0.0, "p1": 0.0, "p2": 0.0, "p3": 0.0, "p4": 0.0, "p5": 0.0, "p6": 0.0,
                   "p7": 0.0, "p8": 0.0, "p9": 0.0, "p10": 0.0, "p11": 0.0, "p12": 0.0,
                   "c0": 0.0, "c1": 0.0, "c2": 0.0, "c3": 0.0, "c4": 0.0, "f0": 0.0, "Vmid": 0.0, "Cn": 0.0}

        # no need to change these three lines:
        model_id = model_id_dict.get(model_name)
        model_function = model_function_dict.get(model_name)
        str_base = "Starting process to model with\n   %s"

        # add default value (this can also be done nicer for sure)
        kv_dict["s0"] = kvarr_dict["s0"][0]
        kv_dict["s1"] = kvarr_dict["s1"][0]
        kv_dict["s2"] = kvarr_dict["s2"][0]
        kv_dict["s3"] = kvarr_dict["s3"][0]
        kv_dict["w0"] = kvarr_dict["w0"][0]
        kv_dict["w1"] = kvarr_dict["w1"][0]
        kv_dict["w2"] = kvarr_dict["w2"][0]
        kv_dict["w3"] = kvarr_dict["w3"][0]
        kv_dict["w4"] = kvarr_dict["w4"][0]
        kv_dict["w5"] = kvarr_dict["w5"][0]
        kv_dict["w6"] = kvarr_dict["w6"][0]
        kv_dict["p0"] = kvarr_dict["p0"][0]
        kv_dict["p1"] = kvarr_dict["p1"][0]
        kv_dict["p2"] = kvarr_dict["p2"][0]
        kv_dict["p3"] = kvarr_dict["p3"][0]
        kv_dict["p4"] = kvarr_dict["p4"][0]
        kv_dict["p5"] = kvarr_dict["p5"][0]
        kv_dict["p6"] = kvarr_dict["p6"][0]
        kv_dict["p7"] = kvarr_dict["p7"][0]
        kv_dict["p8"] = kvarr_dict["p8"][0]
        kv_dict["p9"] = kvarr_dict["p9"][0]
        kv_dict["p10"] = kvarr_dict["p10"][0]
        kv_dict["p11"] = kvarr_dict["p11"][0]
        kv_dict["p12"] = kvarr_dict["p12"][0]
        kv_dict["c0"] = kvarr_dict["c0"][0]
        kv_dict["c1"] = kvarr_dict["c1"][0]
        kv_dict["c2"] = kvarr_dict["c2"][0]
        kv_dict["c3"] = kvarr_dict["c3"][0]
        kv_dict["c4"] = kvarr_dict["c4"][0]
        kv_dict["f0"] = kvarr_dict["f0"][0]
        kv_dict["Vmid"] = kvarr_dict["Vmid"][0]
        kv_dict["Cn"] = kvarr_dict["Cn"][0]

        # iterate through all values that we want to vary (this can also be done nicer for sure)
        for i_p2 in range(len(kvarr_dict["p2"])):
            kv_dict["p2"] = kvarr_dict["p2"][i_p2]
            for i_p0 in range(len(kvarr_dict["p0"])):
                kv_dict["p0"] = kvarr_dict["p0"][i_p0]
                for i_p1 in range(len(kvarr_dict["p1"])):
                    kv_dict["p1"] = kvarr_dict["p1"][i_p1]
                    for i_p3 in range(len(kvarr_dict["p3"])):
                        kv_dict["p3"] = kvarr_dict["p3"][i_p3]
                        for i_p4 in range(len(kvarr_dict["p4"])):
                            kv_dict["p4"] = kvarr_dict["p4"][i_p4]
                            for i_p8 in range(len(kvarr_dict["p8"])):
                                kv_dict["p8"] = kvarr_dict["p8"][i_p8]
                                for i_w0 in range(len(kvarr_dict["w0"])):
                                    kv_dict["w0"] = kvarr_dict["w0"][i_w0]
                                    for i_w4 in range(len(kvarr_dict["w4"])):
                                        kv_dict["w4"] = kvarr_dict["w4"][i_w4]
                                        for i_w5 in range(len(kvarr_dict["w5"])):
                                            kv_dict["w5"] = kvarr_dict["w5"][i_w5]
                                            for i_w6 in range(len(kvarr_dict["w6"])):
                                                kv_dict["w6"] = kvarr_dict["w6"][i_w6]
                                                optimizer_entry = {OPT_FUNC_ID: model_id,
                                                                   OPT_FUNC_NAME: model_function,
                                                                   OPT_VARS: kv_dict,
                                                                   OPT_USE_CAP_NOM: use_cap_nom,
                                                                   OPT_USE_T_PRODUCTION: use_t_prod}
                                                logging.log.info(str_base % kv_dict)
                                                run_model(optimizer_entry, log_dfs,
                                                          fig_list, fig_and_sp_from_pid_arr)
        # p2 -> p0 -> p1 -> p3 -> p4 -> p8 -> w0 -> w4 -> w5 -> w6

    elif STEPPING_MODE == 2:  # stepping list-wise --> see optimize_models_manual_stepping_list.py
        model_name = "model_f056"  # in the dictionaries above
        use_cap_nom = 1.0  # 0.2  # 0.0  # -0.08  # see notes for USE_NOMINAL_CAPACITY_DEFAULT
        use_t_prod = True  # see notes for USE_T_PRODUCTION_DEFAULT

        # no need to change this:
        kv_str_arr = optimizer_list.kv_str_arr  # use list in optimize_models_manual_stepping_list.py
        model_id = model_id_dict.get(model_name)
        model_function = model_function_dict.get(model_name)
        for i in range(len(kv_str_arr)):
            kv_dict = get_kv_dict(kv_str_arr[i])
            optimizer_entry = {OPT_FUNC_ID: model_id, OPT_FUNC_NAME: model_function, OPT_VARS: kv_dict,
                               OPT_USE_CAP_NOM: use_cap_nom, OPT_USE_T_PRODUCTION: use_t_prod}
            logging.log.info("Starting process to model...")
            start_mdl_timestamp = datetime.now()
            run_model(optimizer_entry, log_dfs, fig_list, fig_and_sp_from_pid_arr)
            stop_mdl_timestamp = datetime.now()
            logging.log.info("Modeling runtime: %s h:mm:ss.ms" % str(stop_mdl_timestamp - start_mdl_timestamp))
            iteration = iteration + 1
        STEPPING_MODE = 0  # transition to interactive mode (user prompt) when done to avoid loading data again
        SHOW_IN_BROWSER = SHOW_IN_BROWSER_IF_MANUAL
    
    if STEPPING_MODE == 0:  # interactive mode (user prompt - fully manual)
        while True:
            try:
                print("Press Ctrl + C any time to reset, press enter to use the [default] value")
                
                # ToDo: this is the interactive mode (fully manual optimization) - you may adjust the default values  
                while True:
                    default_model_name = "model_f056"
                    model_function = model_function_dict.get(default_model_name)
                    model_id = model_id_dict.get(default_model_name)
                    default_values = opt_start_value_list.get(default_model_name)
                    kv_dict = get_kv_dict(default_values)

                    model_name = input("Which model do you want to use? ['%s']" % default_model_name)
                    if model_name == "":
                        model_name = default_model_name
                    if (model_name in model_function_dict) and (model_name in model_id_dict):
                        model_function = model_function_dict.get(model_name)
                        model_id = model_id_dict.get(model_name)
                    else:
                        print("Invalid model '%s', try again." % model_name)
                        continue

                    default_values = opt_start_value_list.get(model_name)
                    values = input("Which values do you want to use?\n['%s']\n" % default_values)
                    if values == "":
                        values = default_values

                    # kv_dict = {}
                    kv_dict = get_kv_dict(values)

                    print("I detected:")
                    for k, v in kv_dict.items():
                        print("%s: %.12f" % (k, v))

                    x = input("Is this correct? y = continue, n = redo input [y]:")
                    if (x == "") or (x == "y"):
                        break

                while True:
                    default_value = "y"
                    x = input("Do you want to use t_production as the start (y) or t_start_of_experiment (n)? [%s]: "
                              % default_value)
                    if x == "":
                        x = default_value
                    if x == "y":
                        use_t_prod = True
                    else:
                        use_t_prod = False
                    break

                while True:
                    use_cap_nom = 1.0  # 0.2  # 0.0  # -0.08  # 0  # 0.5
                    x = input("Do you want to use C_nom as the start value (1) or the firstly measured C_initial (0) "
                              "or a value in between (e.g., 0.5)? [%.2f]: " % use_cap_nom)
                    if x != "":
                        try:
                            use_cap_nom = float(x)
                        except FloatingPointError:
                            print("Invalid input, retry.")
                            continue
                    break

                optimizer_entry = {OPT_FUNC_ID: model_id, OPT_FUNC_NAME: model_function, OPT_VARS: kv_dict,
                                   OPT_USE_CAP_NOM: use_cap_nom, OPT_USE_T_PRODUCTION: use_t_prod}

                logging.log.info("Starting process to model...")
                start_mdl_timestamp = datetime.now()
                run_model(optimizer_entry, log_dfs, fig_list, fig_and_sp_from_pid_arr)
                stop_mdl_timestamp = datetime.now()
                logging.log.info("Modeling runtime: %s h:mm:ss.ms" % str(stop_mdl_timestamp - start_mdl_timestamp))
                iteration = iteration + 1
            except KeyboardInterrupt:
                x = input("Restart input? Enter 'y'. Everything else will exit the script")
                if (x == "y") or (x == "yes") or (x == "Y"):
                    continue
                else:
                    break


# run an aging model (called by optimizer_main, depending on the STEPPING_MODE)
def run_model(task_entry, log_dfs, fig_list, fig_and_sp_from_pid_arr):
    processor_number = 0
    queue_entry = task_entry
    if queue_entry is None:
        return

    mdl_id = queue_entry[OPT_FUNC_ID]
    mdl_name = queue_entry[OPT_FUNC_NAME]
    variables = queue_entry[OPT_VARS]
    # variable_bounds = queue_entry[OPT_VAR_LIMS]
    # opt_method = None
    # if OPT_METHOD in queue_entry:
    #     opt_method = queue_entry[OPT_METHOD]
    use_cap_nom = USE_NOMINAL_CAPACITY_DEFAULT
    if OPT_USE_CAP_NOM in queue_entry:
        use_cap_nom = queue_entry[OPT_USE_CAP_NOM]
    use_t_prod = USE_T_PRODUCTION_DEFAULT
    if OPT_USE_T_PRODUCTION in queue_entry:
        use_t_prod = queue_entry[OPT_USE_T_PRODUCTION]

    # logging.log.info("Thread %u model function ID %u - start optimizing" % (processor_number, mdl_id))

    num_infos = 0
    num_warnings = 0
    num_errors = 0
    rmse_total = [np.nan, np.nan]
    result_filename = "no result file!"  # fallback string if writing failed
    # noinspection PyBroadException
    try:
        # prepare figure -> make copy
        fig_list_copy = [None for _ in cfg.age_type]
        for i_fig in range(0, len(fig_list)):
            if fig_list[i_fig] is None:
                continue
            this_fig: go.Figure = fig_list[i_fig]
            # noinspection PyTypeChecker
            fig_list_copy[i_fig] = go.Figure(this_fig)

        # write results to plot and (optionally) show
        rmse_total = mdl_name(np.array(list(variables.values())), log_dfs, mdl_id, use_cap_nom, use_t_prod,
                              True, fig_list_copy, fig_and_sp_from_pid_arr)

    except ProcessingFailure:
        # logging.log.warning("Thread %u model function ID %u - optimizing failed!"
        #                     % (processor_number, function_id))
        pass
    except Exception:  # we don't want ths script to crash when there was an error in the aging model
        logging.log.error("Thread %u model function ID %u - Python Error:\n%s"
                          % (processor_number, mdl_id, traceback.format_exc()))

    # we land here on success or any error

    # reporting to main thread
    report_msg = (f"%s - model function ID %u - optimizing finished: %u infos, %u warnings, %u errors - RMSE: "
                  f"%.4f %% (%.4f %%)" % (result_filename, mdl_id, num_infos, num_warnings, num_errors,
                                          rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    report_level = config_logging.INFO
    if num_errors > 0:
        report_level = config_logging.ERROR
    elif num_warnings > 0:
        report_level = config_logging.WARNING
    logging.log.log(report_level, report_msg)


# ToDo: here comes a long list with all models tested...
#   model_f... is called by the optimizer
#   model_f..._get_delta_age is called for each parameter set id (operating condition) and nr (cells tested with same
#      parameter id). This is also, where the actual aging function is implemented.
#   prepare_model --> same for all: initialize variables
#   add_model_errorbands --> add the model error bands to the model result figure (if plotting and uncertainty modeling
#                            is enabled) -> currently only works for model_f055 and model_f056!
#   prepare_cell_log_df --> same for all: prepare cell log data frame
#   add_model_trace --> same for all: add the lines to the figure (if plotting is enabled)
#   add_model_param_rmse --> same for all: add RMSE to figure (if plotting is enabled)
#   plot_model --> same for all: plot the model result
#   ...
# === model_f001 BEGIN =================================================================================================
def model_f001(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f001"
    a0, a1, a2 = variable_list
    # logging.log.debug("%s (%u) call with a0 = %.24f, a1 = %.24f, a2 = %.24f" % (mdl_name, mdl_id, a0, a1, a2))
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    for pid_index in range(0, n_pids):
        rmse_param, num_rmse_points_param = 0, 0
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                    mdl_cap_delta_row, mdl_cap_row)
            if skip:
                continue

            # noinspection PyBroadException
            try:
                # === USER CODE 2 BEGIN ================================================================================
                log_df.loc[:, mdl_cap_delta_row] = model_f001_get_delta_age(log_df, a0, a1, a2)
                # === USER CODE 2 END ==================================================================================

                log_df.loc[:, mdl_cap_row] = C_0 - log_df[mdl_cap_delta_row].cumsum()  # calculate cap_aged with model
                cond_neg = (log_df[mdl_cap_row][log_df[mdl_cap_row] < 0.0])
                log_df.loc[cond_neg, mdl_cap_row] = 0.0
                rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param, rmse_cell = calc_rmse_cell(
                    mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
                if plot:
                    add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                    fig_list, fig_and_sp_from_pid_arr, rmse_cell)
            except Exception:
                logging.log.error("%s - Python Error:\n%s" % (mdl_name, traceback.format_exc()))

        rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = "a0 = %.24f, a1 = %.24f, a2 = %.24f" % (a0, a1, a2)
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f001_get_delta_age(log_df, a0, a1, a2):
    # calculate delta of aged_cap f_age(...)
    t_func = get_t_func_sqrt(log_df[DF_COL_TIME_USE])

    # log_df.loc[:, mdl_cap_delta_row] = ... # incremental Q_loss per timestep:
    return (0.5 * a0
            * a1 ** log_df[csv_label.V_CELL].astype(np.float64)
            * a2 ** log_df[csv_label.T_CELL].astype(np.float64)
            * t_func)
# === model_f001 END ===================================================================================================


# === model_f002 BEGIN =================================================================================================
def model_f002(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f002"
    a0, a1, a2 = variable_list
    # logging.log.debug("%s (%u) call with a0 = %.24f, a1 = %.24f, a2 = %.24f" % (mdl_name, mdl_id, a0, a1, a2))
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    for pid_index in range(0, n_pids):
        rmse_param, num_rmse_points_param = 0, 0
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                    mdl_cap_delta_row, mdl_cap_row)
            if skip:
                continue

            # noinspection PyBroadException
            try:
                # === USER CODE 2 BEGIN ================================================================================
                log_df.loc[:, mdl_cap_delta_row] = model_f002_get_delta_age(log_df, a0, a1, a2)
                # === USER CODE 2 END ==================================================================================

                log_df.loc[:, mdl_cap_row] = C_0 - log_df[mdl_cap_delta_row].cumsum()  # calculate cap_aged with model
                cond_neg = (log_df[mdl_cap_row][log_df[mdl_cap_row] < 0.0])
                log_df.loc[cond_neg, mdl_cap_row] = 0.0
                rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param, rmse_cell = calc_rmse_cell(
                    mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
                if plot:
                    add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                    fig_list, fig_and_sp_from_pid_arr, rmse_cell)
            except Exception:
                logging.log.error("%s - Python Error:\n%s" % (mdl_name, traceback.format_exc()))

        rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = "a0 = %.24f, a1 = %.24f, a2 = %.24f" % (a0, a1, a2)
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f002_get_delta_age(log_df, a0, a1, a2):
    # calculate delta of aged_cap f_age(...)
    t_func = get_t_func_sqrt(log_df[DF_COL_TIME_USE])

    V0 = 3.5
    T0 = cfg.T0 + 25.0
    DV = 0.1
    DT = 10.0

    # log_df.loc[:, mdl_cap_delta_row] = ... # incremental Q_loss per timestep:
    return (0.5 * a0
            * a1 ** ((log_df[csv_label.V_CELL].astype(np.float64) - V0) / DV)
            * a2 ** ((log_df[csv_label.T_CELL].astype(np.float64) - T0) / DT)
            * t_func)
# === model_f002 END ===================================================================================================


# === model_f003 BEGIN =================================================================================================
def model_f003(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f003"
    a0, a1, a2 = variable_list
    # logging.log.debug("%s (%u) call with a0 = %.24f, a1 = %.24f, a2 = %.24f" % (mdl_name, mdl_id, a0, a1, a2))
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    for pid_index in range(0, n_pids):
        rmse_param, num_rmse_points_param = 0, 0
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                    mdl_cap_delta_row, mdl_cap_row)
            if skip:
                continue

            # noinspection PyBroadException
            try:
                # === USER CODE 2 BEGIN ================================================================================
                log_df.loc[:, mdl_cap_delta_row] = model_f003_get_delta_age(log_df, a0, a1, a2)
                # === USER CODE 2 END ==================================================================================

                log_df.loc[:, mdl_cap_row] = C_0 - log_df[mdl_cap_delta_row].cumsum()  # calculate cap_aged with model
                cond_neg = (log_df[mdl_cap_row][log_df[mdl_cap_row] < 0.0])
                log_df.loc[cond_neg, mdl_cap_row] = 0.0
                rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param, rmse_cell = calc_rmse_cell(
                    mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
                if plot:
                    add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                    fig_list, fig_and_sp_from_pid_arr, rmse_cell)
            except Exception:
                logging.log.error("%s - Python Error:\n%s" % (mdl_name, traceback.format_exc()))

        rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = "a0 = %.24f, a1 = %.24f, a2 = %.24f" % (a0, a1, a2)
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f003_get_delta_age(log_df, a0, a1, a2):
    # calculate delta of aged_cap f_age(...)
    t_func = get_t_func_sqrt(log_df[DF_COL_TIME_USE])

    V0 = 3.6
    T0 = cfg.T0 + 25.0

    # log_df.loc[:, mdl_cap_delta_row] = ... # incremental Q_loss per timestep:
    return (0.5 * a0
            * np.exp(a1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
            * np.exp(a2 * (log_df[csv_label.T_CELL].astype(np.float64) - T0))
            * t_func)
# === model_f003 END ===================================================================================================


# === model_f004 BEGIN =================================================================================================
def model_f004(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f004"
    a0, a1, a2, a3, a4 = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    for pid_index in range(0, n_pids):
        rmse_param, num_rmse_points_param = 0, 0
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                    mdl_cap_delta_row, mdl_cap_row)
            if skip:
                continue

            # noinspection PyBroadException
            try:
                # === USER CODE 2 BEGIN ================================================================================
                log_df.loc[:, mdl_cap_delta_row] = model_f004_get_delta_age(log_df, a0, a1, a2, a3, a4)
                # === USER CODE 2 END ==================================================================================

                log_df.loc[:, mdl_cap_row] = C_0 - log_df[mdl_cap_delta_row].cumsum()  # calculate cap_aged with model
                cond_neg = (log_df[mdl_cap_row][log_df[mdl_cap_row] < 0.0])
                log_df.loc[cond_neg, mdl_cap_row] = 0.0
                rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param, rmse_cell = calc_rmse_cell(
                    mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
                if plot:
                    add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                    fig_list, fig_and_sp_from_pid_arr, rmse_cell)
            except Exception:
                logging.log.error("%s - Python Error:\n%s" % (mdl_name, traceback.format_exc()))

        rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = "a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f" % (a0, a1, a2, a3, a4)
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f004_get_delta_age(log_df, a0, a1, a2, a3, a4):
    # calculate delta of aged_cap f_age(...)
    t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

    V0 = 3.6
    T0 = cfg.T0 + 25.0
    label_I_chg = csv_label.I_CELL + "_chg"

    log_df[label_I_chg] = 0.0
    cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)  # if current >1/1000 C. 80 kWh bat: 80 W
    # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
    log_df.loc[cond_chg, label_I_chg] = log_df[csv_label.I_CELL]

    sei_potential = (  # log_df["model_f004_SEI_potential"] = (
            a0
            * np.exp(a1 * (1.0/log_df[csv_label.T_CELL].astype(np.float64) - 1.0/T0))
            * np.exp(a2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
            * np.exp(a3 * log_df[label_I_chg])
    )

    # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
    # it is a differential equation, I don't know a way how to solve it with pure pandas.
    # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
    # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
    # -> manually accumulate
    sei_potential_np = sei_potential.to_numpy()
    t_func_np = t_func.to_numpy()
    dq_loss_sei_np = np.zeros(len(sei_potential_np))
    q_loss_sei_total = 0
    # dq_loss_sei_np[0] = sei_potential_np[0] * t_func_np[0] -> t_func_np[0] = time diff at first index is 0 anyway
    try:
        for k in range(1, len(sei_potential_np)):
            diff = (sei_potential_np[k] - a4 * q_loss_sei_total)
            if diff > 0:  # SEI layer can only be increased, not reduced (cracking not modeled here)
                dq = diff * t_func_np[k]
                dq_loss_sei_np[k] = dq
                q_loss_sei_total = q_loss_sei_total + dq
            # dq_loss_sei_np[k] = sei_potential_np[k] * t_func_np[k] - a4 * dq_loss_sei_np[k - 1] * t_func_np[k]
    except RuntimeWarning:
        logging.log.warning("model_f004 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f "
                            "caused a RuntimeWarning!" % (a0, a1, a2, a3, a4))
        dq_loss_sei_np = sei_potential_np * t_func_np

    return dq_loss_sei_np
    # # log_df.loc[:, mdl_cap_delta_row] = ... # incremental Q_loss per timestep:
    # return (0.5 * a0
    #         * np.exp(a1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
    #         * np.exp(a2 * (log_df[csv_label.T_CELL].astype(np.float64) - T0))
    #         * t_func)
# === model_f004 END ===================================================================================================


# === model_f005 BEGIN =================================================================================================
def model_f005(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f005"
    a00, a4 = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    for pid_index in range(0, n_pids):
        rmse_param, num_rmse_points_param = 0, 0
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                    mdl_cap_delta_row, mdl_cap_row)
            if skip:
                continue

            # noinspection PyBroadException
            try:
                # === USER CODE 2 BEGIN ================================================================================
                log_df.loc[:, mdl_cap_delta_row] = model_f005_get_delta_age(log_df, a00, a4)
                # === USER CODE 2 END ==================================================================================

                log_df.loc[:, mdl_cap_row] = C_0 - log_df[mdl_cap_delta_row].cumsum()  # calculate cap_aged with model
                cond_neg = (log_df[mdl_cap_row][log_df[mdl_cap_row] < 0.0])
                log_df.loc[cond_neg, mdl_cap_row] = 0.0
                rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param, rmse_cell = calc_rmse_cell(
                    mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
                if plot:
                    add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                    fig_list, fig_and_sp_from_pid_arr, rmse_cell)
            except Exception:
                logging.log.error("%s - Python Error:\n%s" % (mdl_name, traceback.format_exc()))

        rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = "a00 = %.12f, a4 = %.12f" % (a00, a4)
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f005_get_delta_age(log_df, a00, a4):
    # calculate delta of aged_cap f_age(...)
    t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

    # sei_potential = a00
    t_func_np = t_func.to_numpy()
    dq_loss_sei_np = np.zeros(len(t_func_np))
    q_loss_sei_total = 0
    try:
        for k in range(1, len(dq_loss_sei_np)):
            diff = (a00 - a4 * q_loss_sei_total)
            if diff > 0:  # SEI layer can only be increased, not reduced (cracking not modeled here)
                dq = diff * t_func_np[k]
                dq_loss_sei_np[k] = dq
                q_loss_sei_total = q_loss_sei_total + dq
            # dq_loss_sei_np[k] = sei_potential_np[k] * t_func_np[k] - a4 * dq_loss_sei_np[k - 1] * t_func_np[k]
    except RuntimeWarning:
        logging.log.warning("model_f005 (???) call with a00 = %.12f, a4 = %.12f caused a RuntimeWarning!" % (a00, a4))
        dq_loss_sei_np = a00 * t_func_np

    return dq_loss_sei_np
    # # log_df.loc[:, mdl_cap_delta_row] = ... # incremental Q_loss per timestep:
# === model_f005 END ===================================================================================================


# === model_f006 BEGIN =================================================================================================
def model_f006(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f006"
    a0, a1, a2, a3, a4 = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    for pid_index in range(0, n_pids):
        rmse_param, num_rmse_points_param = 0, 0
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                    mdl_cap_delta_row, mdl_cap_row)
            if skip:
                continue

            # noinspection PyBroadException
            try:
                # === USER CODE 2 BEGIN ================================================================================
                log_df.loc[:, mdl_cap_delta_row] = model_f006_get_delta_age(log_df, a0, a1, a2, a3, a4)
                # === USER CODE 2 END ==================================================================================

                log_df.loc[:, mdl_cap_row] = C_0 - log_df[mdl_cap_delta_row].cumsum()  # calculate cap_aged with model
                cond_neg = (log_df[mdl_cap_row] < 0.0)
                log_df.loc[cond_neg, mdl_cap_row] = 0.0
                rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param, rmse_cell = calc_rmse_cell(
                    mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
                if plot:
                    add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                    fig_list, fig_and_sp_from_pid_arr, rmse_cell)
            except Exception:
                logging.log.error("%s - Python Error:\n%s" % (mdl_name, traceback.format_exc()))

        rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = "a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f" % (a0, a1, a2, a3, a4)
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f006_get_delta_age(log_df, a0, a1, a2, a3, a4):
    # calculate delta of aged_cap f_age(...)
    t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

    V0 = 3.73  # 3.6
    T0 = cfg.T0 + 25.0

    log_df[LABEL_C_CHG] = 0.0
    cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)  # if current >1/1000 C. 80 kWh bat: 80 W
    # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
    log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL] / cfg.CELL_CAPACITY_NOMINAL

    sei_potential = (  # log_df["model_f006_SEI_potential"] = (
            a0
            * np.exp(a1 * (1.0/log_df[csv_label.T_CELL].astype(np.float64) - 1.0/T0))
            * np.exp(a2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
            + a3 * log_df[LABEL_C_CHG] * t_func  # a3 * dQ_chg
            # * np.exp(a3 * log_df[label_I_chg])
    )

    # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
    # it is a differential equation, I don't know a way how to solve it with pure pandas.
    # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
    # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
    try:
        # # -> manually accumulate
        # sei_potential_np = sei_potential.to_numpy()
        # t_func_np = t_func.to_numpy()
        # dq_loss_sei_np = np.zeros(len(sei_potential_np))
        # q_loss_sei_total = 0
        # # dq_loss_sei_np[0] = sei_potential_np[0] * t_func_np[0] -> t_func_np[0]= time diff at first index is 0 anyway
        # for k in range(1, len(sei_potential_np)):
        #     diff = (sei_potential_np[k] - a4 * q_loss_sei_total)
        #     if diff > 0:  # SEI layer can only be increased, not reduced (cracking not modeled here)
        #         dq = diff * t_func_np[k]
        #         dq_loss_sei_np[k] = dq
        #         q_loss_sei_total = q_loss_sei_total + dq
        #     # dq_loss_sei_np[k] = sei_potential_np[k] * t_func_np[k] - a4 * dq_loss_sei_np[k - 1] * t_func_np[k]

        # faster accumulation using DataFrame chunks
        CHUNK_SIZE = int(CHUNK_DURATION / 30)
        q_loss_sei_total = 0
        df_length = sei_potential.shape[0]
        dq_loss_sei = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        i_start = 0
        while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
            i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end is not included when slicing df.iloc[i_start:i_end]
            # diff = (sei_potential.iloc[i_start:i_end]
            #         - a4 * q_loss_sei_total / (1.0 + q_loss_lam_cathode.iloc[i_start:i_end]))
            diff = (sei_potential.iloc[i_start:i_end] - a4 * q_loss_sei_total)
            diff[diff < 0] = 0.0  # SEI layer can only be increased, not reduced
            dq = diff * t_func.iloc[i_start:i_end]
            dq_loss_sei.iloc[i_start:i_end] = dq
            q_loss_sei_total = q_loss_sei_total + dq.sum()  # (dq.sum() / cfg.CELL_CAPACITY_NOMINAL)
            i_start = i_start + CHUNK_SIZE
    except RuntimeWarning:
        logging.log.warning("model_f006 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f "
                            "caused a RuntimeWarning!" % (a0, a1, a2, a3, a4))
        # dq_loss_sei_np = sei_potential_np * t_func_np
        dq_loss_sei = sei_potential * t_func

    # return dq_loss_sei_np
    return dq_loss_sei * cfg.CELL_CAPACITY_NOMINAL
    # # log_df.loc[:, mdl_cap_delta_row] = ... # incremental Q_loss per timestep:
    # return (0.5 * a0
    #         * np.exp(a1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
    #         * np.exp(a2 * (log_df[csv_label.T_CELL].astype(np.float64) - T0))
    #         * t_func)
# === model_f006 END ===================================================================================================


# === model_f007 BEGIN =================================================================================================
def model_f007(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f007"
    a0, a1, a2, a3, a4, a5 = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    for pid_index in range(0, n_pids):
        rmse_param, num_rmse_points_param = 0, 0
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                    mdl_cap_delta_row, mdl_cap_row)
            if skip:
                continue

            # noinspection PyBroadException
            try:
                # === USER CODE 2 BEGIN ================================================================================
                # log_df.loc[:, mdl_cap_delta_row] = model_f007_get_age(log_df, C_0, a0, a1, a2, a3, a4, a5)
                log_df.loc[:, mdl_cap_row] = model_f007_get_age(log_df, C_0, a0, a1, a2, a3, a4, a5)
                # === USER CODE 2 END ==================================================================================

                # log_df.loc[:, mdl_cap_row] = C_0 - log_df[mdl_cap_delta_row].cumsum()  # calculate cap_aged with model
                cond_neg = (log_df[mdl_cap_row] < 0.0)
                log_df.loc[cond_neg, mdl_cap_row] = 0.0
                rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param, rmse_cell = calc_rmse_cell(
                    mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
                if plot:
                    add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                    fig_list, fig_and_sp_from_pid_arr, rmse_cell)
            except Exception:
                logging.log.error("%s - Python Error:\n%s" % (mdl_name, traceback.format_exc()))

        rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = "a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, a5 = %.12f" % (a0, a1, a2, a3, a4, a5)
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f007_get_age(log_df, c0, a0, a1, a2, a3, a4, a5):
    # calculate aged_cap f_age(...)
    t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

    V0 = 3.73  # 3.6
    T0 = cfg.T0 + 25.0

    log_df[LABEL_C_CHG] = 0.0
    cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)  # if current >1/1000 C. 80 kWh bat: 80 W
    # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
    log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL] / cfg.CELL_CAPACITY_NOMINAL

    sei_potential = (  # log_df["model_f007_SEI_potential"] = (
            a0
            * np.exp(a1 * (1.0/log_df[csv_label.T_CELL].astype(np.float64) - 1.0/T0))
            * np.exp(a2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
            + a3 * log_df[LABEL_C_CHG] * t_func  # a3 * dQ_chg
            # * np.exp(a3 * log_df[label_I_chg])
    )

    dq_loss_lam_cathode = ((log_df[LABEL_C_CHG] * t_func)  # ΔQ per timestep
                           * (np.exp(a5 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)**2) - 1.0))
    q_loss_lam_cathode = dq_loss_lam_cathode.cumsum()

    # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
    # it is a differential equation, I don't know a way how to solve it with pure pandas.
    # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
    # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
    try:
        # # -> manually accumulate
        # sei_potential_np = sei_potential.to_numpy()
        # t_func_np = t_func.to_numpy()
        # dq_loss_sei_np = np.zeros(len(sei_potential_np))
        # q_loss_sei_total = 0
        # # dq_loss_sei_np[0] = sei_potential_np[0] * t_func_np[0] -> t_func_np[0]: time diff at first index is 0 anyway
        # for k in range(1, len(sei_potential_np)):
        #     diff = (sei_potential_np[k] - a4 * q_loss_sei_total / (1.0 + q_loss_lam_cathode[k]))
        #     if diff > 0:  # SEI layer can only be increased, not reduced (cracking not modeled here)
        #         dq = diff * t_func_np[k]
        #         dq_loss_sei_np[k] = dq
        #         q_loss_sei_total = q_loss_sei_total + (dq / cfg.CELL_CAPACITY_NOMINAL)
        #     # dq_loss_sei_np[k] = sei_potential_np[k] * t_func_np[k] - a4 * dq_loss_sei_np[k - 1] * t_func_np[k]

        # faster accumulation using DataFrame chunks
        CHUNK_SIZE = int(CHUNK_DURATION / 30)
        q_loss_sei_total = 0
        df_length = sei_potential.shape[0]
        dq_loss_sei = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        i_start = 0
        while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
            i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end is not included when slicing df.iloc[i_start:i_end]
            # diff = (sei_potential.iloc[i_start:i_end]
            #         - a4 * q_loss_sei_total / (1.0 + q_loss_lam_cathode.iloc[i_start:i_end]))
            diff = (sei_potential.iloc[i_start:i_end] - a4 * q_loss_sei_total)
            diff[diff < 0] = 0.0  # SEI layer can only be increased, not reduced
            dq = diff * t_func.iloc[i_start:i_end]
            dq_loss_sei.iloc[i_start:i_end] = dq
            q_loss_sei_total = q_loss_sei_total + dq.sum()  # (dq.sum() / cfg.CELL_CAPACITY_NOMINAL)
            i_start = i_start + CHUNK_SIZE

    except RuntimeWarning:
        logging.log.warning("model_f007 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f "
                            "caused a RuntimeWarning!" % (a0, a1, a2, a3, a4))
        # dq_loss_sei_np = sei_potential_np * t_func_np
        dq_loss_sei = sei_potential * t_func

    # return dq_loss_sei_np
    # return dq_loss_sei
    q_loss_sei = dq_loss_sei.cumsum()

    # determine maximum loss of q_loss_sei, q_loss_lam_cathode, ...
    # q_loss_effective = np.minimum(q_loss_sei, q_loss_lam_cathode)
    # q_loss_effective = pd.concat([q_loss_sei, q_loss_lam_cathode], axis=1).max(axis=1)
    q_loss_effective = q_loss_sei.clip(lower=q_loss_lam_cathode)  # faster?
    # .clip() was suggested in https://stackoverflow.com/a/57628811/2738240

    return c0 - q_loss_effective * cfg.CELL_CAPACITY_NOMINAL
    # # log_df.loc[:, mdl_cap_delta_row] = ... # incremental Q_loss per timestep:
    # return (0.5 * a0
    #         * np.exp(a1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
    #         * np.exp(a2 * (log_df[csv_label.T_CELL].astype(np.float64) - T0))
    #         * t_func)
# === model_f007 END ===================================================================================================


# === model_f008 BEGIN =================================================================================================
def model_f008(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f008"
    a0, a1, a2, a3, a4, a5 = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    for pid_index in range(0, n_pids):
        rmse_param, num_rmse_points_param = 0, 0
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                    mdl_cap_delta_row, mdl_cap_row)
            if skip:
                continue

            # noinspection PyBroadException
            try:
                # === USER CODE 2 BEGIN ================================================================================
                log_df.loc[:, mdl_cap_delta_row] = model_f008_get_delta_age(log_df, a0, a1, a2, a3, a4, a5)
                # === USER CODE 2 END ==================================================================================

                log_df.loc[:, mdl_cap_row] = C_0 - log_df[mdl_cap_delta_row].cumsum()  # calculate cap_aged with model
                cond_neg = (log_df[mdl_cap_row] < 0.0)
                log_df.loc[cond_neg, mdl_cap_row] = 0.0
                rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param, rmse_cell = calc_rmse_cell(
                    mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
                if plot:
                    add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                    fig_list, fig_and_sp_from_pid_arr, rmse_cell)
            except Exception:
                logging.log.error("%s - Python Error:\n%s" % (mdl_name, traceback.format_exc()))

        rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = "a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, a5 = %.12f" % (a0, a1, a2, a3, a4, a5)
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f008_get_delta_age(log_df, a0, a1, a2, a3, a4, a5):
    # calculate delta of aged_cap f_age(...)
    t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

    V0 = 3.73  # 3.6
    V_mid = 3.73  # 3.6
    T0 = cfg.T0 + 25.0

    log_df[LABEL_C_CHG] = 0.0
    cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)  # if current >1/1000 C. 80 kWh bat: 80 W
    # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
    log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL] / cfg.CELL_CAPACITY_NOMINAL

    sei_potential = (
            a0
            * np.exp(a1 * (1.0/log_df[csv_label.T_CELL].astype(np.float64) - 1.0/T0))
            * np.exp(a2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
            # + (a3 * log_df[LABEL_C_CHG] * t_func  # a3 * dQ_chg  # remove a4 if it works without
            #    * np.exp(a4 * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid)**2))
            + (a3 * log_df[LABEL_C_CHG] * t_func  # a3 * dQ_chg
               * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid)**2)
            # * np.exp(a3 * log_df[label_I_chg])
    )

    # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
    # it is a differential equation, I don't know a way how to solve it with pure pandas.
    # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
    # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
    try:
        # faster accumulation using DataFrame chunks
        CHUNK_SIZE = int(CHUNK_DURATION / 30)
        q_loss_sei_total = 0
        df_length = sei_potential.shape[0]
        dq_loss_sei = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        i_start = 0
        while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
            i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end is not included when slicing df.iloc[i_start:i_end]
            # diff = (sei_potential.iloc[i_start:i_end]
            #         - a5 * q_loss_sei_total / (1.0 + q_loss_lam_cathode.iloc[i_start:i_end]))
            diff = (sei_potential.iloc[i_start:i_end] - a5 * q_loss_sei_total)
            diff[diff < 0] = 0.0  # SEI layer can only be increased, not reduced
            dq = diff * t_func.iloc[i_start:i_end]
            dq_loss_sei.iloc[i_start:i_end] = dq
            q_loss_sei_total = q_loss_sei_total + dq.sum()  # (dq.sum() / cfg.CELL_CAPACITY_NOMINAL)
            i_start = i_start + CHUNK_SIZE
    except RuntimeWarning:
        logging.log.warning("model_f008 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, "
                            "a5 = %.12f caused a RuntimeWarning!" % (a0, a1, a2, a3, a4, a5))
        dq_loss_sei = sei_potential * t_func

    return dq_loss_sei * cfg.CELL_CAPACITY_NOMINAL
# === model_f008 END ===================================================================================================


# === model_f009 BEGIN =================================================================================================
def model_f009(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f009"
    a0, a1, a2, a3, a4, a5 = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    for pid_index in range(0, n_pids):
        rmse_param, num_rmse_points_param = 0, 0
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                    mdl_cap_delta_row, mdl_cap_row)
            if skip:
                continue

            # noinspection PyBroadException
            try:
                # === USER CODE 2 BEGIN ================================================================================
                log_df.loc[:, mdl_cap_delta_row] = model_f009_get_delta_age(log_df, a0, a1, a2, a3, a4, a5)
                # === USER CODE 2 END ==================================================================================

                log_df.loc[:, mdl_cap_row] = C_0 - log_df[mdl_cap_delta_row].cumsum()  # calculate cap_aged with model
                cond_neg = (log_df[mdl_cap_row] < 0.0)
                log_df.loc[cond_neg, mdl_cap_row] = 0.0
                rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param, rmse_cell = calc_rmse_cell(
                    mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
                if plot:
                    add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                    fig_list, fig_and_sp_from_pid_arr, rmse_cell)
            except Exception:
                logging.log.error("%s - Python Error:\n%s" % (mdl_name, traceback.format_exc()))

        rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = "a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, a5 = %.12f" % (a0, a1, a2, a3, a4, a5)
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f009_get_delta_age(log_df, a0, a1, a2, a3, a4, a5):
    # calculate delta of aged_cap f_age(...)
    t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

    V0 = 3.73  # 3.6
    V_mid = 3.73  # 3.6
    T0 = cfg.T0 + 25.0

    log_df[LABEL_C_CHG] = 0.0
    cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)  # if current >1/1000 C. 80 kWh bat: 80 W
    # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
    log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL] / cfg.CELL_CAPACITY_NOMINAL

    sei_potential = (
            a0
            * np.exp(a1 * (1.0/log_df[csv_label.T_CELL].astype(np.float64) - 1.0/T0))
            * np.exp(a2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
    )
    cyclic_age_potential = (a3 * log_df[LABEL_C_CHG] * t_func  # a3 * dQ_chg
                            * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid)**2)

    # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
    # it is a differential equation, I don't know a way how to solve it with pure pandas.
    # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
    # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
    try:
        # faster accumulation using DataFrame chunks
        CHUNK_SIZE = int(CHUNK_DURATION / 30)
        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        df_length = sei_potential.shape[0]
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        i_start = 0
        while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
            i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end is not included when slicing df.iloc[i_start:i_end]
            diff_a = (sei_potential.iloc[i_start:i_end] - a4 * q_loss_sei_total)
            diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
            diff_b = (cyclic_age_potential.iloc[i_start:i_end] - a5 * q_loss_cyclic_total)
            diff_b[diff_b < 0] = 0.0
            dq_a = diff_a * t_func.iloc[i_start:i_end]
            dq_b = diff_b * t_func.iloc[i_start:i_end]
            dq_loss.iloc[i_start:i_end] = dq_a + dq_b
            q_loss_sei_total = q_loss_sei_total + dq_a.sum()
            q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
            i_start = i_start + CHUNK_SIZE
    except RuntimeWarning:
        logging.log.warning("model_f009 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, "
                            "a5 = %.12f caused a RuntimeWarning!" % (a0, a1, a2, a3, a4, a5))
        dq_loss = (sei_potential + cyclic_age_potential) * t_func

    return dq_loss * cfg.CELL_CAPACITY_NOMINAL
# === model_f009 END ===================================================================================================


# === model_f010 BEGIN =================================================================================================
def model_f010(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f010"
    a0, a1, a2, a3, a4, a5, a6 = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    for pid_index in range(0, n_pids):
        rmse_param, num_rmse_points_param = 0, 0
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                    mdl_cap_delta_row, mdl_cap_row)
            if skip:
                continue

            # noinspection PyBroadException
            try:
                # === USER CODE 2 BEGIN ================================================================================
                log_df.loc[:, mdl_cap_delta_row] = model_f010_get_delta_age(log_df, a0, a1, a2, a3, a4, a5, a6)
                # === USER CODE 2 END ==================================================================================

                log_df.loc[:, mdl_cap_row] = C_0 - log_df[mdl_cap_delta_row].cumsum()  # calculate cap_aged with model
                cond_neg = (log_df[mdl_cap_row] < 0.0)
                log_df.loc[cond_neg, mdl_cap_row] = 0.0
                rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param, rmse_cell = calc_rmse_cell(
                    mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
                if plot:
                    add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                    fig_list, fig_and_sp_from_pid_arr, rmse_cell)
            except Exception:
                logging.log.error("%s - Python Error:\n%s" % (mdl_name, traceback.format_exc()))

        rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, a5 = %.12f, a6 = %.12f"
                  % (a0, a1, a2, a3, a4, a5, a6))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f010_get_delta_age(log_df, a0, a1, a2, a3, a4, a5, a6):
    # calculate delta of aged_cap f_age(...)
    t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

    V0 = 3.65  # 3.6  # 3.73  # 3.6
    V_mid = 3.65  # 3.6  # 3.73  # 3.6
    T0 = cfg.T0 + 25.0

    log_df[LABEL_C_CHG] = 0.0
    cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)  # if current >1/1000 C. 80 kWh bat: 80 W
    # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
    log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL] / cfg.CELL_CAPACITY_NOMINAL

    sei_potential = (
            a0
            * np.exp(a1 * (1.0/log_df[csv_label.T_CELL].astype(np.float64) - 1.0/T0))
            * np.exp(a2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
    )
    cyclic_age_potential = (a3 * log_df[LABEL_C_CHG] * t_func  # a3 * dQ_chg
                            * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid)**2)

    cyclic_age_wearout = a6 * log_df[LABEL_C_CHG] * t_func  # a6 * dQ_chg

    # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
    # it is a differential equation, I don't know a way how to solve it with pure pandas.
    # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
    # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
    try:
        # faster accumulation using DataFrame chunks
        CHUNK_SIZE = int(CHUNK_DURATION / 30)
        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        df_length = sei_potential.shape[0]
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        i_start = 0
        while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
            i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end is not included when slicing df.iloc[i_start:i_end]
            diff_a = (sei_potential.iloc[i_start:i_end] - a4 * q_loss_sei_total)
            diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
            diff_b = (cyclic_age_potential.iloc[i_start:i_end] - a5 * q_loss_cyclic_total)
            diff_b[diff_b < 0] = 0.0
            dq_a = diff_a * t_func.iloc[i_start:i_end]
            dq_b = diff_b * t_func.iloc[i_start:i_end]
            dq_loss.iloc[i_start:i_end] = dq_a + dq_b
            q_loss_sei_total = q_loss_sei_total + dq_a.sum()
            q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
            i_start = i_start + CHUNK_SIZE
    except RuntimeWarning:
        logging.log.warning("model_f010 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, "
                            "a5 = %.12f, a6 = %.12f caused a RuntimeWarning!" % (a0, a1, a2, a3, a4, a5, a6))
        dq_loss = (sei_potential + cyclic_age_potential) * t_func

    return (dq_loss + cyclic_age_wearout) * cfg.CELL_CAPACITY_NOMINAL
# === model_f010 END ===================================================================================================


# === model_f011 BEGIN =================================================================================================
def model_f011(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f011"
    a0, a1, a2, a3, a4, a5, a6 = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    for pid_index in range(0, n_pids):
        rmse_param, num_rmse_points_param = 0, 0
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                    mdl_cap_delta_row, mdl_cap_row)
            if skip:
                continue

            # noinspection PyBroadException
            try:
                # === USER CODE 2 BEGIN ================================================================================
                log_df.loc[:, mdl_cap_delta_row] = model_f011_get_delta_age(log_df, a0, a1, a2, a3, a4, a5, a6)
                # === USER CODE 2 END ==================================================================================

                log_df.loc[:, mdl_cap_row] = C_0 - log_df[mdl_cap_delta_row].cumsum()  # calculate cap_aged with model
                cond_neg = (log_df[mdl_cap_row] < 0.0)
                log_df.loc[cond_neg, mdl_cap_row] = 0.0
                rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param, rmse_cell = calc_rmse_cell(
                    mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
                if plot:
                    add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                    fig_list, fig_and_sp_from_pid_arr, rmse_cell)
            except Exception:
                logging.log.error("%s - Python Error:\n%s" % (mdl_name, traceback.format_exc()))

        rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, a5 = %.12f, a6 = %.12f"
                  % (a0, a1, a2, a3, a4, a5, a6))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f011_get_delta_age(log_df, a0, a1, a2, a3, a4, a5, a6):
    # calculate delta of aged_cap f_age(...)
    t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

    V0 = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
    V_mid = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
    T0 = cfg.T0 + 25.0

    log_df[LABEL_C_CHG] = 0.0
    cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)  # if current >1/1000 C. 80 kWh bat: 80 W
    # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
    log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL] / cfg.CELL_CAPACITY_NOMINAL

    sei_potential = (
            a0
            * np.exp(a1 * (1.0/log_df[csv_label.T_CELL].astype(np.float64) - 1.0/T0))
            * np.exp(a2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
    )
    cyclic_age_potential = (a3 * log_df[LABEL_C_CHG] * t_func  # a3 * dQ_chg
                            * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid)**2)

    cyclic_age_wearout = a6 * log_df[LABEL_C_CHG] * t_func  # a6 * dQ_chg

    # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
    # it is a differential equation, I don't know a way how to solve it with pure pandas.
    # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
    # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
    try:
        # faster accumulation using DataFrame chunks
        CHUNK_SIZE = int(CHUNK_DURATION / 30)
        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        df_length = sei_potential.shape[0]
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        i_start = 0
        while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
            i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end is not included when slicing df.iloc[i_start:i_end]
            # diff_a = (sei_potential.iloc[i_start:i_end] - a4 * q_loss_sei_total)
            diff_a = (sei_potential.iloc[i_start:i_end]
                      - a4 * q_loss_sei_total / (1.0 + cyclic_age_wearout.iloc[i_start:i_end]))
            diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
            diff_b = (cyclic_age_potential.iloc[i_start:i_end] - a5 * q_loss_cyclic_total)
            diff_b[diff_b < 0] = 0.0
            dq_a = diff_a * t_func.iloc[i_start:i_end]
            dq_b = diff_b * t_func.iloc[i_start:i_end]
            dq_loss.iloc[i_start:i_end] = dq_a + dq_b
            q_loss_sei_total = q_loss_sei_total + dq_a.sum()
            q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
            i_start = i_start + CHUNK_SIZE
    except RuntimeWarning:
        logging.log.warning("model_f011 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, "
                            "a5 = %.12f, a6 = %.12f caused a RuntimeWarning!" % (a0, a1, a2, a3, a4, a5, a6))
        dq_loss = (sei_potential + cyclic_age_potential) * t_func

    return (dq_loss + cyclic_age_wearout) * cfg.CELL_CAPACITY_NOMINAL
# === model_f011 END ===================================================================================================


# === model_f012 BEGIN =================================================================================================
def model_f012(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f012"
    a0, a1, a2, a3, a4, a5, a6, a7 = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    for pid_index in range(0, n_pids):
        rmse_param, num_rmse_points_param = 0, 0
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                    mdl_cap_delta_row, mdl_cap_row)
            if skip:
                continue

            # noinspection PyBroadException
            try:
                # === USER CODE 2 BEGIN ================================================================================
                log_df.loc[:, mdl_cap_delta_row] = model_f012_get_delta_age(log_df, a0, a1, a2, a3, a4, a5, a6, a7)
                # === USER CODE 2 END ==================================================================================

                log_df.loc[:, mdl_cap_row] = C_0 - log_df[mdl_cap_delta_row].cumsum()  # calculate cap_aged with model
                cond_neg = (log_df[mdl_cap_row] < 0.0)
                log_df.loc[cond_neg, mdl_cap_row] = 0.0
                rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param, rmse_cell = calc_rmse_cell(
                    mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
                if plot:
                    add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                    fig_list, fig_and_sp_from_pid_arr, rmse_cell)
            except Exception:
                logging.log.error("%s - Python Error:\n%s" % (mdl_name, traceback.format_exc()))

        rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, a5 = %.12f, a6 = %.12f, a7 = %.12f"
                  % (a0, a1, a2, a3, a4, a5, a6, a7))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f012_get_delta_age(log_df, a0, a1, a2, a3, a4, a5, a6, a7):
    # calculate delta of aged_cap f_age(...)
    t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

    V0 = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
    V_mid = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
    T0 = cfg.T0 + 25.0

    log_df[LABEL_C_CHG] = 0.0
    cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)  # if current >1/1000 C. 80 kWh bat: 80 W
    # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
    log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL] / cfg.CELL_CAPACITY_NOMINAL

    sei_potential = (
            a0
            * np.exp(a1 * (1.0/log_df[csv_label.T_CELL].astype(np.float64) - 1.0/T0))
            * np.exp(a2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
    )
    cyclic_age_potential = (a5 * log_df[LABEL_C_CHG] * t_func  # a5 * dQ_chg
                            * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid)**2)

    dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
    q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

    # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
    # it is a differential equation, I don't know a way how to solve it with pure pandas.
    # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
    # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
    try:
        # faster accumulation using DataFrame chunks
        CHUNK_SIZE = int(CHUNK_DURATION / 30)
        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        df_length = sei_potential.shape[0]
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        i_start = 0
        while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
            i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end is not included when slicing df.iloc[i_start:i_end]
            # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
            diff_a = (sei_potential.iloc[i_start:i_end]
                      - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_wearout.iloc[i_start:i_end]))
            diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
            diff_b = (cyclic_age_potential.iloc[i_start:i_end] - a6 * q_loss_cyclic_total)
            diff_b[diff_b < 0] = 0.0
            dq_a = diff_a * t_func.iloc[i_start:i_end]
            dq_b = diff_b * t_func.iloc[i_start:i_end]
            dq_loss.iloc[i_start:i_end] = dq_a + dq_b
            q_loss_sei_total = q_loss_sei_total + dq_a.sum()
            q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
            i_start = i_start + CHUNK_SIZE
    except RuntimeWarning:
        logging.log.warning("model_f012 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, "
                            "a5 = %.12f, a6 = %.12f, a7 = %.12f caused a RuntimeWarning!"
                            % (a0, a1, a2, a3, a4, a5, a6, a7))
        dq_loss = (sei_potential + cyclic_age_potential) * t_func

    return (dq_loss + dq_cyclic_age_wearout) * cfg.CELL_CAPACITY_NOMINAL
# === model_f012 END ===================================================================================================


# === model_f013 BEGIN =================================================================================================
def model_f013(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f013"
    a0, a1, a2, a3, a4, a5, a6, a7, a8 = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    for pid_index in range(0, n_pids):
        rmse_param, num_rmse_points_param = 0, 0
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                    mdl_cap_delta_row, mdl_cap_row)
            if skip:
                continue

            # noinspection PyBroadException
            try:
                # === USER CODE 2 BEGIN ================================================================================
                log_df.loc[:, mdl_cap_delta_row] = model_f013_get_delta_age(log_df, a0, a1, a2, a3, a4, a5, a6, a7, a8)
                # === USER CODE 2 END ==================================================================================

                log_df.loc[:, mdl_cap_row] = C_0 - log_df[mdl_cap_delta_row].cumsum()  # calculate cap_aged with model
                cond_neg = (log_df[mdl_cap_row] < 0.0)
                log_df.loc[cond_neg, mdl_cap_row] = 0.0
                rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param, rmse_cell = calc_rmse_cell(
                    mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
                if plot:
                    add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                    fig_list, fig_and_sp_from_pid_arr, rmse_cell)
            except Exception:
                logging.log.error("%s - Python Error:\n%s" % (mdl_name, traceback.format_exc()))

        rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, a5 = %.12f, a6 = %.12f, a7 = %.12f, "
                  "a8 = %.12f" % (a0, a1, a2, a3, a4, a5, a6, a7, a8))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f013_get_delta_age(log_df, a0, a1, a2, a3, a4, a5, a6, a7, a8):
    # calculate delta of aged_cap f_age(...)
    t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

    V0 = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
    V_mid = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
    T0 = cfg.T0 + 25.0

    log_df[LABEL_C_CHG] = 0.0
    cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)  # if current >1/1000 C. 80 kWh bat: 80 W
    # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
    log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL

    sei_potential = (
            a0
            * np.exp(a1 * (1.0/log_df[csv_label.T_CELL].astype(np.float64) - 1.0/T0))
            * np.exp(a2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
    )
    cyclic_age_potential = (a4 * log_df[LABEL_C_CHG].astype(np.float64) * t_func  # a4 * dQ_chg
                            * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid)**2)

    # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
    #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current collector
    #  corrosion? structural disordering? loss of electrical contact?)

    # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
    # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

    # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
    # it is a differential equation, I don't know a way how to solve it with pure pandas.
    # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
    # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
    # faster accumulation using DataFrame chunks
    if t_func.shape[0] > 1:
        dt = t_func.max()
        if (dt == 2) or (dt == 10) or (dt == 30):
            CHUNK_SIZE = int(CHUNK_DURATION / dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f013 unexpected dt = %u" % dt)
    else:
        CHUNK_SIZE = int(CHUNK_DURATION / 30)
        logging.log.warning("model_f013 cannot read dt, t_func too short")

    # # store individual aging cumsum() to plot them later
    # log_df.loc[:, LABEL_Q_LOSS_SEI] = 0.0
    # log_df.loc[:, LABEL_Q_LOSS_wearout] = 0.0
    # log_df.loc[:, LABEL_Q_LOSS_PLATING] = 0.0

    try:
        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        df_length = sei_potential.shape[0]
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        i_start = 0
        while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
            i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end is not included when slicing df.iloc[i_start:i_end]
            # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
            # diff_a = (sei_potential.iloc[i_start:i_end]
            #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_wearout.iloc[i_start:i_end]))
            diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total / 1.0)
            diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
            diff_b = (cyclic_age_potential.iloc[i_start:i_end] - a5 * q_loss_cyclic_total)
            diff_b[diff_b < 0] = 0.0
            dq_a = diff_a * t_func.iloc[i_start:i_end]
            dq_b = diff_b * t_func.iloc[i_start:i_end]
            dq_loss.iloc[i_start:i_end] = dq_a + dq_b
            q_loss_sei_total = q_loss_sei_total + dq_a.sum()
            q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
            # log_df.loc[:, LABEL_Q_LOSS_SEI] = dq_a.cumsum()
            # log_df.loc[:, LABEL_Q_LOSS_wearout] = dq_b.cumsum()
            i_start = i_start + CHUNK_SIZE
    except RuntimeWarning:
        logging.log.warning("model_f013 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, "
                            "a5 = %.12f, a6 = %.12f, a7 = %.12f, a8 = %.12f caused a RuntimeWarning!"
                            % (a0, a1, a2, a3, a4, a5, a6, a7, a8))
        dq_loss = (sei_potential + cyclic_age_potential) * t_func

    # lithium plating is described in yangModelingLithiumPlating2017: Yang X.-G., Leng Y., Zhang G., et al.:
    #   "Modeling of lithium plating induced aging of lithium-ion batteries: Transition from linear to nonlinear aging"
    # and vonludersModelingLithiumPlating2019: von Lüders C., Keil J., Webersberger M., et al.:
    #   "Modeling of lithium plating and lithium stripping in lithium-ion batteries"
    # also see schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
    #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
    #   -> example for anode potential over SoC
    # severe Li plating -> clogging, steep resistance rise => probably high plating rates even at medium T + C-rates

    # a6: r_film (=R_film * C_nom) at T0
    # a7: temperature coefficient for r_film
    # a8: plating rate (ΔV_anode(<0) -> relative Δq_loss per second)

    try:
        anode_potential = get_v_anode_from_v_terminal_df(log_df[csv_label.V_CELL])
        q_loss_others = dq_loss.cumsum()
        q_loss_plating_total = 0
        q_loss_total = 0
        df_length = sei_potential.shape[0]
        dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        i_start = 0
        while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
            i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end is not included when slicing df.iloc[i_start:i_end]
            v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                         - log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64)
                           * a6 * (np.exp(-a7 * (log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64) - T0))
                                   / (1.0 - q_loss_total)))
            cond_pos = (v_plating > 0.0)
            v_plating[cond_pos] = 0.0
            dq_c = -v_plating * a8 * t_func.iloc[i_start:i_end]
            dq_loss_plating.iloc[i_start:i_end] = dq_c
            q_loss_plating_total = q_loss_plating_total + dq_c.sum()
            q_loss_total = q_loss_others.iloc[i_end - 1] + q_loss_plating_total
            # log_df.loc[:, LABEL_Q_LOSS_PLATING] = dq_c.cumsum()
            i_start = i_start + CHUNK_SIZE
        dq_loss = dq_loss + dq_loss_plating
    except RuntimeWarning:
        logging.log.warning("model_f013 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, "
                            "a5 = %.12f, a6 = %.12f, a7 = %.12f, a8 = %.12f caused a RuntimeWarning (plating calc.)!"
                            % (a0, a1, a2, a3, a4, a5, a6, a7, a8))

    # return (dq_loss + dq_cyclic_age_wearout) * cfg.CELL_CAPACITY_NOMINAL
    return dq_loss * cfg.CELL_CAPACITY_NOMINAL
# === model_f013 END ===================================================================================================


# === model_f013p BEGIN ================================================================================================
def model_f013p(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f013p"
    a0, a1, a2, a3, a4, a5, a6, a7, a8 = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    for pid_index in range(0, n_pids):
        rmse_param, num_rmse_points_param = 0, 0
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                    mdl_cap_delta_row, mdl_cap_row)
            if skip:
                continue

            # noinspection PyBroadException
            try:
                # === USER CODE 2 BEGIN ================================================================================
                log_df.loc[:, mdl_cap_delta_row] = model_f013p_get_delta_age(log_df, a0, a1, a2, a3, a4, a5, a6, a7, a8)
                # === USER CODE 2 END ==================================================================================

                log_df.loc[:, mdl_cap_row] = C_0 - log_df[mdl_cap_delta_row].cumsum()  # calculate cap_aged with model
                cond_neg = (log_df[mdl_cap_row] < 0.0)
                log_df.loc[cond_neg, mdl_cap_row] = 0.0
                rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param, rmse_cell = calc_rmse_cell(
                    mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
                if plot:
                    add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                    fig_list, fig_and_sp_from_pid_arr, rmse_cell)
            except Exception:
                logging.log.error("%s - Python Error:\n%s" % (mdl_name, traceback.format_exc()))

        rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, rmse_param, num_rmse_points_param)
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, a5 = %.12f, a6 = %.12f, a7 = %.12f, "
                  "a8 = %.12f" % (a0, a1, a2, a3, a4, a5, a6, a7, a8))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f013p_get_delta_age(log_df, a0, a1, a2, a3, a4, a5, a6, a7, a8):
    # calculate delta of aged_cap f_age(...)
    t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

    V0 = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
    V_mid = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
    T0 = cfg.T0 + 25.0

    log_df[LABEL_C_CHG] = 0.0
    cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)  # if current >1/1000 C. 80 kWh bat: 80 W
    # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
    log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL

    sei_potential = (
            a0
            * np.exp(a1 * (1.0/log_df[csv_label.T_CELL].astype(np.float64) - 1.0/T0))
            * np.exp(a2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
    )
    cyclic_age_potential = (a4 * log_df[LABEL_C_CHG].astype(np.float64) * t_func  # a4 * dQ_chg
                            * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid)**2)

    # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
    #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current collector
    #  corrosion? structural disordering? loss of electrical contact?)

    # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
    # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

    # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
    # it is a differential equation, I don't know a way how to solve it with pure pandas.
    # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
    # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
    # faster accumulation using DataFrame chunks
    if t_func.shape[0] > 1:
        dt = t_func.max()
        if (dt == 2) or (dt == 10) or (dt == 30):
            CHUNK_SIZE = int(CHUNK_DURATION / dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f013p unexpected dt = %u" % dt)
    else:
        CHUNK_SIZE = int(CHUNK_DURATION / 30)
        logging.log.warning("model_f013p cannot read dt, t_func too short")
    try:
        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        df_length = sei_potential.shape[0]
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        i_start = 0
        while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
            i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end is not included when slicing df.iloc[i_start:i_end]
            # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
            # diff_a = (sei_potential.iloc[i_start:i_end]
            #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_wearout.iloc[i_start:i_end]))
            diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total / 1.0)
            diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
            diff_b = (cyclic_age_potential.iloc[i_start:i_end] - a5 * q_loss_cyclic_total)
            diff_b[diff_b < 0] = 0.0
            dq_a = diff_a * t_func.iloc[i_start:i_end]
            dq_b = diff_b * t_func.iloc[i_start:i_end]
            dq_loss.iloc[i_start:i_end] = dq_a + dq_b
            q_loss_sei_total = q_loss_sei_total + dq_a.sum()
            q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
            i_start = i_start + CHUNK_SIZE
    except RuntimeWarning:
        logging.log.warning("model_f013p (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, "
                            "a5 = %.12f, a6 = %.12f, a7 = %.12f, a8 = %.12f caused a RuntimeWarning!"
                            % (a0, a1, a2, a3, a4, a5, a6, a7, a8))
        dq_loss = (sei_potential + cyclic_age_potential) * t_func

    # lithium plating is described in yangModelingLithiumPlating2017: Yang X.-G., Leng Y., Zhang G., et al.:
    #   "Modeling of lithium plating induced aging of lithium-ion batteries: Transition from linear to nonlinear aging"
    # and vonludersModelingLithiumPlating2019: von Lüders C., Keil J., Webersberger M., et al.:
    #   "Modeling of lithium plating and lithium stripping in lithium-ion batteries"
    # also see schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
    #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
    #   -> example for anode potential over SoC
    # severe Li plating -> clogging, steep resistance rise => probably high plating rates even at medium T + C-rates

    # a6: r_film (=R_film * C_nom) at T0
    # a7: temperature coefficient for r_film
    # a8: plating rate (ΔV_anode(<0) -> relative Δq_loss per second)

    try:
        anode_potential = get_v_anode_from_v_terminal_df(log_df[csv_label.V_CELL])
        q_loss_others = dq_loss.cumsum()
        q_loss_plating_total = 0
        q_loss_total = 0
        df_length = sei_potential.shape[0]
        dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        i_start = 0
        while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
            i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end is not included when slicing df.iloc[i_start:i_end]
            v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                         - log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64)
                           * a6 * (np.exp(-a7 * (log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64) - T0))
                                   / (1.0 - q_loss_total)))
            cond_pos = (v_plating > 0.0)
            v_plating[cond_pos] = 0.0
            dq_c = -v_plating * a8 * t_func.iloc[i_start:i_end]
            dq_loss_plating.iloc[i_start:i_end] = dq_c
            q_loss_plating_total = q_loss_plating_total + dq_c.sum()
            q_loss_total = q_loss_others.iloc[i_end - 1] + q_loss_plating_total
            i_start = i_start + CHUNK_SIZE
        # dq_loss = dq_loss + dq_loss_plating
        dq_loss = dq_loss_plating  # ONLY RETURN PLATING IN THIS (DEBUGGING) MODEL!
    except RuntimeWarning:
        logging.log.warning("model_f013p (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, "
                            "a5 = %.12f, a6 = %.12f, a7 = %.12f, a8 = %.12f caused a RuntimeWarning (plating calc.)!"
                            % (a0, a1, a2, a3, a4, a5, a6, a7, a8))

    # return (dq_loss + dq_cyclic_age_wearout) * cfg.CELL_CAPACITY_NOMINAL
    return dq_loss * cfg.CELL_CAPACITY_NOMINAL
# === model_f013p END ==================================================================================================


# === model_f014 BEGIN =================================================================================================
def model_f014(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f014"
    a0, a1, a2, a3, a4, a5, a6, a7, a8 = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue

            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index}
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f014_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports

        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        if this_result is None:
            break  # no more reports

        log_df = this_result["log_df"]
        pid_index = this_result["pid_index"]
        pnr_index = this_result["pnr_index"]
        result = this_result["result"]

        log_df.loc[:, mdl_cap_row] = C_0[pid_index][pnr_index] - result.cumsum()
        cond_neg = (log_df[mdl_cap_row] < 0.0)
        log_df.loc[cond_neg, mdl_cap_row] = 0.0
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p, rmse_cell = calc_rmse_cell(
            mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)

    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f, a5 = %.12f, a6 = %.12f, a7 = %.12f, "
                  "a8 = %.12f" % (a0, a1, a2, a3, a4, a5, a6, a7, a8))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f014_get_delta_age(log_df, a0, a1, a2, a3, a4, a5, a6, a7, a8):
def model_f014_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        a0, a1, a2, a3, a4, a5, a6, a7, a8 = queue_entry["vars"]

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_mid = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        log_df[LABEL_C_CHG] = 0.0
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL

        sei_potential = (
                a0
                * np.exp(a1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(a2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (a4 * log_df[LABEL_C_CHG].astype(np.float64) * t_func  # a4 * dQ_chg
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid) ** 2)

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f014 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f014 cannot read dt, t_func too short")

        try:
            q_loss_sei_total = 0
            q_loss_cyclic_total = 0
            df_length = sei_potential.shape[0]
            dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_wearout.iloc[i_start:i_end]))
                diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total / 1.0)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - a5 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0
                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end]
                dq_loss.iloc[i_start:i_end] = dq_a + dq_b
                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f014 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, "
                                "a4 = %.12f, a5 = %.12f, a6 = %.12f, a7 = %.12f, a8 = %.12f caused a RuntimeWarning!"
                                % (a0, a1, a2, a3, a4, a5, a6, a7, a8))
            dq_loss = (sei_potential + cyclic_age_potential) * t_func

        # lithium plating is described in yangModelingLithiumPlating2017: Yang X.-G., Leng Y., Zhang G., et al.:
        #   "Modeling of lithium plating induced aging of lithium-ion batteries: Transition from linear to nonlinear
        #   aging"
        # and vonludersModelingLithiumPlating2019: von Lüders C., Keil J., Webersberger M., et al.:
        #   "Modeling of lithium plating and lithium stripping in lithium-ion batteries"
        # also see schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
        #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
        #   -> example for anode potential over SoC
        # severe Li plating -> clogging, steep resistance rise => probably high plating rates even at medium T + C-rates

        # a6: r_film (=R_film * C_nom) at T0
        # a7: temperature coefficient for r_film
        # a8: plating rate (ΔV_anode(<0) -> relative Δq_loss per second)

        try:
            anode_potential = get_v_anode_from_v_terminal_df(log_df[csv_label.V_CELL])
            q_loss_others = dq_loss.cumsum()
            q_loss_plating_total = 0
            q_loss_total = 0
            df_length = sei_potential.shape[0]
            dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64) * a6
                               * (np.exp(-a7 * (log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64) - T0))
                                  / (1.0 - q_loss_total)))
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0
                dq_c = -v_plating * a8 * t_func.iloc[i_start:i_end]
                dq_loss_plating.iloc[i_start:i_end] = dq_c
                q_loss_plating_total = q_loss_plating_total + dq_c.sum()
                q_loss_total = q_loss_others.iloc[i_end - 1] + q_loss_plating_total
                i_start = i_start + CHUNK_SIZE
            dq_loss = dq_loss + dq_loss_plating
        except RuntimeWarning:
            logging.log.warning("model_f014 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f,"
                                "a4 = %.12f, a5 = %.12f, a6 = %.12f, a7 = %.12f, a8 = %.12f caused a RuntimeWarning"
                                "(plating calc.)!" % (a0, a1, a2, a3, a4, a5, a6, a7, a8))

        # return (dq_loss + dq_cyclic_age_wearout) * cfg.CELL_CAPACITY_NOMINAL
        result_entry = queue_entry
        result_entry["result"] = dq_loss * cfg.CELL_CAPACITY_NOMINAL
        result_queue.put(result_entry)

    task_queue.close()
    logging.log.debug("exiting thread")
# === model_f014 END ===================================================================================================


# === model_f015 BEGIN =================================================================================================
def model_f015(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f015"
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue

            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index}
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f015_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports

        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        if this_result is None:
            break  # no more reports

        log_df = this_result["log_df"]
        pid_index = this_result["pid_index"]
        pnr_index = this_result["pnr_index"]
        result = this_result["result"]

        log_df.loc[:, mdl_cap_row] = C_0[pid_index][pnr_index] - result.cumsum()
        cond_neg = (log_df[mdl_cap_row] < 0.0)
        log_df.loc[cond_neg, mdl_cap_row] = 0.0
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p, rmse_cell = calc_rmse_cell(
            mdl_name, log_df, mdl_cap_row, rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)

    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("a0 = %.12f, a1 = %.3f, a2 = %.6f, a3 = %.12f, a4 = %.12f, a5 = %.12f, a6 = %.9f, a7 = %.9f, "
                  "a8 = %.12f, a9 = %.3f" % (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f015_get_delta_age(log_df, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
def model_f015_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = queue_entry["vars"]

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_mid = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        log_df[LABEL_C_CHG] = 0.0
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL

        sei_potential = (
                a0
                * np.exp(a1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(a2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (a4 * log_df[LABEL_C_CHG].astype(np.float64) * t_func  # a4 * dQ_chg
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid) ** 2)

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f015 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f015 cannot read dt, t_func too short")

        try:
            q_loss_sei_total = 0
            q_loss_cyclic_total = 0
            df_length = sei_potential.shape[0]
            dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_wearout.iloc[i_start:i_end]))
                diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total / 1.0)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - a5 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0
                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end]
                dq_loss.iloc[i_start:i_end] = dq_a + dq_b
                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f015 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f,"
                                " a5 = %.12f, a6 = %.12f, a7 = %.12f, a8 = %.12f , a9 = %.12f caused a RuntimeWarning!"
                                % (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9))
            dq_loss = (sei_potential + cyclic_age_potential) * t_func

        # lithium plating is described in yangModelingLithiumPlating2017: Yang X.-G., Leng Y., Zhang G., et al.:
        #   "Modeling of lithium plating induced aging of lithium-ion batteries: Transition from linear to nonlinear
        #   aging"
        # and vonludersModelingLithiumPlating2019: von Lüders C., Keil J., Webersberger M., et al.:
        #   "Modeling of lithium plating and lithium stripping in lithium-ion batteries"
        # also see schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
        #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
        #   -> example for anode potential over SoC
        # severe Li plating -> clogging, steep resistance rise => probably high plating rates even at medium T + C-rates

        # a6: r_film (=R_film * C_nom) at T0
        # a7: temperature coefficient for r_film
        # a8: plating rate (ΔV_anode(<0) -> relative Δq_loss per second)

        try:
            anode_potential = get_v_anode_from_v_terminal_df(log_df[csv_label.V_CELL])
            q_loss_others = dq_loss.cumsum()
            q_loss_plating_total = 0
            q_loss_total = 0
            df_length = sei_potential.shape[0]
            dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64) * a6
                               * (np.exp(-a7 * (log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64) - T0))
                                  / (1.0 - a9 * q_loss_total)))
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0
                dq_c = -v_plating * a8 * t_func.iloc[i_start:i_end]
                dq_loss_plating.iloc[i_start:i_end] = dq_c
                q_loss_plating_total = q_loss_plating_total + dq_c.sum()
                q_loss_total = q_loss_others.iloc[i_end - 1] + q_loss_plating_total
                i_start = i_start + CHUNK_SIZE
            dq_loss = dq_loss + dq_loss_plating
        except RuntimeWarning:
            logging.log.warning("model_f015 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f,"
                                " a5 = %.12f, a6 = %.12f, a7 = %.12f, a8 = %.12f, a9 = %.12f caused a RuntimeWarning"
                                "(plating calc.)!" % (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9))

        # return (dq_loss + dq_cyclic_age_wearout) * cfg.CELL_CAPACITY_NOMINAL
        result_entry = queue_entry
        result_entry["result"] = dq_loss * cfg.CELL_CAPACITY_NOMINAL
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f015 END ===================================================================================================


# === model_f016 BEGIN =================================================================================================
def model_f016(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f016"
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f016_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("a0 = %.12f, a1 = %.3f, a2 = %.6f, a3 = %.12f, a4 = %.12f, a5 = %.12f, a6 = %.9f, a7 = %.9f, "
                  "a8 = %.12f, a9 = %.3f" % (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f016_get_delta_age(log_df, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
def model_f016_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = queue_entry["vars"]

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_mid = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        log_df[LABEL_C_CHG] = 0.0
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL

        sei_potential = (
                a0
                * np.exp(a1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(a2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (a4 * log_df[LABEL_C_CHG].astype(np.float64) * t_func  # a4 * dQ_chg
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid) ** 2)

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f016 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f016 cannot read dt, t_func too short")

        try:
            q_loss_sei_total = 0
            q_loss_cyclic_total = 0
            df_length = sei_potential.shape[0]
            dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_wearout.iloc[i_start:i_end]))
                diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - a5 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0
                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end]
                dq_loss.iloc[i_start:i_end] = dq_a + dq_b
                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f016 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f,"
                                " a5 = %.12f, a6 = %.12f, a7 = %.12f, a8 = %.12f , a9 = %.12f caused a RuntimeWarning!"
                                % (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9))
            dq_loss = (sei_potential + cyclic_age_potential) * t_func

        # lithium plating is described in yangModelingLithiumPlating2017: Yang X.-G., Leng Y., Zhang G., et al.:
        #   "Modeling of lithium plating induced aging of lithium-ion batteries: Transition from linear to nonlinear
        #   aging"
        # and vonludersModelingLithiumPlating2019: von Lüders C., Keil J., Webersberger M., et al.:
        #   "Modeling of lithium plating and lithium stripping in lithium-ion batteries"
        # also see schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
        #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
        #   -> example for anode potential over SoC
        # severe Li plating -> clogging, steep resistance rise => probably high plating rates even at medium T + C-rates

        # a6: r_film (=R_film * C_nom) at T0
        # a7: temperature coefficient for r_film
        # a8: plating rate (ΔV_anode(<0) -> relative Δq_loss per second)

        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            q_loss_others = dq_loss.cumsum()
            q_loss_plating_total = 0
            q_loss_total = 0
            df_length = sei_potential.shape[0]
            dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64) * a6
                               * (np.exp(-a7 * (log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64) - T0))
                                  / (1.0 - a9 * q_loss_total)))
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0
                dq_c = -v_plating * a8 * t_func.iloc[i_start:i_end]
                dq_loss_plating.iloc[i_start:i_end] = dq_c
                q_loss_plating_total = q_loss_plating_total + dq_c.sum()
                q_loss_total = q_loss_others.iloc[i_end - 1] + q_loss_plating_total
                i_start = i_start + CHUNK_SIZE
            dq_loss = dq_loss + dq_loss_plating
        except RuntimeWarning:
            logging.log.warning("model_f016 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f,"
                                " a5 = %.12f, a6 = %.12f, a7 = %.12f, a8 = %.12f, a9 = %.12f caused a RuntimeWarning"
                                "(plating calc.)!" % (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f016" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]
        # this_fig = queue_entry["this_fig"]
        # i_row = queue_entry["i_row"]
        # i_col = queue_entry["i_col"]
        # if queue_entry["plot"]:
        #     add_model_trace_2("model_f016", log_df, mdl_cap_row, pid_index, pnr_index,
        #                       this_fig, i_row, i_col, rmse_cell)

        result_entry = {"log_df": log_df,  # "this_fig": this_fig,  #   # "result": result, -> don't need them anymore
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f016 END ===================================================================================================


# === model_f017 BEGIN =================================================================================================
def model_f017(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f017"
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f017_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("a0 = %.12f, a1 = %.3f, a2 = %.6f, a3 = %.12f, a4 = %.12f, a5 = %.12f, a6 = %.9f, a7 = %.9f, "
                  "a8 = %.12f, a9 = %.3f" % (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f017_get_delta_age(log_df, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
def model_f017_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = queue_entry["vars"]

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_mid = 3.8  # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        log_df[LABEL_C_CHG] = 0.0
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL

        sei_potential = (
                a0
                * np.exp(a1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(a2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (a4 * log_df[LABEL_C_CHG].astype(np.float64) * t_func  # a4 * dQ_chg
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid) ** 2)

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f017 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f017 cannot read dt, t_func too short")

        try:
            q_loss_sei_total = 0
            q_loss_cyclic_total = 0
            df_length = sei_potential.shape[0]
            dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_wearout.iloc[i_start:i_end]))
                diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - a5 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0
                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end]
                dq_loss.iloc[i_start:i_end] = dq_a + dq_b
                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f017 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f,"
                                " a5 = %.12f, a6 = %.12f, a7 = %.12f, a8 = %.12f , a9 = %.12f caused a RuntimeWarning!"
                                % (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9))
            dq_loss = (sei_potential + cyclic_age_potential) * t_func

        # lithium plating is described in yangModelingLithiumPlating2017: Yang X.-G., Leng Y., Zhang G., et al.:
        #   "Modeling of lithium plating induced aging of lithium-ion batteries: Transition from linear to nonlinear
        #   aging"
        # and vonludersModelingLithiumPlating2019: von Lüders C., Keil J., Webersberger M., et al.:
        #   "Modeling of lithium plating and lithium stripping in lithium-ion batteries"
        # also see schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
        #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
        #   -> example for anode potential over SoC
        # severe Li plating -> clogging, steep resistance rise => probably high plating rates even at medium T + C-rates

        # a6: r_film (=R_film * C_nom) at T0
        # a7: temperature coefficient for r_film
        # a8: plating rate (ΔV_anode(<0) -> relative Δq_loss per second)

        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # q_loss_others = dq_loss.cumsum()
            q_loss_plating_total = 0
            # q_loss_total = 0
            df_length = sei_potential.shape[0]
            dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64) * a6
                               * (np.exp(-a7 * (log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64) - T0))
                                  / (1.0 - a9 * q_loss_plating_total)))  # q_loss_total)))
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0
                dq_c = -v_plating * a8 * t_func.iloc[i_start:i_end]
                dq_loss_plating.iloc[i_start:i_end] = dq_c
                q_loss_plating_total = q_loss_plating_total + dq_c.sum()
                # q_loss_total = q_loss_others.iloc[i_end - 1] + q_loss_plating_total
                i_start = i_start + CHUNK_SIZE
            dq_loss = dq_loss + dq_loss_plating
        except RuntimeWarning:
            logging.log.warning("model_f017 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f,"
                                " a5 = %.12f, a6 = %.12f, a7 = %.12f, a8 = %.12f, a9 = %.12f caused a RuntimeWarning"
                                "(plating calc.)!" % (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f017" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]
        # this_fig = queue_entry["this_fig"]
        # i_row = queue_entry["i_row"]
        # i_col = queue_entry["i_col"]
        # if queue_entry["plot"]:
        #     add_model_trace_2("model_f017", log_df, mdl_cap_row, pid_index, pnr_index,
        #                       this_fig, i_row, i_col, rmse_cell)

        result_entry = {"log_df": log_df,  # "this_fig": this_fig,  #   # "result": result, -> don't need them anymore
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f017 END ===================================================================================================


# === model_f018 BEGIN =================================================================================================
def model_f018(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f018"
    s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f018_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-7, w0: %.3fe-7, w1: %.3f, w2: %.3fe-6, p0: %.3f, "
                  "p1: %.4f, p2: %.3fe-7, p3: %.3f, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e7, w0*1.0e7, w1, w2*1.0e6, p0, p1, p2*1.0e7, p3, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f018_get_delta_age(log_df, s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, Vmid):
def model_f018_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, Vmid = queue_entry["vars"]

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        log_df[LABEL_C_CHG] = 0.0
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL

        sei_potential = (
                s0
                * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0 * log_df[LABEL_C_CHG].astype(np.float64) * t_func  # a4 * dQ_chg
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w1)

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f018 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f018 cannot read dt, t_func too short")

        try:
            q_loss_sei_total = 0
            q_loss_cyclic_total = 0
            df_length = sei_potential.shape[0]
            dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_wearout.iloc[i_start:i_end]))
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w2 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0
                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end]
                dq_loss.iloc[i_start:i_end] = dq_a + dq_b
                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f018 (???) call with s0 = %.12f, s1 = %.12f, s2 = %.12f, s3 = %.12f, w0 = %.12f,"
                                " w1 = %.3f, w2 = %.12f, p0 = %.12f, p1 = %.12f, p2 = %.12f , p3 = %.12f, Vmid = %.3f "
                                "caused a RuntimeWarning!" % (s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, Vmid))
            dq_loss = (sei_potential + cyclic_age_potential) * t_func

        # lithium plating is described in yangModelingLithiumPlating2017: Yang X.-G., Leng Y., Zhang G., et al.:
        #   "Modeling of lithium plating induced aging of lithium-ion batteries: Transition from linear to nonlinear
        #   aging"
        # and vonludersModelingLithiumPlating2019: von Lüders C., Keil J., Webersberger M., et al.:
        #   "Modeling of lithium plating and lithium stripping in lithium-ion batteries"
        # also see schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
        #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
        #   -> example for anode potential over SoC
        # severe Li plating -> clogging, steep resistance rise => probably high plating rates even at medium T + C-rates

        # a6: r_film (=R_film * C_nom) at T0
        # a7: temperature coefficient for r_film
        # a8: plating rate (ΔV_anode(<0) -> relative Δq_loss per second)

        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            q_loss_others = dq_loss.cumsum()
            q_loss_plating_total = 0
            q_loss_total = 0
            df_length = sei_potential.shape[0]
            dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64) * p0
                               * np.exp(-p1 * (log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64) - T0))
                               * (1.0 + (p3 * q_loss_total)**3))  # q_loss_plating_total)))
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0
                dq_c = -v_plating * p2 * t_func.iloc[i_start:i_end]
                dq_loss_plating.iloc[i_start:i_end] = dq_c
                q_loss_plating_total = q_loss_plating_total + dq_c.sum()
                q_loss_total = q_loss_others.iloc[i_end - 1] + q_loss_plating_total
                i_start = i_start + CHUNK_SIZE
            dq_loss = dq_loss + dq_loss_plating
        except RuntimeWarning:
            logging.log.warning("model_f018 (???) call with s0 = %.12f, s1 = %.12f, s2 = %.12f, s3 = %.12f, w0 = %.12f,"
                                " w1 = %.3f, w2 = %.12f, p0 = %.12f, p1 = %.12f, p2 = %.12f , p3 = %.12f, Vmid = %.3f "
                                "caused a RuntimeWarning (plating calc.)!"
                                % (s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f018" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]
        # this_fig = queue_entry["this_fig"]
        # i_row = queue_entry["i_row"]
        # i_col = queue_entry["i_col"]
        # if queue_entry["plot"]:
        #     add_model_trace_2("model_f018", log_df, mdl_cap_row, pid_index, pnr_index,
        #                       this_fig, i_row, i_col, rmse_cell)

        result_entry = {"log_df": log_df,  # "this_fig": this_fig,  #   # "result": result, -> don't need them anymore
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f018 END ===================================================================================================


# === model_f019 BEGIN =================================================================================================
def model_f019(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f019"
    s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f019_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-7, w0: %.3fe-7, w1: %.3f, w2: %.3fe-6, p0: %.3f, "
                  "p1: %.4f, p2: %.3fe-7, p3: %.3f, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e7, w0*1.0e7, w1, w2*1.0e6, p0, p1, p2*1.0e7, p3, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f019_get_delta_age(log_df, s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, Vmid):
def model_f019_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, Vmid = queue_entry["vars"]

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        log_df[LABEL_C_CHG] = 0.0
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL

        sei_potential = (
                s0
                * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0 * log_df[LABEL_C_CHG].astype(np.float64) * t_func  # a4 * dQ_chg
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w1)

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f019 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f019 cannot read dt, t_func too short")

        try:
            q_loss_sei_total = 0
            q_loss_cyclic_total = 0
            df_length = sei_potential.shape[0]
            dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_wearout.iloc[i_start:i_end]))
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w2 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0
                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end]
                dq_loss.iloc[i_start:i_end] = dq_a + dq_b
                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f019 (???) call with s0 = %.12f, s1 = %.12f, s2 = %.12f, s3 = %.12f, w0 = %.12f,"
                                " w1 = %.3f, w2 = %.12f, p0 = %.12f, p1 = %.12f, p2 = %.12f , p3 = %.12f, Vmid = %.3f "
                                "caused a RuntimeWarning!" % (s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, Vmid))
            dq_loss = (sei_potential + cyclic_age_potential) * t_func

        # lithium plating is described in yangModelingLithiumPlating2017: Yang X.-G., Leng Y., Zhang G., et al.:
        #   "Modeling of lithium plating induced aging of lithium-ion batteries: Transition from linear to nonlinear
        #   aging"
        # and vonludersModelingLithiumPlating2019: von Lüders C., Keil J., Webersberger M., et al.:
        #   "Modeling of lithium plating and lithium stripping in lithium-ion batteries"
        # also see schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
        #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
        #   -> example for anode potential over SoC
        # severe Li plating -> clogging, steep resistance rise => probably high plating rates even at medium T + C-rates

        # a6: r_film (=R_film * C_nom) at T0
        # a7: temperature coefficient for r_film
        # a8: plating rate (ΔV_anode(<0) -> relative Δq_loss per second)

        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # q_loss_others = dq_loss.cumsum()
            q_loss_plating_total = 0
            # q_loss_total = 0
            df_length = sei_potential.shape[0]
            dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64) * p0
                               * np.exp(-p1 * (log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64) - T0))
                               / (1.0 - p3 * q_loss_plating_total))  # q_loss_plating_total)))
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0
                dq_c = -v_plating * p2 * t_func.iloc[i_start:i_end]
                dq_loss_plating.iloc[i_start:i_end] = dq_c
                q_loss_plating_total = q_loss_plating_total + dq_c.sum()
                # q_loss_total = q_loss_others.iloc[i_end - 1] + q_loss_plating_total
                i_start = i_start + CHUNK_SIZE
            dq_loss = dq_loss + dq_loss_plating
        except RuntimeWarning:
            logging.log.warning("model_f019 (???) call with s0 = %.12f, s1 = %.12f, s2 = %.12f, s3 = %.12f, w0 = %.12f,"
                                " w1 = %.3f, w2 = %.12f, p0 = %.12f, p1 = %.12f, p2 = %.12f , p3 = %.12f, Vmid = %.3f "
                                "caused a RuntimeWarning (plating calc.)!"
                                % (s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f019" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f019 END ===================================================================================================


# === model_f020 BEGIN =================================================================================================
def model_f020(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f020"
    s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f020_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-7, w0: %.3fe-7, w1: %.3f, w2: %.3fe-6, p0: %.3f, "
                  "p1: %.4f, p2: %.3fe-7, p3: %.3f, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e7, w0*1.0e7, w1, w2*1.0e6, p0, p1, p2*1.0e7, p3, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f020_get_delta_age(log_df, s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, Vmid):
def model_f020_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, Vmid = queue_entry["vars"]

        # if pid_index == 64:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        log_df[LABEL_C_CHG] = 0.0
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL

        sei_potential = (
                s0
                * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0 * log_df[LABEL_C_CHG].astype(np.float64) * t_func  # a4 * dQ_chg
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w1)

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f020 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f020 cannot read dt, t_func too short")

        try:
            q_loss_sei_total = 0
            q_loss_cyclic_total = 0
            df_length = sei_potential.shape[0]
            dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_wearout.iloc[i_start:i_end]))
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w2 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0
                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end]
                dq_loss.iloc[i_start:i_end] = dq_a + dq_b
                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f020 (???) call with s0 = %.12f, s1 = %.12f, s2 = %.12f, s3 = %.12f, w0 = %.12f,"
                                " w1 = %.3f, w2 = %.12f, p0 = %.12f, p1 = %.12f, p2 = %.12f , p3 = %.12f, Vmid = %.3f "
                                "caused a RuntimeWarning!" % (s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, Vmid))
            dq_loss = (sei_potential + cyclic_age_potential) * t_func

        # lithium plating is described in yangModelingLithiumPlating2017: Yang X.-G., Leng Y., Zhang G., et al.:
        #   "Modeling of lithium plating induced aging of lithium-ion batteries: Transition from linear to nonlinear
        #   aging"
        # and vonludersModelingLithiumPlating2019: von Lüders C., Keil J., Webersberger M., et al.:
        #   "Modeling of lithium plating and lithium stripping in lithium-ion batteries"
        # also see schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
        #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
        #   -> example for anode potential over SoC
        # severe Li plating -> clogging, steep resistance rise => probably high plating rates even at medium T + C-rates

        # a6: r_film (=R_film * C_nom) at T0
        # a7: temperature coefficient for r_film
        # a8: plating rate (ΔV_anode(<0) -> relative Δq_loss per second)

        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            q_loss_others = dq_loss.cumsum()
            q_loss_plating_total = 0
            q_loss_total = 0
            df_length = sei_potential.shape[0]
            dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64) * p0
                               * np.exp(-p1 * (log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64) - T0))
                               / max(1.0 - p3 * q_loss_total, 1.0e-9))  # q_loss_plating_total)))
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0
                dq_c = -v_plating * p2 * t_func.iloc[i_start:i_end]
                dq_loss_plating.iloc[i_start:i_end] = dq_c
                q_loss_plating_total = q_loss_plating_total + dq_c.sum()
                q_loss_total = q_loss_others.iloc[i_end - 1] + q_loss_plating_total
                i_start = i_start + CHUNK_SIZE
            dq_loss = dq_loss + dq_loss_plating
        except RuntimeWarning:
            logging.log.warning("model_f020 (???) call with s0 = %.12f, s1 = %.12f, s2 = %.12f, s3 = %.12f, w0 = %.12f,"
                                " w1 = %.3f, w2 = %.12f, p0 = %.12f, p1 = %.12f, p2 = %.12f , p3 = %.12f, Vmid = %.3f "
                                "caused a RuntimeWarning (plating calc.)!"
                                % (s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f020" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f020 END ===================================================================================================


# === model_f021 BEGIN =================================================================================================
def model_f021(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f021"
    s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, c0, c1, c2, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f021_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-7, w0: %.3fe-7, w1: %.3f, w2: %.3fe-6, p0: %.3f, "
                  "p1: %.4f, p2: %.3fe-7, p3: %.3f, c0: %.3fe-9, c1: %.3f, c2 %.3fe-7, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e7, w0*1.0e7, w1, w2*1.0e6, p0, p1, p2*1.0e7, p3,
                     c0*1.0e9, c1, c2*1.0e7, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f021_get_delta_age(log_df, s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, c0, c1, c2, Vmid):
def model_f021_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, c0, c1, c2, Vmid = queue_entry["vars"]

        # if pid_index == 64:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        log_df[LABEL_C_CHG] = 0.0
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL

        sei_potential = (
                s0
                * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0 * log_df[LABEL_C_CHG].astype(np.float64) * t_func  # a4 * dQ_chg
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w1)
        corrosion_potential = (c0 * t_func  # c0 * dQ_chg
                               * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f021 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f021 cannot read dt, t_func too short")

        try:
            q_loss_sei_total = 0
            q_loss_cyclic_total = 0
            q_loss_corrosion_total = 0
            df_length = sei_potential.shape[0]
            dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_wearout.iloc[i_start:i_end]))
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w2 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0
                diff_c = (corrosion_potential.iloc[i_start:i_end] - c2 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end]
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c
                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                q_loss_corrosion_total = q_loss_corrosion_total + dq_c.sum()
                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f021 (???) call with s0 = %.12f, s1 = %.12f, s2 = %.12f, s3 = %.12f, w0 = %.12f,"
                                " w1 = %.3f, w2 = %.12f, p0 = %.12f, p1 = %.12f, p2 = %.12f , p3 = %.12f, "
                                "c0: %.3fe-9, c1: %.3f, c2 %.3fe-7, Vmid = %.3f caused a RuntimeWarning!"
                                % (s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, c0, c1, c2, Vmid))
            dq_loss = (sei_potential + cyclic_age_potential + corrosion_potential) * t_func

        # lithium plating is described in yangModelingLithiumPlating2017: Yang X.-G., Leng Y., Zhang G., et al.:
        #   "Modeling of lithium plating induced aging of lithium-ion batteries: Transition from linear to nonlinear
        #   aging"
        # and vonludersModelingLithiumPlating2019: von Lüders C., Keil J., Webersberger M., et al.:
        #   "Modeling of lithium plating and lithium stripping in lithium-ion batteries"
        # also see schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
        #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
        #   -> example for anode potential over SoC
        # severe Li plating -> clogging, steep resistance rise => probably high plating rates even at medium T + C-rates

        # a6: r_film (=R_film * C_nom) at T0
        # a7: temperature coefficient for r_film
        # a8: plating rate (ΔV_anode(<0) -> relative Δq_loss per second)

        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            q_loss_others = dq_loss.cumsum()
            q_loss_plating_total = 0
            q_loss_total = 0
            df_length = sei_potential.shape[0]
            dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64) * p0
                               * np.exp(-p1 * (log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64) - T0))
                               / max(1.0 - p3 * q_loss_total, 1.0e-9))  # q_loss_plating_total)))
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0
                dq_c = -v_plating * p2 * t_func.iloc[i_start:i_end]
                dq_loss_plating.iloc[i_start:i_end] = dq_c
                q_loss_plating_total = q_loss_plating_total + dq_c.sum()
                q_loss_total = q_loss_others.iloc[i_end - 1] + q_loss_plating_total
                i_start = i_start + CHUNK_SIZE
            dq_loss = dq_loss + dq_loss_plating
        except RuntimeWarning:
            logging.log.warning("model_f021 (???) call with s0 = %.12f, s1 = %.12f, s2 = %.12f, s3 = %.12f, w0 = %.12f,"
                                " w1 = %.3f, w2 = %.12f, p0 = %.12f, p1 = %.12f, p2 = %.12f , p3 = %.12f, c0: %.3fe-9, "
                                "c1 %.3f, c2 %.3fe-7, Vmid = %.3f caused a RuntimeWarning (plating calc.)!"
                                % (s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, c0, c1, c2, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f021" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f021 END ===================================================================================================


# === model_f022 BEGIN =================================================================================================
def model_f022(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f022"
    s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, c0, c1, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f022_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, w0: %.3fe-9, w1: %.3f, w2: %.3fe-6, p0: %.3f, "
                  "p1: %.4f, p2: %.3fe-9, p3: %.3f, c0: %.3fe-9, c1: %.3fe-6, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e9, w1, w2*1.0e6, p0, p1, p2*1.0e9, p3,
                     c0*1.0e9, c1*1.0e6, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f022_get_delta_age(log_df, s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, c0, c1, Vmid):
def model_f022_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, c0, c1, Vmid = queue_entry["vars"]

        # if pid_index == 64:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = 3.3
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        log_df[LABEL_C_CHG] = 0.0
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL

        sei_potential = (
                s0
                * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0 * log_df[LABEL_C_CHG].astype(np.float64) * t_func  # a4 * dQ_chg
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w1)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0 * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64)))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f022 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f022 cannot read dt, t_func too short")

        try:
            q_loss_sei_total = 0
            q_loss_cyclic_total = 0
            q_loss_corrosion_total = 0
            df_length = sei_potential.shape[0]
            dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_wearout.iloc[i_start:i_end]))
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w2 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0
                diff_c = (corrosion_potential.iloc[i_start:i_end] - c1 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end]
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c
                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                q_loss_corrosion_total = q_loss_corrosion_total + dq_c.sum()
                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f022 (???) call with s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, "
                                "w0: %.3fe-9, w1: %.3f, w2: %.3fe-6, p0: %.3f, p1: %.4f, p2: %.3fe-9, p3: %.3f, "
                                "c0: %.3fe-9, c1: %.3fe-6, Vmid: %.3f caused a RuntimeWarning!"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e9, w1, w2 * 1.0e6, p0, p1, p2 * 1.0e9, p3,
                                   c0 * 1.0e9, c1 * 1.0e6, Vmid))
            dq_loss = (sei_potential + cyclic_age_potential + corrosion_potential) * t_func

        # lithium plating is described in yangModelingLithiumPlating2017: Yang X.-G., Leng Y., Zhang G., et al.:
        #   "Modeling of lithium plating induced aging of lithium-ion batteries: Transition from linear to nonlinear
        #   aging"
        # and vonludersModelingLithiumPlating2019: von Lüders C., Keil J., Webersberger M., et al.:
        #   "Modeling of lithium plating and lithium stripping in lithium-ion batteries"
        # also see schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
        #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
        #   -> example for anode potential over SoC
        # severe Li plating -> clogging, steep resistance rise => probably high plating rates even at medium T + C-rates

        # a6: r_film (=R_film * C_nom) at T0
        # a7: temperature coefficient for r_film
        # a8: plating rate (ΔV_anode(<0) -> relative Δq_loss per second)

        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            q_loss_others = dq_loss.cumsum()
            q_loss_plating_total = 0
            q_loss_total = 0
            df_length = sei_potential.shape[0]
            dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64) * p0
                               * np.exp(-p1 * (log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64) - T0))
                               / max(1.0 - p3 * q_loss_total, 1.0e-9))  # q_loss_plating_total)))
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0
                dq_c = -v_plating * p2 * t_func.iloc[i_start:i_end]
                dq_loss_plating.iloc[i_start:i_end] = dq_c
                q_loss_plating_total = q_loss_plating_total + dq_c.sum()
                q_loss_total = q_loss_others.iloc[i_end - 1] + q_loss_plating_total
                i_start = i_start + CHUNK_SIZE
            dq_loss = dq_loss + dq_loss_plating
        except RuntimeWarning:
            logging.log.warning("model_f022 (???) call with s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, "
                                "w0: %.3fe-9, w1: %.3f, w2: %.3fe-6, p0: %.3f, p1: %.4f, p2: %.3fe-9, p3: %.3f, "
                                "c0: %.3fe-9, c1: %.3fe-6, Vmid: %.3f caused a RuntimeWarning (plating calc.)!"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e9, w1, w2 * 1.0e6, p0, p1, p2 * 1.0e9, p3,
                                   c0 * 1.0e9, c1 * 1.0e6, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f022" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f022 END ===================================================================================================


# === model_f023 BEGIN =================================================================================================
def model_f023(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f023"
    s0, s1, s2, s3, w0, p0, p1, p2, p3, c0, c1, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f023_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, w0: %.3fe-9, p0: %.3f, p1: %.4f, p2: %.3fe-9, "
                  "p3: %.3f, c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e9, p0, p1, p2*1.0e9, p3,
                     c0*1.0e6, c1*1.0e6, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f023_get_delta_age(log_df, s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, c0, c1, Vmid):
def model_f023_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, p0, p1, p2, p3, c0, c1, Vmid = queue_entry["vars"]

        # if pid_index == 64:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = 3.3
        # V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        log_df[LABEL_C_CHG] = 0.0
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL

        sei_potential = (
                s0
                * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = w0 * log_df[LABEL_C_CHG].astype(np.float64) * t_func  # w0 * dQ_chg
        # corrosion_potential = (c0 * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0 * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64)))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f023 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f023 cannot read dt, t_func too short")

        try:
            q_loss_sei_total = 0
            # q_loss_cyclic_total = 0
            q_loss_corrosion_total = 0
            df_length = sei_potential.shape[0]
            dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_wearout.iloc[i_start:i_end]))
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
                # diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w2 * q_loss_cyclic_total)
                # diff_b[diff_b < 0] = 0.0
                diff_c = (corrosion_potential.iloc[i_start:i_end] - c1 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                dq_a = diff_a * t_func.iloc[i_start:i_end]
                # dq_b = diff_b * t_func.iloc[i_start:i_end]
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c
                dq_loss.iloc[i_start:i_end] = dq_a + dq_c
                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                # q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                q_loss_corrosion_total = q_loss_corrosion_total + dq_c.sum()
                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f023 (???) call with s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, "
                                "w0: %.3fe-9, p0: %.3f, p1: %.4f, p2: %.3fe-9, p3: %.3f, "
                                "c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f caused a RuntimeWarning!"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e9, p0, p1, p2 * 1.0e9, p3,
                                   c0 * 1.0e6, c1 * 1.0e6, Vmid))
            dq_loss = (sei_potential + corrosion_potential) * t_func

        dq_loss = dq_loss + cyclic_age_potential

        # lithium plating is described in yangModelingLithiumPlating2017: Yang X.-G., Leng Y., Zhang G., et al.:
        #   "Modeling of lithium plating induced aging of lithium-ion batteries: Transition from linear to nonlinear
        #   aging"
        # and vonludersModelingLithiumPlating2019: von Lüders C., Keil J., Webersberger M., et al.:
        #   "Modeling of lithium plating and lithium stripping in lithium-ion batteries"
        # also see schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
        #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
        #   -> example for anode potential over SoC
        # severe Li plating -> clogging, steep resistance rise => probably high plating rates even at medium T + C-rates

        # a6: r_film (=R_film * C_nom) at T0
        # a7: temperature coefficient for r_film
        # a8: plating rate (ΔV_anode(<0) -> relative Δq_loss per second)

        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            q_loss_others = dq_loss.cumsum()
            q_loss_plating_total = 0
            q_loss_total = 0
            df_length = sei_potential.shape[0]
            dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64) * p0
                               * np.exp(-p1 * (log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64) - T0))
                               / max(1.0 - p3 * q_loss_total, 1.0e-9))  # q_loss_plating_total)))
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0
                dq_c = -v_plating * p2 * t_func.iloc[i_start:i_end]
                dq_loss_plating.iloc[i_start:i_end] = dq_c
                q_loss_plating_total = q_loss_plating_total + dq_c.sum()
                q_loss_total = q_loss_others.iloc[i_end - 1] + q_loss_plating_total
                i_start = i_start + CHUNK_SIZE
            dq_loss = dq_loss + dq_loss_plating
        except RuntimeWarning:
            logging.log.warning("model_f023 (???) call with s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, "
                                "w0: %.3fe-9, p0: %.3f, p1: %.4f, p2: %.3fe-9, p3: %.3f, "
                                "c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f caused a RuntimeWarning (plating calc.)!"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e9, p0, p1, p2 * 1.0e9, p3,
                                   c0 * 1.0e6, c1 * 1.0e6, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f023" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f023 END ===================================================================================================


# === model_f024 BEGIN =================================================================================================
def model_f024(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f024"
    s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, c0, c1, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f024_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, w0: %.3fe-9, w1: %.3f, w2: %.3fe-6, p0: %.3f, "
                  "p1: %.1f, p2: %.3fe-9, p3: %.3f, c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e9, w1, w2*1.0e6, p0, p1, p2*1.0e9, p3,
                     c0*1.0e6, c1*1.0e6, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f024_get_delta_age(log_df, s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, c0, c1, Vmid):
def model_f024_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, c0, c1, Vmid = queue_entry["vars"]

        # if pid_index == 64:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = 3.3
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        log_df[LABEL_C_CHG] = 0.0
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL

        sei_potential = (
                s0
                * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0 * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w1)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0 * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64)))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f024 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f024 cannot read dt, t_func too short")

        try:
            q_loss_sei_total = 0
            q_loss_cyclic_total = 0
            q_loss_corrosion_total = 0
            df_length = sei_potential.shape[0]
            dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_wearout.iloc[i_start:i_end]))
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w2 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0
                diff_c = (corrosion_potential.iloc[i_start:i_end] - c1 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64)
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c
                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                q_loss_corrosion_total = q_loss_corrosion_total + dq_c.sum()
                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f024 (???) call with s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, "
                                "w0: %.3fe-9, w1: %.3f, w2: %.3fe-6, p0: %.3f, p1: %.1f, p2: %.3fe-9, p3: %.3f, "
                                "c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f caused a RuntimeWarning!"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e9, w1, w2 * 1.0e6, p0, p1, p2 * 1.0e9, p3,
                                   c0 * 1.0e6, c1 * 1.0e6, Vmid))
            dq_loss = (sei_potential + cyclic_age_potential + corrosion_potential) * t_func

        # lithium plating is described in yangModelingLithiumPlating2017: Yang X.-G., Leng Y., Zhang G., et al.:
        #   "Modeling of lithium plating induced aging of lithium-ion batteries: Transition from linear to nonlinear
        #   aging"
        # and vonludersModelingLithiumPlating2019: von Lüders C., Keil J., Webersberger M., et al.:
        #   "Modeling of lithium plating and lithium stripping in lithium-ion batteries"
        # also see schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
        #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
        #   -> example for anode potential over SoC
        # severe Li plating -> clogging, steep resistance rise => probably high plating rates even at medium T + C-rates

        # a6: r_film (=R_film * C_nom) at T0
        # a7: temperature coefficient for r_film
        # a8: plating rate (ΔV_anode(<0) -> relative Δq_loss per second)

        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            q_loss_others = dq_loss.cumsum()
            q_loss_plating_total = 0
            q_loss_total = 0
            df_length = sei_potential.shape[0]
            dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64) * p0
                               * np.exp(p1 * (1.0 / log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64)
                                              - 1.0 / T0))
                               / max((1.0 - p3 * q_loss_total)**2, 1.0e-9))  # q_loss_plating_total)))
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0
                dq_c = -v_plating * p2 * t_func.iloc[i_start:i_end]
                dq_loss_plating.iloc[i_start:i_end] = dq_c
                q_loss_plating_total = q_loss_plating_total + dq_c.sum()
                q_loss_total = q_loss_others.iloc[i_end - 1] + q_loss_plating_total
                i_start = i_start + CHUNK_SIZE
            dq_loss = dq_loss + dq_loss_plating
        except RuntimeWarning:
            logging.log.warning("model_f024 (???) call with s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, "
                                "w0: %.3fe-9, w1: %.3f, w2: %.3fe-6, p0: %.3f, p1: %.1f, p2: %.3fe-9, p3: %.3f, "
                                "c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f caused a RuntimeWarning (plating calc.)!"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e9, w1, w2 * 1.0e6, p0, p1, p2 * 1.0e9, p3,
                                   c0 * 1.0e6, c1 * 1.0e6, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f024" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f024 END ===================================================================================================


# === model_f025 BEGIN =================================================================================================
def model_f025(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f025"
    s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, p4, c0, c1, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f025_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, w0: %.3fe-9, w1: %.3f, w2: %.3fe-6, p0: %.3f, "
                  "p1: %.1f, p2: %.3fe-9, p3: %.3f, p4: %.3f, c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e9, w1, w2*1.0e6, p0, p1, p2*1.0e9, p3, p4,
                     c0*1.0e6, c1*1.0e6, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f025_get_delta_age(log_df, s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, p4, c0, c1, Vmid):
def model_f025_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, p4, c0, c1, Vmid = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = 3.3
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        log_df[LABEL_C_CHG] = 0.0
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        log_df.loc[cond_chg, LABEL_C_CHG] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL

        sei_potential = (
                s0
                * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0 * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w1)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0 * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64)))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f025 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f025 cannot read dt, t_func too short")

        try:
            q_loss_sei_total = 0
            q_loss_cyclic_total = 0
            q_loss_corrosion_total = 0
            df_length = sei_potential.shape[0]
            dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_wearout.iloc[i_start:i_end]))
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced
                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w2 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0
                diff_c = (corrosion_potential.iloc[i_start:i_end] - c1 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64)
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c
                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                q_loss_corrosion_total = q_loss_corrosion_total + dq_c.sum()
                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f025 (???) call with s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, "
                                "w0: %.3fe-9, w1: %.3f, w2: %.3fe-6, p0: %.3f, p1: %.1f, p2: %.3fe-9, p3: %.3f, "
                                "p4: %.3f, c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f caused a RuntimeWarning!"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e9, w1, w2 * 1.0e6, p0, p1, p2 * 1.0e9, p3,
                                   p4, c0 * 1.0e6, c1 * 1.0e6, Vmid))
            dq_loss = (sei_potential + cyclic_age_potential + corrosion_potential) * t_func

        # lithium plating is described in yangModelingLithiumPlating2017: Yang X.-G., Leng Y., Zhang G., et al.:
        #   "Modeling of lithium plating induced aging of lithium-ion batteries: Transition from linear to nonlinear
        #   aging"
        # and vonludersModelingLithiumPlating2019: von Lüders C., Keil J., Webersberger M., et al.:
        #   "Modeling of lithium plating and lithium stripping in lithium-ion batteries"
        # also see schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
        #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
        #   -> example for anode potential over SoC
        # severe Li plating -> clogging, steep resistance rise => probably high plating rates even at medium T + C-rates

        # a6: r_film (=R_film * C_nom) at T0
        # a7: temperature coefficient for r_film
        # a8: plating rate (ΔV_anode(<0) -> relative Δq_loss per second)

        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            q_loss_others = dq_loss.cumsum()
            q_loss_plating_total = 0
            q_loss_total = 0
            df_length = sei_potential.shape[0]
            dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - (log_df[LABEL_C_CHG].iloc[i_start:i_end].astype(np.float64))**p4 * p0
                               * np.exp(p1 * (1.0 / log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64)
                                              - 1.0 / T0))
                               / max((1.0 - p3 * q_loss_total)**2, 1.0e-9))  # q_loss_plating_total)))
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0
                dq_c = -v_plating * p2 * t_func.iloc[i_start:i_end]
                dq_loss_plating.iloc[i_start:i_end] = dq_c
                q_loss_plating_total = q_loss_plating_total + dq_c.sum()
                q_loss_total = q_loss_others.iloc[i_end - 1] + q_loss_plating_total
                i_start = i_start + CHUNK_SIZE
            dq_loss = dq_loss + dq_loss_plating
        except RuntimeWarning:
            logging.log.warning("model_f025 (???) call with s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, "
                                "w0: %.3fe-9, w1: %.3f, w2: %.3fe-6, p0: %.3f, p1: %.1f, p2: %.3fe-9, p3: %.3f, p4: "
                                "%.3f, c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f caused a RuntimeWarning (plating calc.)!"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e9, w1, w2 * 1.0e6, p0, p1, p2 * 1.0e9, p3,
                                   p4, c0 * 1.0e6, c1 * 1.0e6, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f025" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f025 END ===================================================================================================


# === model_f026 BEGIN =================================================================================================
def model_f026(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f026"
    s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, p4, c0, c1, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f026_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, w0: %.3fe-9, w1: %.3f, w2: %.3fe-6, p0: %.3f, "
                  "p1: %.1f, p2: %.3fe-9, p3: %.3f, p4: %.3f, c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e9, w1, w2*1.0e6, p0, p1, p2*1.0e9, p3, p4,
                     c0*1.0e6, c1*1.0e6, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f026_get_delta_age(log_df, s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, p4, c0, c1, Vmid):
def model_f026_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, p4, c0, c1, Vmid = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = 3.3
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        c_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        c_rate_rel.loc[cond_chg] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL

        sei_potential = (
                s0
                * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0 * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w1)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0 * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64)))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f026 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f026 cannot read dt, t_func too short")

        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = c_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w2 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c1 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0

                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi**p4 * p0
                             * np.exp(p1 * (1.0 / log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64)
                                            - 1.0 / T0))
                             / max((1.0 - p3 * q_loss_plating_total) ** 2, 1.0e-5))  # q_loss_total
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                dq_d = -v_plating * p2 * t_func.iloc[i_start:i_end]

                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                q_loss_corrosion_total = q_loss_corrosion_total + dq_c.sum()
                q_loss_plating_total = q_loss_plating_total + dq_d.sum()

                q_loss_total = q_loss_sei_total + q_loss_cyclic_total + q_loss_corrosion_total + q_loss_plating_total

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f026 (???) call with s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, "
                                "w0: %.3fe-9, w1: %.3f, w2: %.3fe-6, p0: %.3f, p1: %.1f, p2: %.3fe-9, p3: %.3f, "
                                "p4: %.3f, c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f caused a RuntimeWarning!"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e9, w1, w2 * 1.0e6, p0, p1, p2 * 1.0e9, p3,
                                   p4, c0 * 1.0e6, c1 * 1.0e6, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f026" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f026 END ===================================================================================================


# === model_f027 BEGIN =================================================================================================
def model_f027(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f027"
    s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, p4, p5, p6, c0, c1, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f027_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, w0: %.3fe-6, w1: %.2f, w2: %.3fe-6, p0: %.3f, "
                  "p1: %.0f, p2: %.3fe-9, p3: %.2f, p4: %.2f, p5: %.2f, p6: %.2f, c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6,
                     c0*1.0e6, c1*1.0e6, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f027_get_delta_age(log_df, s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, p4, p5, p6, c0, c1, Vmid):
def model_f027_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, p4, p5, p6, c0, c1, Vmid = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = 3.3
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        c_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        c_rate_rel.loc[cond_chg] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * (log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL)**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0 * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w1)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0 * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64)))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f027 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f027 cannot read dt, t_func too short")

        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = c_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w2 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c1 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0

                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi**p4 * p0
                             # * np.exp(p1 * (1.0 / log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64)
                             #                - 1.0 / T0))
                             * np.exp(p1 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0))
                             / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                dq_d = -v_plating * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2

                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                q_loss_corrosion_total = q_loss_corrosion_total + dq_c.sum()
                q_loss_plating_total = q_loss_plating_total + dq_d.sum()

                q_loss_total = q_loss_sei_total + q_loss_cyclic_total + q_loss_corrosion_total + q_loss_plating_total

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f027 (???) call with s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, "
                                "w0: %.3fe-6, w1: %.2f, w2: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.2f, p4: "
                                "%.2f, p5: %.2f, p6: %.2f, c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2 * 1.0e6, p0, p1, p2 * 1.0e9, p3,
                                   p4, p5, p6, c0 * 1.0e6, c1 * 1.0e6, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f027" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f027 END ===================================================================================================


# === model_f028 BEGIN =================================================================================================
def model_f028(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f028"
    s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, p4, p5, p6, c0, c1, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f028_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, w0: %.3fe-6, w1: %.2f, w2: %.3fe-6, p0: %.3f, "
                  "p1: %.0f, p2: %.3fe-9, p3: %.2f, p4: %.2f, p5: %.2f, p6: %.2f, c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6,
                     c0*1.0e6, c1*1.0e6, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f028_get_delta_age(log_df, s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, p4, p5, p6, c0, c1, Vmid):
def model_f028_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, p0, p1, p2, p3, p4, p5, p6, c0, c1, Vmid = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = 3.3
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        c_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        c_rate_rel.loc[cond_chg] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * (log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL)**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0 * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w1)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0 * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64)))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f028 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f028 cannot read dt, t_func too short")

        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = c_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w2 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c1 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0

                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi * p0
                             # * np.exp(p1 * (1.0 / log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64)
                             #                - 1.0 / T0))
                             * np.exp(p1 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0))
                             / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2

                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                q_loss_corrosion_total = q_loss_corrosion_total + dq_c.sum()
                q_loss_plating_total = q_loss_plating_total + dq_d.sum()

                q_loss_total = q_loss_sei_total + q_loss_cyclic_total + q_loss_corrosion_total + q_loss_plating_total

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f028 (???) call with s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, "
                                "w0: %.3fe-6, w1: %.2f, w2: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.2f, p4: "
                                "%.2f, p5: %.2f, p6: %.2f, c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2 * 1.0e6, p0, p1, p2 * 1.0e9, p3,
                                   p4, p5, p6, c0 * 1.0e6, c1 * 1.0e6, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f028" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f028 END ===================================================================================================


# === model_f029 BEGIN =================================================================================================
def model_f029(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f029"
    s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, c0, c1, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f029_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, w0: %.3fe-6, w1: %.1f, w2: %.2f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.2f, p4: %.2f, p5: %.2f, p6: %.2f, "
                  "c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6,
                     c0*1.0e6, c1*1.0e6, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f029_get_delta_age(log_df, s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, c0, c1, Vmid):
def model_f029_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, c0, c1, Vmid = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = 3.3
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        c_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        c_rate_rel.loc[cond_chg] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * (log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL)**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0 * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64)))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f029 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f029 cannot read dt, t_func too short")

        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = c_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c1 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0

                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi * p0
                             # * np.exp(p1 * (1.0 / log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64)
                             #                - 1.0 / T0))
                             * np.exp(p1 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0))
                             / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2

                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                q_loss_corrosion_total = q_loss_corrosion_total + dq_c.sum()
                q_loss_plating_total = q_loss_plating_total + dq_d.sum()

                q_loss_total = q_loss_sei_total + q_loss_cyclic_total + q_loss_corrosion_total + q_loss_plating_total

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f029 (???) call with s0: %.3fe-9, s1: %.1f, s2: %.3f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.1f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.2f, p4: "
                                "%.2f, p5: %.2f, p6: %.2f, c0: %.3fe-6, c1: %.3fe-6, Vmid: %.3f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, c0 * 1.0e6, c1 * 1.0e6, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f029" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f029 END ===================================================================================================


# === model_f030 BEGIN =================================================================================================
def model_f030(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f030"
    s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, c0, c1, c2, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f030_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.1f, "
                  "c0: %.2fe-6, c1: %.0f, c2: %.3fe-6, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6,
                     c0*1.0e6, c1, c2*1.0e6, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


# def model_f030_get_delta_age(log_df, s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, c0, c1, c2, Vmid):
def model_f030_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, c0, c1, c2, Vmid = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = 3.3
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        c_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        c_rate_rel.loc[cond_chg] = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * (log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL)**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64)))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f030 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f030 cannot read dt, t_func too short")

        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = c_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c2 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0

                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi * p0
                             # * np.exp(p1 * (1.0 / log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64)
                             #                - 1.0 / T0))
                             * np.exp(p1 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0))
                             / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2

                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                q_loss_corrosion_total = q_loss_corrosion_total + dq_c.sum()
                q_loss_plating_total = q_loss_plating_total + dq_d.sum()

                q_loss_total = q_loss_sei_total + q_loss_cyclic_total + q_loss_corrosion_total + q_loss_plating_total

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f030 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.1f, c0: %.2fe-6, c1: %.0f, c2: %.3fe-6, Vmid: %.3f caused a Warning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, c0 * 1.0e6, c1, c2 * 1.0e6, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f030" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f030 END ===================================================================================================


# === model_f031 BEGIN =================================================================================================
def model_f031(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f031"
    s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f031_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, "
                  "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7,
                     c0*1.0e6, c1, c2, c3*1.0e6, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("c0", "<br>c0"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f031_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, Vmid = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f031 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f031 cannot read dt, t_func too short")

        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # FIXME might be good to calculate time-/capacity-invariant parts outside of the loop
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi * p0
                             # * np.exp(p1 * (1.0 / log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64)
                             #                - 1.0 / T0))
                             * np.exp(p1 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0))
                             / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                q_loss_sei_total = q_loss_sei_total + dq_a.sum()
                q_loss_cyclic_total = q_loss_cyclic_total + dq_b.sum()
                q_loss_corrosion_total = q_loss_corrosion_total + dq_c.sum()
                q_loss_plating_total = q_loss_plating_total + dq_d.sum()

                q_loss_total = q_loss_sei_total + q_loss_cyclic_total + q_loss_corrosion_total + q_loss_plating_total

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f031 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, Vmid: %.3f"
                                " caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, c0 * 1.0e6, c1, c2, c3 * 1.0e6, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f031" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f031 END ===================================================================================================


# === model_f032 BEGIN =================================================================================================
def model_f032(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f032"
    s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f032_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, "
                  "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7,
                     c0*1.0e6, c1, c2, c3*1.0e6, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("c0", "<br>c0"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f032_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, Vmid = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f032 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f032 cannot read dt, t_func too short")

        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # FIXME might be good to calculate time-/capacity-invariant parts outside of the loop
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi * p0
                             # * np.exp(p1 * (1.0 / log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64)
                             #                - 1.0 / T0))
                             * np.exp(p1 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0))
                             / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_plating_total - p7))
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f032 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, Vmid: %.3f"
                                " caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, c0 * 1.0e6, c1, c2, c3 * 1.0e6, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f032" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f032 END ===================================================================================================


# === model_f033 BEGIN =================================================================================================
def model_f033(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f033"
    s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f033_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, "
                  "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7,
                     c0*1.0e6, c1, c2, c3*1.0e6, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("c0", "<br>c0"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f033_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, Vmid = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f033 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f033 cannot read dt, t_func too short")

        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # FIXME might be good to calculate time-/capacity-invariant parts outside of the loop
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi * p0
                             # * np.exp(p1 * (1.0 / log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64)
                             #                - 1.0 / T0))
                             * np.exp(p1 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0))
                             / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * max(q_loss_plating_total - p7, 0)**p6
                fac = 1.0
                if p6 > 0.0:
                    if p7 != 0.0:
                        fac = 1.0 + max((q_loss_plating_total - p7) / p7, 0.0)**p6
                    else:
                        fac = 1.0 + max(q_loss_plating_total, 0.0)**p6
                dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * fac
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f033 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, Vmid: %.3f"
                                " caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, c0 * 1.0e6, c1, c2, c3 * 1.0e6, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f033" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f033 END ===================================================================================================


# === model_f034 BEGIN =================================================================================================
def model_f034(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f034"
    s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f034_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, "
                  "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7,
                     c0*1.0e6, c1, c2, c3*1.0e6, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("c0", "<br>c0"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f034_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, Vmid = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f034 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f034 cannot read dt, t_func too short")

        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # FIXME might be good to calculate time-/capacity-invariant parts outside of the loop
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi * p0
                             # * np.exp(p1 * (1.0 / log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64)
                             #                - 1.0 / T0))
                             * np.exp(p1 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0))
                             / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                dq_d = (abs(v_plating)**p4 * t_func.iloc[i_start:i_end]
                        * p2 * np.exp(p6 * max(q_loss_plating_total - p7, 0.0)))
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f034 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, Vmid: %.3f"
                                " caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, c0 * 1.0e6, c1, c2, c3 * 1.0e6, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f034" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f034 END ===================================================================================================


# === model_f035 BEGIN =================================================================================================
def model_f035(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f035"
    s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f035_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, "
                  "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7,
                     c0*1.0e6, c1, c2, c3*1.0e6, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("c0", "<br>c0"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f035_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, Vmid = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f035 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f035 cannot read dt, t_func too short")

        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # FIXME might be good to calculate time-/capacity-invariant parts outside of the loop
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi * p0
                             # * np.exp(p1 * (1.0 / log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64)
                             #                - 1.0 / T0))
                             * np.exp(p1 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0))
                             / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                fac = 1.0
                if p6 > 0.0:
                    if p7 != 0.0:
                        fac = 1.0 + max((q_loss_total - p7) / p7, 0.0)**p6
                    else:
                        fac = 1.0 + max(q_loss_total, 0.0)**p6
                dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * fac
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f035 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, Vmid: %.3f"
                                " caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, c0 * 1.0e6, c1, c2, c3 * 1.0e6, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f035" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f035 END ===================================================================================================


# === model_f036 BEGIN =================================================================================================
def model_f036(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f036"
    s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f036_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, "
                  "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7,
                     c0*1.0e6, c1, c2, c3*1.0e6, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("c0", "<br>c0"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f036_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, Vmid = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f036 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f036 cannot read dt, t_func too short")

        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # FIXME might be good to calculate time-/capacity-invariant parts outside of the loop
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi * p0
                             # * np.exp(p1 * (1.0 / log_df[csv_label.T_CELL].iloc[i_start:i_end].astype(np.float64)
                             #                - 1.0 / T0))
                             * np.exp(p1 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0))
                             / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f036 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, Vmid: %.3f"
                                " caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, c0 * 1.0e6, c1, c2, c3 * 1.0e6, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f036" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f036 END ===================================================================================================


# === model_f037 BEGIN =================================================================================================
def model_f037(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f037"
    s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, c4, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f037_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, "
                  "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7,
                     c0*1.0e6, c1, c2, c3*1.0e6, c4, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("c0", "<br>c0"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f037_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, c4, Vmid = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f037 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f037 cannot read dt, t_func too short")

        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - s3 * max(q_loss_sei_total - 20.0 * q_loss_plating_total, 0.0))
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # FIXME might be good to calculate time-/capacity-invariant parts outside of the loop
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi * p0t.iloc[i_start:i_end]
                             / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f037 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, "
                                "Vmid: %.3f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f037" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f037 END ===================================================================================================


# === model_f038 BEGIN =================================================================================================
def model_f038(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f038"
    s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, p8, c0, c1, c2, c3, c4, Vmid = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f038_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, p8: %.4f, "
                  "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7, p8,
                     c0*1.0e6, c1, c2, c3*1.0e6, c4, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("c0", "<br>c0"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f038_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, p8, c0, c1, c2, c3, c4, Vmid = (
            queue_entry)["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f038 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f038 cannot read dt, t_func too short")

        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - s3 * max(q_loss_sei_total - 20.0 * q_loss_plating_total, 0.0))
                diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # FIXME might be good to calculate time-/capacity-invariant parts outside of the loop
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi**p8 * p0t.iloc[i_start:i_end]
                             # - (np.exp(0.646 * c_rate_rel_roi) - 0.908) * p0t.iloc[i_start:i_end]  # p8??
                             / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f038 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: p8: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, "
                                "c3: %.3fe-6, c4: %.3f, Vmid: %.3f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, p8, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f038" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f038 END ===================================================================================================


# === model_f039 BEGIN =================================================================================================
def model_f039(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f039"
    s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0, c1, c2, c3, c4, Vmid = (
        variable_list)
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f039_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, p8: %.4f, "
                  "p9: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7, p8,
                     p9, c0*1.0e6, c1, c2, c3*1.0e6, c4, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("c0", "<br>c0"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f039_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0, c1, c2, c3, c4, Vmid = (
            queue_entry)["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))
        # FIXME: maybe instead of using the T-dependency here, we need to have v_anode' = v_anode/t_internal_use, see
        #  eq. (12) in yangModelingLithiumPlating2017 -> then maybe "cut off" exp(c * v_anode') or make it:
        #  "exp(c1 * v_anode') - c2" --> "allow" a certain lithium plating? (try without first)

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f039 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f039 cannot read dt, t_func too short")

        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - s3 * max(q_loss_sei_total - 20.0 * q_loss_plating_total, 0.0))
                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p9 * q_loss_plating_total)
                diff_a = (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi**p8 * p0t.iloc[i_start:i_end]
                             # - (np.exp(0.646 * c_rate_rel_roi) - 0.908) * p0t.iloc[i_start:i_end]  # p8??
                             # / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                             * np.exp(p3 * q_loss_total))  # q_loss_plating_total
                # why "np.exp(p3 * q_loss_total)"? -> resistance increase (modeled different in
                # yangModelingLithiumPlating2017 -> eq. 14 + 15, where c_SEI/LI is roughly my q_loss_SEI/plating)
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                dq_d = ((np.exp(p4 * abs(v_plating)) - 1.0) * t_func.iloc[i_start:i_end]
                        * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0)))
                # why "((np.exp(p4 * abs(v_plating)) - 1.0)" -> positive feedback loop, "exponential increase of lithium
                #   plating rate": yangModelingLithiumPlating2017
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                # Although I'm not an electrochemistry expert, I think the plating model here roughly translates to
                #   equation (10) in yangModelingLithiumPlating2017:
                #   -> (phi_s - phi_e) = anode voltage?
                #   -> or (phi_s - phi_e - U_SEI)? -> maybe we need to use a lower v_anode potential
                #   -> R_film in my case is dependent on the temperature (might also have to do with the maximum Li
                #      intercalation rate??) and age of the cell (Yang: caused by SEI growth)

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f039 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: p8: %.4f, p9: %.4f, c0: %.2fe-6, "
                                "c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, Vmid: %.3f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, p8, p9, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f039" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f039 END ===================================================================================================


# === model_f040 BEGIN =================================================================================================
def model_f040(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f040"
    s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0, c1, c2, c3, c4, Vmid = (
        variable_list)
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f040_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, p8: %.4f, "
                  "p9: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, Vmid: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7, p8,
                     p9, c0*1.0e6, c1, c2, c3*1.0e6, c4, Vmid))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("c0", "<br>c0"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f040_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0, c1, c2, c3, c4, Vmid = (
            queue_entry)["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f040 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f040 cannot read dt, t_func too short")

        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - s3 * max(q_loss_sei_total - 20.0 * q_loss_plating_total, 0.0))
                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p9 * q_loss_plating_total)
                diff_a = (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi**p8 * p0
                             # / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                             * np.exp(p3 * q_loss_total))  # q_loss_plating_total
                # why "np.exp(p3 * q_loss_total)"? -> resistance increase (modeled different in
                # yangModelingLithiumPlating2017 -> eq. 14 + 15, where c_SEI/LI is roughly my q_loss_SEI/plating)
                # cond_pos = (v_plating > 0.0)
                # v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                # dq_d = (np.exp(-v_plating * p1 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0))
                #         * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0))) - p4
                dq_d = ((np.exp(-v_plating * p1 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0))
                        * np.exp(p6 * max(q_loss_total - p7, 0.0))) - p4) * t_func.iloc[i_start:i_end] * p2
                cond_no_plating = (dq_d < 0.0)
                dq_d[cond_no_plating] = 0.0
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                # Although I'm not an electrochemistry expert, I think the plating model here roughly translates to
                #   equation (10) in yangModelingLithiumPlating2017:
                #   -> (phi_s - phi_e) = anode voltage?
                #   -> or (phi_s - phi_e - U_SEI)? -> maybe we need to use a lower v_anode potential
                #   -> R_film in my case is dependent on the temperature (might also have to do with the maximum Li
                #      intercalation rate??) and age of the cell (Yang: caused by SEI growth)

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f040 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: p8: %.4f, p9: %.4f, c0: %.2fe-6, "
                                "c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, Vmid: %.3f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, p8, p9, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, Vmid))

        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f040" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f040 END ===================================================================================================


# === model_f041 BEGIN =================================================================================================
def model_f041(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f041"
    s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0, c1, c2, c3, c4, Vmid, Cn = (
        variable_list)
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Cn)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f041_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, p8: %.4f, "
                  "p9: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, Vmid: %.3f, Cn: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7, p8,
                     p9, c0*1.0e6, c1, c2, c3*1.0e6, c4, Vmid, Cn))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("c0", "<br>c0"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f041_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0, c1, c2, c3, c4, Vmid, Cn = (
            queue_entry)["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f041 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f041 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0)) * np.exp(s2 * (v_storage - V0))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = (sei_potential_storage - s3 * q_loss_sei_storage)
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f041 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: p8: %.4f, p9: %.4f, c0: %.2fe-6, "
                                "c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, Vmid: %.3f, Cn: %.3f "
                                "caused a RuntimeWarning (stor.)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, p8, p9, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, Vmid, Cn))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))
        # FIXME: maybe instead of using the T-dependency here, we need to have v_anode' = v_anode/t_internal_use, see
        #  eq. (12) in yangModelingLithiumPlating2017 -> then maybe "cut off" exp(c * v_anode') or make it:
        #  "exp(c1 * v_anode') - c2" --> "allow" a certain lithium plating? (try without first)

        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - s3 * max(q_loss_sei_total - 20.0 * q_loss_plating_total, 0.0))
                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p9 * q_loss_plating_total)
                diff_a = (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi**p8 * p0t.iloc[i_start:i_end]
                             # - (np.exp(0.646 * c_rate_rel_roi) - 0.908) * p0t.iloc[i_start:i_end]  # p8??
                             # / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                             * np.exp(p3 * q_loss_total))  # q_loss_plating_total
                # why "np.exp(p3 * q_loss_total)"? -> resistance increase (modeled different in
                # yangModelingLithiumPlating2017 -> eq. 14 + 15, where c_SEI/LI is roughly my q_loss_SEI/plating)
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                dq_d = ((np.exp(p4 * abs(v_plating)) - 1.0) * t_func.iloc[i_start:i_end]
                        * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0)))
                # why "((np.exp(p4 * abs(v_plating)) - 1.0)" -> positive feedback loop, "exponential increase of lithium
                #   plating rate": yangModelingLithiumPlating2017
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                # Although I'm not an electrochemistry expert, I think the plating model here roughly translates to
                #   equation (10) in yangModelingLithiumPlating2017:
                #   -> (phi_s - phi_e) = anode voltage?
                #   -> or (phi_s - phi_e - U_SEI)? -> maybe we need to use a lower v_anode potential
                #   -> R_film in my case is dependent on the temperature (might also have to do with the maximum Li
                #      intercalation rate??) and age of the cell (Yang: caused by SEI growth)

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f041 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: p8: %.4f, p9: %.4f, c0: %.2fe-6, "
                                "c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, Vmid: %.3f, Cn: %.3f "
                                "caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, p8, p9, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, Vmid, Cn))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f041" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f041 END ===================================================================================================


# === model_f042 BEGIN =================================================================================================
def model_f042(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f042"
    s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0, c1, c2, c3, c4, f0, Vmid, Cn = (
        variable_list)
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Cn)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f042_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, p8: %.4f, "
                  "p9: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7, p8,
                     p9, c0*1.0e6, c1, c2, c3*1.0e6, c4, f0, Vmid, Cn))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("c0", "<br>c0"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f042_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0, c1, c2, c3, c4, f0, Vmid, Cn = (
            queue_entry)["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0

        df_length = log_df.shape[0]

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f042 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f042 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0)) * np.exp(s2 * (v_storage - V0))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = f0 * (sei_potential_storage - s3 * q_loss_sei_storage)
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f042 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: p8: %.4f, p9: %.4f, c0: %.2fe-6, "
                                "c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f "
                                "caused a RuntimeWarning (stor.)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, p8, p9, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        sei_potential = (
                s0
                # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))
        # FIXME: maybe instead of using the T-dependency here, we need to have v_anode' = v_anode/t_internal_use, see
        #  eq. (12) in yangModelingLithiumPlating2017 -> then maybe "cut off" exp(c * v_anode') or make it:
        #  "exp(c1 * v_anode') - c2" --> "allow" a certain lithium plating? (try without first)

        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - s3 * max(q_loss_sei_total - 20.0 * q_loss_plating_total, 0.0))
                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p9 * q_loss_plating_total)
                diff_a = f0 * (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = f0 * (cyclic_age_potential.iloc[i_start:i_end] - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = f0 * (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi**p8 * p0t.iloc[i_start:i_end]
                             # - (np.exp(0.646 * c_rate_rel_roi) - 0.908) * p0t.iloc[i_start:i_end]  # p8??
                             # / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                             * np.exp(p3 * q_loss_total))  # q_loss_plating_total
                # why "np.exp(p3 * q_loss_total)"? -> resistance increase (modeled different in
                # yangModelingLithiumPlating2017 -> eq. 14 + 15, where c_SEI/LI is roughly my q_loss_SEI/plating)
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                dq_d = ((np.exp(p4 * abs(v_plating)) - 1.0) * t_func.iloc[i_start:i_end]
                        * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0)))
                # why "((np.exp(p4 * abs(v_plating)) - 1.0)" -> positive feedback loop, "exponential increase of lithium
                #   plating rate": yangModelingLithiumPlating2017
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                # Although I'm not an electrochemistry expert, I think the plating model here roughly translates to
                #   equation (10) in yangModelingLithiumPlating2017:
                #   -> (phi_s - phi_e) = anode voltage?
                #   -> or (phi_s - phi_e - U_SEI)? -> maybe we need to use a lower v_anode potential
                #   -> R_film in my case is dependent on the temperature (might also have to do with the maximum Li
                #      intercalation rate??) and age of the cell (Yang: caused by SEI growth)

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f042 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: p8: %.4f, p9: %.4f, c0: %.2fe-6, "
                                "c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f "
                                "caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, p8, p9, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f042" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f042 END ===================================================================================================


# === model_f043 BEGIN =================================================================================================
def model_f043(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f043"
    s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0, c1, c2, c3, c4, f0, Vmid, Cn = (
        variable_list)
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Cn)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f043_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, p8: %.4f, "
                  "p9: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7, p8,
                     p9, c0*1.0e6, c1, c2, c3*1.0e6, c4, f0, Vmid, Cn))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("c0", "<br>c0"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f043_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0, c1, c2, c3, c4, f0, Vmid, Cn = (
            queue_entry)["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        # V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V0_SEI = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0
        T0_SEI = cfg.T0 + MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC

        df_length = log_df.shape[0]

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f043 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f043 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        # sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0)) * np.exp(s2 * (v_storage - V0))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0_SEI)) * np.exp(s2 * (v_storage - V0_SEI))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = f0 * (sei_potential_storage - s3 * q_loss_sei_storage)
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f043 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: p8: %.4f, p9: %.4f, c0: %.2fe-6, "
                                "c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f "
                                "caused a RuntimeWarning (stor.)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, p8, p9, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        # sei_potential = (
        #         s0
        #         # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
        #         * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
        #         * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        # )
        sei_potential = (  # when using final model in publication/diss -> calculate s0 to match T0/V0 instead _SEI
                s0
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0_SEI))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0_SEI))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))
        # FIXME: maybe instead of using the T-dependency here, we need to have v_anode' = v_anode/t_internal_use, see
        #  eq. (12) in yangModelingLithiumPlating2017 -> then maybe "cut off" exp(c * v_anode') or make it:
        #  "exp(c1 * v_anode') - c2" --> "allow" a certain lithium plating? (try without first)

        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - s3 * max(q_loss_sei_total - 20.0 * q_loss_plating_total, 0.0))
                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p9 * q_loss_plating_total)
                diff_a = f0 * (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = f0 * (cyclic_age_potential.iloc[i_start:i_end] - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = f0 * (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi**p8 * p0t.iloc[i_start:i_end]
                             # - (np.exp(0.646 * c_rate_rel_roi) - 0.908) * p0t.iloc[i_start:i_end]  # p8??
                             # / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                             * np.exp(p3 * q_loss_total))  # q_loss_plating_total
                # why "np.exp(p3 * q_loss_total)"? -> resistance increase (modeled different in
                # yangModelingLithiumPlating2017 -> eq. 14 + 15, where c_SEI/LI is roughly my q_loss_SEI/plating)
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                dq_d = ((np.exp(p4 * abs(v_plating)) - 1.0) * t_func.iloc[i_start:i_end]
                        * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0)))
                # why "((np.exp(p4 * abs(v_plating)) - 1.0)" -> positive feedback loop, "exponential increase of lithium
                #   plating rate": yangModelingLithiumPlating2017
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                # Although I'm not an electrochemistry expert, I think the plating model here roughly translates to
                #   equation (10) in yangModelingLithiumPlating2017:
                #   -> (phi_s - phi_e) = anode voltage?
                #   -> or (phi_s - phi_e - U_SEI)? -> maybe we need to use a lower v_anode potential
                #   -> R_film in my case is dependent on the temperature (might also have to do with the maximum Li
                #      intercalation rate??) and age of the cell (Yang: caused by SEI growth)

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f043 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, "
                                "p5: %.1f, p6: %.3f, p7: %.4f, c0: p8: %.4f, p9: %.4f, c0: %.2fe-6, "
                                "c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f "
                                "caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, p8, p9, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f043" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f043 END ===================================================================================================


# === model_f044 BEGIN =================================================================================================
def model_f044(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f044"
    s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0, c1, c2, c3, c4, f0, Vmid, Cn = (
        variable_list)
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Cn)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f044_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6"
                  ", p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, p8: %.4f, "
                  "p9: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7,
                     p8, p9, c0*1.0e6, c1, c2, c3*1.0e6, c4, f0, Vmid, Cn))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("p6", "<br>p6"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f044_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0, c1, c2, c3, c4, f0, Vmid, Cn = (
            queue_entry)["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        # V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V0_SEI = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0
        T0_SEI = cfg.T0 + MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC

        df_length = log_df.shape[0]

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f044 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f044 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        # sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0)) * np.exp(s2 * (v_storage - V0))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0_SEI)) * np.exp(s2 * (v_storage - V0_SEI))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = f0 * (sei_potential_storage - s3 * q_loss_sei_storage)
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f044 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, "
                                "p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, c0: p8: %.4f, p9: %.4f, c0: %.2fe-6, "
                                "c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f "
                                "caused a RuntimeWarning (stor.)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, p8, p9, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        # sei_potential = (
        #         s0
        #         # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
        #         * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
        #         * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        # )
        sei_potential = (  # when using final model in publication/diss -> calculate s0 to match T0/V0 instead _SEI
                s0
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0_SEI))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0_SEI))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))
        # FIXME: maybe instead of using the T-dependency here, we need to have v_anode' = v_anode/t_internal_use, see
        #  eq. (12) in yangModelingLithiumPlating2017 -> then maybe "cut off" exp(c * v_anode') or make it:
        #  "exp(c1 * v_anode') - c2" --> "allow" a certain lithium plating? (try without first)

        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - s3 * max(q_loss_sei_total - 20.0 * q_loss_plating_total, 0.0))
                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p9 * q_loss_plating_total)
                diff_a = f0 * (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = f0 * (cyclic_age_potential.iloc[i_start:i_end] - w4 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = f0 * (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi**p8 * p0t.iloc[i_start:i_end]
                             # - (np.exp(0.646 * c_rate_rel_roi) - 0.908) * p0t.iloc[i_start:i_end]  # p8??
                             # / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                             * np.exp(p3 * q_loss_total))  # q_loss_plating_total
                # why "np.exp(p3 * q_loss_total)"? -> resistance increase (modeled different in
                # yangModelingLithiumPlating2017 -> eq. 14 + 15, where c_SEI/LI is roughly my q_loss_SEI/plating)
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi**w3
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                dq_d = ((np.exp(p4 * abs(v_plating)) - 1.0) * t_func.iloc[i_start:i_end]
                        * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0)))
                # why "((np.exp(p4 * abs(v_plating)) - 1.0)" -> positive feedback loop, "exponential increase of lithium
                #   plating rate": yangModelingLithiumPlating2017
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                # Although I'm not an electrochemistry expert, I think the plating model here roughly translates to
                #   equation (10) in yangModelingLithiumPlating2017:
                #   -> (phi_s - phi_e) = anode voltage?
                #   -> or (phi_s - phi_e - U_SEI)? -> maybe we need to use a lower v_anode potential
                #   -> R_film in my case is dependent on the temperature (might also have to do with the maximum Li
                #      intercalation rate??) and age of the cell (Yang: caused by SEI growth)

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f044 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, "
                                "p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, c0: p8: %.4f, p9: %.4f, c0: %.2fe-6, "
                                "c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f "
                                "caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, p8, p9, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f044" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f044 END ===================================================================================================


# === model_f045 BEGIN =================================================================================================
def model_f045(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f045"
    (s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, c0, c1, c2, c3, c4, f0,
     Vmid, Cn) = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Cn)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f045_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6"
                  ", p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, p8: %.4f, "
                  "p9: %.4f, p10: %.3f, p11: %.0f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f,"
                  "Vmid: %.3f, Cn: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7,
                     p8, p9, p10, p11, c0*1.0e6, c1, c2, c3*1.0e6, c4, f0, Vmid, Cn))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("p8", "<br>p8"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f045_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        (s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, c0, c1, c2, c3, c4, f0,
         Vmid, Cn) = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        # V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V0_SEI = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0
        T0_SEI = cfg.T0 + MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC

        df_length = log_df.shape[0]

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f045 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f045 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        # sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0)) * np.exp(s2 * (v_storage - V0))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0_SEI)) * np.exp(s2 * (v_storage - V0_SEI))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = f0 * (sei_potential_storage - s3 * q_loss_sei_storage)
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f045 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, "
                                "p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, c0: p8: %.4f, p9: %.4f, p10: %.3f, p11: %.0f, "
                                "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f"
                                " caused a RuntimeWarning (stor.)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, p8, p9, p10, p11,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        # sei_potential = (
        #         s0
        #         # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
        #         * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
        #         * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        # )
        sei_potential = (  # when using final model in publication/diss -> calculate s0 to match T0/V0 instead _SEI
                s0
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0_SEI))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0_SEI))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))
        # FIXME: maybe instead of using the T-dependency here, we need to have v_anode' = v_anode/t_internal_use, see
        #  eq. (12) in yangModelingLithiumPlating2017 -> then maybe "cut off" exp(c * v_anode') or make it:
        #  "exp(c1 * v_anode') - c2" --> "allow" a certain lithium plating? (try without first)

        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        plt_base = p2 * t_func * np.exp(p11 * (1.0 / t_internal_use - 1.0 / T0))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - s3 * max(q_loss_sei_total - 20.0 * q_loss_plating_total, 0.0))
                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p9 * q_loss_plating_total)
                diff_a = f0 * (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = f0 * (cyclic_age_potential.iloc[i_start:i_end] - w4 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = f0 * (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi**p8 * p0t.iloc[i_start:i_end]
                             # - (np.exp(0.646 * c_rate_rel_roi) - 0.908) * p0t.iloc[i_start:i_end]  # p8??
                             # / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                             * np.exp(p3 * q_loss_total))  # q_loss_plating_total
                # why "np.exp(p3 * q_loss_total)"? -> resistance increase (modeled different in
                # yangModelingLithiumPlating2017 -> eq. 14 + 15, where c_SEI/LI is roughly my q_loss_SEI/plating)
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi**w3
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                dq_d = ((np.exp(p4 * abs(v_plating)) - 1.0) * np.exp(p6 * max(q_loss_total - p7, 0.0))
                        * np.exp(p10 * c_rate_rel_roi) * plt_base.iloc[i_start:i_end])
                # why "((np.exp(p4 * abs(v_plating)) - 1.0)" -> positive feedback loop, "exponential increase of lithium
                #   plating rate": yangModelingLithiumPlating2017
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                # Although I'm not an electrochemistry expert, I think the plating model here roughly translates to
                #   equation (10) in yangModelingLithiumPlating2017:
                #   -> (phi_s - phi_e) = anode voltage?
                #   -> or (phi_s - phi_e - U_SEI)? -> maybe we need to use a lower v_anode potential
                #   -> R_film in my case is dependent on the temperature (might also have to do with the maximum Li
                #      intercalation rate??) and age of the cell (Yang: caused by SEI growth)

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f045 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, "
                                "p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, c0: p8: %.4f, p9: %.4f, p10: %.3f, p11: %.0f, "
                                "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f"
                                " caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, p8, p9, p10, p11, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4,
                                   f0, Vmid, Cn))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f045" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f045 END ===================================================================================================


# === model_f046 BEGIN =================================================================================================
def model_f046(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f046"
    (s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, c0, c1, c2, c3, c4, f0,
     Vmid, Cn) = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Cn)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f046_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6"
                  ", p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, p8: %.4f, "
                  "p9: %.4f, p10: %.3f, p11: %.0f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f,"
                  "Vmid: %.3f, Cn: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2*1.0e9, p3, p4, p5, p6, p7,
                     p8, p9, p10, p11, c0*1.0e6, c1, c2, c3*1.0e6, c4, f0, Vmid, Cn))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("p8", "<br>p8"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f046_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        (s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, c0, c1, c2, c3, c4, f0,
         Vmid, Cn) = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        # V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V0_SEI = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0
        T0_SEI = cfg.T0 + MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC

        df_length = log_df.shape[0]

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f046 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f046 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        # sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0)) * np.exp(s2 * (v_storage - V0))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0_SEI)) * np.exp(s2 * (v_storage - V0_SEI))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = f0 * (sei_potential_storage - s3 * q_loss_sei_storage)
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f046 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, "
                                "p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, c0: p8: %.4f, p9: %.4f, p10: %.3f, p11: %.0f, "
                                "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f"
                                " caused a RuntimeWarning (stor.)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, p8, p9, p10, p11,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64)
                          + p5 * c_rate_rel**2)

        # sei_potential = (
        #         s0
        #         # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
        #         * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
        #         * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        # )
        sei_potential = (  # when using final model in publication/diss -> calculate s0 to match T0/V0 instead _SEI
                s0
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0_SEI))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0_SEI))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))
        # FIXME: maybe instead of using the T-dependency here, we need to have v_anode' = v_anode/t_internal_use, see
        #  eq. (12) in yangModelingLithiumPlating2017 -> then maybe "cut off" exp(c * v_anode') or make it:
        #  "exp(c1 * v_anode') - c2" --> "allow" a certain lithium plating? (try without first)

        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        plt_base = p2 * t_func * np.exp(p11 * (1.0 / t_internal_use - 1.0 / T0))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - s3 * max(q_loss_sei_total - 20.0 * q_loss_plating_total, 0.0))
                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p9 * q_loss_plating_total)
                diff_a = f0 * (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = f0 * (cyclic_age_potential.iloc[i_start:i_end] - w4 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = f0 * (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - c_rate_rel_roi**p8 * p0t.iloc[i_start:i_end]
                             # - (np.exp(0.646 * c_rate_rel_roi) - 0.908) * p0t.iloc[i_start:i_end]  # p8??
                             # / max(1.0 - p3 * q_loss_total, 1.0e-5))  # q_loss_plating_total
                             * np.exp(p3 * q_loss_total))  # q_loss_plating_total
                # why "np.exp(p3 * q_loss_total)"? -> resistance increase (modeled different in
                # yangModelingLithiumPlating2017 -> eq. 14 + 15, where c_SEI/LI is roughly my q_loss_SEI/plating)
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi**w3
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                dq_d = (abs(v_plating)**p4 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                        * np.exp(p10 * c_rate_rel_roi) * plt_base.iloc[i_start:i_end])
                # why "((np.exp(p4 * abs(v_plating)) - 1.0)" -> positive feedback loop, "exponential increase of lithium
                #   plating rate": yangModelingLithiumPlating2017
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                # Although I'm not an electrochemistry expert, I think the plating model here roughly translates to
                #   equation (10) in yangModelingLithiumPlating2017:
                #   -> (phi_s - phi_e) = anode voltage?
                #   -> or (phi_s - phi_e - U_SEI)? -> maybe we need to use a lower v_anode potential
                #   -> R_film in my case is dependent on the temperature (might also have to do with the maximum Li
                #      intercalation rate??) and age of the cell (Yang: caused by SEI growth)

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f046 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3fe-9, p3: %.1f, "
                                "p4: %.1f, p5: %.1f, p6: %.3f, p7: %.4f, c0: p8: %.4f, p9: %.4f, p10: %.3f, p11: %.0f, "
                                "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f"
                                " caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2 * 1.0e9,
                                   p3, p4, p5, p6, p7, p8, p9, p10, p11, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4,
                                   f0, Vmid, Cn))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f046" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f046 END ===================================================================================================


# === model_f047 BEGIN =================================================================================================
def model_f047(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f047"
    s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, c4, f0, Vmid, Cn = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Cn)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f047_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6"
                  ", p0: %.3f, p1: %.0f, p2: %.3f, p3: %.3fe-9, p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3f, "
                  "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2, p3*1.0e9, p4, p5, p6, p7,
                     c0*1.0e6, c1, c2, c3*1.0e6, c4, f0, Vmid, Cn))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("c0", "<br>c0"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f047_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        (s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, c0, c1, c2, c3, c4, f0, Vmid, Cn
         ) = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        # V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V0_SEI = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0
        T0_SEI = cfg.T0 + MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC

        df_length = log_df.shape[0]

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f047 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f047 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        # sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0)) * np.exp(s2 * (v_storage - V0))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0_SEI)) * np.exp(s2 * (v_storage - V0_SEI))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = f0 * (sei_potential_storage - s3 * q_loss_sei_storage)
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f047 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3f, p3: %.3fe-9, "
                                "p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, "
                                "c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f caused a RuntimeWarning (stor.)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2,
                                   p3 * 1.0e9, p4, p5, p6, p7, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        # t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64) + p5 * c_rate_rel**2)
        t_internal_use = log_df[csv_label.T_CELL].astype(np.float64)

        # sei_potential = (
        #         s0
        #         # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
        #         * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
        #         * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        # )
        sei_potential = (  # when using final model in publication/diss -> calculate s0 to match T0/V0 instead _SEI
                s0
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0_SEI))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0_SEI))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))
        # FIXME: maybe instead of using the T-dependency here, we need to have v_anode' = v_anode/t_internal_use, see
        #  eq. (12) in yangModelingLithiumPlating2017 -> then maybe "cut off" exp(c * v_anode') or make it:
        #  "exp(c1 * v_anode') - c2" --> "allow" a certain lithium plating? (try without first)

        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # plt_base = p2 * t_func * np.exp(p11 * (1.0 / t_internal_use - 1.0 / T0))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - s3 * max(q_loss_sei_total - 20.0 * q_loss_plating_total, 0.0))
                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p7 * q_loss_plating_total)
                diff_a = f0 * (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = f0 * (cyclic_age_potential.iloc[i_start:i_end] - w4 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = f0 * (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - p0t.iloc[i_start:i_end] * c_rate_rel_roi
                               * np.exp(p2 * q_loss_total))  # q_loss_plating_total
                # why "np.exp(p3 * q_loss_total)"? -> resistance increase (modeled different in
                # yangModelingLithiumPlating2017 -> eq. 14 + 15, where c_SEI/LI is roughly my q_loss_SEI/plating)
                cond_pos = (v_plating > 0.0)
                v_plating[cond_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * c_rate_rel_roi**w3
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                # dq_d = (abs(v_plating)**p4 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                #         * np.exp(p10 * c_rate_rel_roi) * plt_base.iloc[i_start:i_end])
                dq_d = (p3 * np.exp(-p4 * v_plating / t_internal_use.iloc[i_start:i_end])
                        * np.exp(p5 * max(q_loss_total - p6, 0.0)))
                dq_d[cond_pos] = 0.0
                # why "((np.exp(p4 * abs(v_plating)) - 1.0)" -> positive feedback loop, "exponential increase of lithium
                #   plating rate": yangModelingLithiumPlating2017
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                # Although I'm not an electrochemistry expert, I think the plating model here roughly translates to
                #   equation (10) in yangModelingLithiumPlating2017:
                #   -> (phi_s - phi_e) = anode voltage?
                #   -> or (phi_s - phi_e - U_SEI)? -> maybe we need to use a lower v_anode potential
                #   -> R_film in my case is dependent on the temperature (might also have to do with the maximum Li
                #      intercalation rate??) and age of the cell (Yang: caused by SEI growth)

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f047 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3f, p3: %.3fe-9, "
                                "p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, "
                                "c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2,
                                   p3 * 1.0e9, p4, p5, p6, p7, c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f047" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f047 END ===================================================================================================


# === model_f048 BEGIN =================================================================================================
def model_f048(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f048"
    s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0, c1, c2, c3, c4, f0, Vmid, Cn = (
        variable_list)
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Cn)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f048_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6"
                  ", p0: %.3f, p1: %.0f, p2: %.3f, p3: %.3fe-4, p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3f, p8: %.3f, "
                  "p9: %.3f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2, p3*1.0e4, p4, p5, p6, p7,
                     p8, p9, c0*1.0e6, c1, c2, c3*1.0e6, c4, f0, Vmid, Cn))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("p6", "<br>p6"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f048_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        (s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0, c1, c2, c3, c4, f0, Vmid, Cn
         ) = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        # V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V0_SEI = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0
        T0_SEI = cfg.T0 + MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC

        df_length = log_df.shape[0]

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f048 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f048 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        # sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0)) * np.exp(s2 * (v_storage - V0))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0_SEI)) * np.exp(s2 * (v_storage - V0_SEI))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = f0 * (sei_potential_storage - s3 * q_loss_sei_storage)
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f048 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3f, p3: %.3fe-4, "
                                "p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3f, p8: %.3f, p9: %.3f, c0: %.2fe-6, c1: %.0f, "
                                "c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f "
                                "caused a RuntimeWarning (stor.)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2,
                                   p3 * 1.0e4, p4, p5, p6, p7, p8, p9,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        # t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64) + p5 * c_rate_rel**2)
        t_internal_use = log_df[csv_label.T_CELL].astype(np.float64)

        # sei_potential = (
        #         s0
        #         # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
        #         * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
        #         * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        # )
        sei_potential = (  # when using final model in publication/diss -> calculate s0 to match T0/V0 instead _SEI
                s0
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0_SEI))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0_SEI))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?) -> already in cyclic wearout?

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))  # FIXME: other T-dependency?
        p4_abs_t = p4 + cfg.T0
        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # plt_base = p2 * t_func * np.exp(p11 * (1.0 / t_internal_use - 1.0 / T0))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - s3 * max(q_loss_sei_total - 20.0 * q_loss_plating_total, 0.0))
                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p9 * q_loss_plating_total)
                diff_a = f0 * (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = f0 * (cyclic_age_potential.iloc[i_start:i_end] - w4 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = f0 * (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - p0t.iloc[i_start:i_end] * c_rate_rel_roi
                               * np.exp(p2 * q_loss_total))  # q_loss_plating_total
                # why "np.exp(p3 * q_loss_total)"? -> resistance increase (modeled different in
                # yangModelingLithiumPlating2017 -> eq. 14 + 15, where c_SEI/LI is roughly my q_loss_SEI/plating)
                cond_v_plating_pos = (v_plating > 0.0)
                v_plating[cond_v_plating_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * abs(c_rate_rel_roi)**w3
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total)**2
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * (1.0 + p6 * q_loss_total**p7)
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * (q_loss_total - p7))
                # dq_d = abs(v_plating)**p4 * t_func.iloc[i_start:i_end] * p2 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                # dq_d = (abs(v_plating)**p4 * np.exp(p6 * max(q_loss_total - p7, 0.0))
                #         * np.exp(p10 * c_rate_rel_roi) * plt_base.iloc[i_start:i_end])
                dq_d = ((p3 * abs(t_internal_use.iloc[i_start:i_end] - p4_abs_t))**p5 * abs(c_rate_rel_roi)**p6
                        * np.exp(p7 * max(q_loss_total - p8, 0.0)))
                cond_not_chg = (c_rate_rel_roi < 0.0)
                dq_d[cond_v_plating_pos | cond_not_chg] = 0.0
                # why "((np.exp(p4 * abs(v_plating)) - 1.0)" -> positive feedback loop, "exponential increase of lithium
                #   plating rate": yangModelingLithiumPlating2017
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                # Although I'm not an electrochemistry expert, I think the plating model here roughly translates to
                #   equation (10) in yangModelingLithiumPlating2017:
                #   -> (phi_s - phi_e) = anode voltage?
                #   -> or (phi_s - phi_e - U_SEI)? -> maybe we need to use a lower v_anode potential
                #   -> R_film in my case is dependent on the temperature (might also have to do with the maximum Li
                #      intercalation rate??) and age of the cell (Yang: caused by SEI growth)

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f048 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.0f, p2: %.3f, p3: %.3fe-4, "
                                "p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3f, p8: %.3f, p9: %.3f, c0: %.2fe-6, c1: %.0f, c2:"
                                " %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2,
                                   p3 * 1.0e4, p4, p5, p6, p7, p8, p9,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f048" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f048 END ===================================================================================================


# === model_f049 BEGIN =================================================================================================
def model_f049(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f049"
    (s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, c0, c1, c2, c3, c4,
     f0, Vmid, Cn) = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Cn)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f049_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.2f, "
                  "w4: %.3fe-6, p0: %.3f, p1: %.5f, p2: %.3f, p3: %.3fe-4, p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3f, "
                  "p8: %.3f, p9: %.3f, p10: %.3f, p11: %.1f, p12: %.2f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, "
                  "c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2, p3*1.0e4, p4, p5, p6, p7,
                     p8, p9, p10, p11, p12, c0*1.0e6, c1, c2, c3*1.0e6, c4, f0, Vmid, Cn))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("p6", "<br>p6"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f049_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        (s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, c0, c1, c2, c3, c4,
         f0, Vmid, Cn) = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        # V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V0_SEI = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0
        T0_SEI = cfg.T0 + MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC

        df_length = log_df.shape[0]

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f049 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f049 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        # sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0)) * np.exp(s2 * (v_storage - V0))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0_SEI)) * np.exp(s2 * (v_storage - V0_SEI))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = f0 * (sei_potential_storage - s3 * q_loss_sei_storage)
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f049 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.5f, p2: %.3f, p3: %.3fe-4, "
                                "p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3f, p8: %.3f, p9: %.3f, p10: %.3f, p11: %.1f, "
                                "p12: %.2f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, "
                                "f0: %.3f, Vmid: %.3f, Cn: %.3f caused a RuntimeWarning (stor.)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2,
                                   p3 * 1.0e4, p4, p5, p6, p7, p8, p9, p10, p11, p12,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        # t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64) + p5 * c_rate_rel**2)
        t_internal_use = log_df[csv_label.T_CELL].astype(np.float64)

        # sei_potential = (
        #         s0
        #         # * np.exp(s1 * (1.0 / log_df[csv_label.T_CELL].astype(np.float64) - 1.0 / T0))
        #         * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
        #         * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        # )
        sei_potential = (  # when using final model in publication/diss -> calculate s0 to match T0/V0 instead _SEI
                s0
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0_SEI))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0_SEI))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?) -> already in cyclic wearout?

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        # p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))  # other T-dependency?
        p0t = p0 + ((p1 * abs(t_internal_use - p11))**p12)
        p4_abs_t = p4 + cfg.T0
        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # plt_base = p2 * t_func * np.exp(p11 * (1.0 / t_internal_use - 1.0 / T0))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.OCV_EST])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total)  # q_loss_sei_total / 1.0
                # diff_a = (sei_potential.iloc[i_start:i_end]
                #           - s3 * max(q_loss_sei_total - 20.0 * q_loss_plating_total, 0.0))
                # diff_a = (sei_potential.iloc[i_start:i_end] - s3 * max(q_loss_sei_total - q_loss_plating_total, 0.0))
                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p9 * q_loss_plating_total)
                diff_a = f0 * (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = f0 * (cyclic_age_potential.iloc[i_start:i_end] - w4 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = f0 * (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - p0t.iloc[i_start:i_end] * c_rate_rel_roi
                               * np.exp(p2 * q_loss_total))  # q_loss_plating_total
                # why "np.exp(p3 * q_loss_total)"? -> resistance increase (modeled different in
                # yangModelingLithiumPlating2017 -> eq. 14 + 15, where c_SEI/LI is roughly my q_loss_SEI/plating)
                cond_v_plating_pos = (v_plating > -p10)
                v_plating[cond_v_plating_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * abs(c_rate_rel_roi)**w3
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                dq_d = ((p3 * abs(t_internal_use.iloc[i_start:i_end] - p4_abs_t))**p5 * abs(c_rate_rel_roi)**p6
                        * np.exp(p7 * max(q_loss_total - p8, 0.0)))
                cond_not_chg = (c_rate_rel_roi < 0.0)
                dq_d[cond_v_plating_pos | cond_not_chg] = 0.0
                # why "((np.exp(p4 * abs(v_plating)) - 1.0)" -> positive feedback loop, "exponential increase of lithium
                #   plating rate": yangModelingLithiumPlating2017
                # since this is a rate, it is multiplied by time and not by charge -> current already in v_plating

                # Although I'm not an electrochemistry expert, I think the plating model here roughly translates to
                #   equation (10) in yangModelingLithiumPlating2017:
                #   -> (phi_s - phi_e) = anode voltage?
                #   -> or (phi_s - phi_e - U_SEI)? -> maybe we need to use a lower v_anode potential
                #   -> R_film in my case is dependent on the temperature (might also have to do with the maximum Li
                #      intercalation rate??) and age of the cell (Yang: caused by SEI growth)

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f049 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.5f, p2: %.3f, p3: %.3fe-4, "
                                "p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3f, p8: %.3f, p9: %.3f, p10: %.3f, p11: %.1f, "
                                "p12: %.2f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, "
                                "f0: %.3f, Vmid: %.3f, Cn: %.3f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2,
                                   p3 * 1.0e4, p4, p5, p6, p7, p8, p9, p10, p11, p12,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f049" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f049 END ===================================================================================================


# === model_f050 BEGIN =================================================================================================
def model_f050(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f050"
    (s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
     c0, c1, c2, c3, c4, f0, Vmid, Cn) = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Cn)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f050_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.2f, "
                  "w4: %.3fe-6, p0: %.3f, p1: %.5f, p2: %.1f, p3: %.2f, p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3fe-4, "
                  "p8: %.0f, p9: %.3f, p10: %.3f, p11: %.3f, p12: %.4f, p13: %.2f, c0: %.2fe-6, c1: %.0f, c2: %.3f, "
                  "c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2, p3, p4, p5, p6, p7*1.0e4,
                     p8, p9, p10, p11, p12, p13, c0*1.0e6, c1, c2, c3*1.0e6, c4, f0, Vmid, Cn))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("p6", "<br>p6"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f050_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        (s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
         c0, c1, c2, c3, c4, f0, Vmid, Cn) = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        # V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V0_SEI = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0
        T0_SEI = cfg.T0 + MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC

        df_length = log_df.shape[0]

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f050 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f050 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        # sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0)) * np.exp(s2 * (v_storage - V0))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0_SEI)) * np.exp(s2 * (v_storage - V0_SEI))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = f0 * (sei_potential_storage - s3 * q_loss_sei_storage)
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f050 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.5f, p2: %.1f, p3: %.2f, "
                                "p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3fe-4, p8: %.0f, p9: %.3f, p10: %.3f, p11: %.3f, "
                                "p12: %.4f, p13: %.2f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, "
                                "f0: %.3f, Vmid: %.3f, Cn: %.3f caused a RuntimeWarning (stor.)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2,
                                   p3, p4, p5, p6, p7 * 1.0e4, p8, p9, p10, p11, p12, p13,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        # t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64) + p5 * c_rate_rel**2)
        t_internal_use = log_df[csv_label.T_CELL].astype(np.float64)

        # sei_potential = (
        #         s0
        #         * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
        #         * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        # )
        sei_potential = (  # when using final model in publication/diss -> calculate s0 to match T0/V0 instead _SEI
                s0
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0_SEI))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0_SEI))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?) -> already in cyclic wearout?

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))

        # p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))  # other T-dependency?
        p0t = p0 + (p1 * abs(t_internal_use - (p2 + cfg.T0)))**p3  # base film resistance
        p7t = p7 * t_func * np.exp(p8 * (1.0 / t_internal_use - 1.0 / T0))  # base plating rate

        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p13 * q_loss_plating_total)
                diff_a = f0 * (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = f0 * (cyclic_age_potential.iloc[i_start:i_end] - w4 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = f0 * (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017?
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - p0t.iloc[i_start:i_end] * np.exp(p4 * max(q_loss_total - p5, 0)) * c_rate_rel_roi)

                cond_v_plating_pos = (v_plating > -p12)
                v_plating[cond_v_plating_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * abs(c_rate_rel_roi)**w3
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = ((p3 * abs(t_internal_use.iloc[i_start:i_end] - p4_abs_t))**p5 * abs(c_rate_rel_roi)**p6
                #         * np.exp(p7 * max(q_loss_total - p8, 0.0)))
                # cond_not_chg = (c_rate_rel_roi < 0.0)
                # dq_d[cond_v_plating_pos | cond_not_chg] = 0.0

                dq_d = (abs(v_plating) * p7t.iloc[i_start:i_end] * np.exp(p9 * max(q_loss_total - p10, 0.0))
                        * np.exp(p11 * c_rate_rel_roi))

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f050 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.5f, p2: %.1f, p3: %.2f, "
                                "p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3fe-4, p8: %.0f, p9: %.3f, p10: %.3f, p11: %.3f, "
                                "p12: %.4f, p13: %.2f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, "
                                "f0: %.3f, Vmid: %.3f, Cn: %.3f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2,
                                   p3, p4, p5, p6, p7 * 1.0e4, p8, p9, p10, p11, p12, p13,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f050" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f050 END ===================================================================================================


# === model_f051 BEGIN =================================================================================================
def model_f051(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f051"
    (s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
     c0, c1, c2, c3, c4, f0, Vmid, Cn) = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Cn)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f051_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.2f, "
                  "w4: %.3fe-6, p0: %.3f, p1: %.5f, p2: %.1f, p3: %.2f, p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3fe-4, "
                  "p8: %.0f, p9: %.3f, p10: %.3f, p11: %.3f, p12: %.3f, p13: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, "
                  "c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2, p3, p4, p5, p6, p7*1.0e4,
                     p8, p9, p10, p11, p12, p13, c0*1.0e6, c1, c2, c3*1.0e6, c4, f0, Vmid, Cn))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("p6", "<br>p6"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f051_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        (s0, s1, s2, s3, w0, w1, w2, w3, w4, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
         c0, c1, c2, c3, c4, f0, Vmid, Cn) = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        # V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V0_SEI = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0
        T0_SEI = cfg.T0 + MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC

        df_length = log_df.shape[0]

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f051 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f051 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        # sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0)) * np.exp(s2 * (v_storage - V0))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0_SEI)) * np.exp(s2 * (v_storage - V0_SEI))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = f0 * (sei_potential_storage - s3 * q_loss_sei_storage)
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f051 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.5f, p2: %.1f, p3: %.2f, "
                                "p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3fe-4, p8: %.0f, p9: %.3f, p10: %.3f, p11: %.3f, "
                                "p12: %.3f, p13: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, "
                                "f0: %.3f, Vmid: %.3f, Cn: %.3f caused a RuntimeWarning (stor.)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2,
                                   p3, p4, p5, p6, p7 * 1.0e4, p8, p9, p10, p11, p12, p13,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        # t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64) + p5 * c_rate_rel**2)
        t_internal_use = log_df[csv_label.T_CELL].astype(np.float64)

        # sei_potential = (
        #         s0
        #         * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
        #         * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        # )
        sei_potential = (  # when using final model in publication/diss -> calculate s0 to match T0/V0 instead _SEI
                s0
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0_SEI))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0_SEI))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?) -> already in cyclic wearout?

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))

        # p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))  # other T-dependency?
        d_temperature = (p2 + cfg.T0) - t_internal_use
        d_temperature[d_temperature < 0.0] = 0.0
        p0t = p0 + (p1 * d_temperature)**p3  # base film resistance
        p7t = p7 * t_func  # base plating rate

        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p13 * q_loss_plating_total)
                diff_a = f0 * (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = f0 * (cyclic_age_potential.iloc[i_start:i_end] - w4 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = f0 * (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017?
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - p0t.iloc[i_start:i_end] * np.exp(p4 * max(q_loss_total - p5, 0)) * c_rate_rel_roi)

                cond_v_plating_pos = (v_plating > -p12)
                v_plating[cond_v_plating_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * abs(c_rate_rel_roi)**w3
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = ((p3 * abs(t_internal_use.iloc[i_start:i_end] - p4_abs_t))**p5 * abs(c_rate_rel_roi)**p6
                #         * np.exp(p7 * max(q_loss_total - p8, 0.0)))
                # cond_not_chg = (c_rate_rel_roi < 0.0)
                # dq_d[cond_v_plating_pos | cond_not_chg] = 0.0

                dq_d = (abs(v_plating) * p7t.iloc[i_start:i_end] * np.exp(p10 * max(q_loss_total - p11, 0.0))
                        * np.exp(p8 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0) * c_rate_rel_roi**p9))

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f051 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3fe-6, p0: %.3f, p1: %.5f, p2: %.1f, p3: %.2f, "
                                "p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3fe-4, p8: %.0f, p9: %.3f, p10: %.3f, p11: %.3f, "
                                "p12: %.3f, p13: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, c4: %.3f, "
                                "f0: %.3f, Vmid: %.3f, Cn: %.3f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4*1.0e6, p0, p1, p2,
                                   p3, p4, p5, p6, p7 * 1.0e4, p8, p9, p10, p11, p12, p13,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f051" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f051 END ===================================================================================================


# === model_f052 BEGIN =================================================================================================
def model_f052(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f052"
    (s0, s1, s2, s3, w0, w1, w2, w3, w4, w5, w6, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
     c0, c1, c2, c3, c4, f0, Vmid, Cn) = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Cn)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f052_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3f, "
                  "w5: %.3f, w6: %.3fe-6, p0: %.3f, p1: %.5f, p2: %.1f, p3: %.2f, p4: %.3f, p5: %.3f, p6: %.3f, "
                  "p7: %.3fe-4, p8: %.0f, p9: %.3f, p10: %.3f, p11: %.3f, p12: %.3f, p13: %.4f, c0: %.2fe-6, c1: %.0f, "
                  "c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3, w4, w5, w6*1.0e6, p0, p1, p2, p3, p4, p5, p6,
                     p7*1.0e4, p8, p9, p10, p11, p12, p13, c0*1.0e6, c1, c2, c3*1.0e6, c4, f0, Vmid, Cn))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("p5", "<br>p5"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f052_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        (s0, s1, s2, s3, w0, w1, w2, w3, w4, w5, w6, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
         c0, c1, c2, c3, c4, f0, Vmid, Cn) = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        # V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V0_SEI = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0
        T0_SEI = cfg.T0 + MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC

        df_length = log_df.shape[0]

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f052 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f052 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        # sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0)) * np.exp(s2 * (v_storage - V0))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0_SEI)) * np.exp(s2 * (v_storage - V0_SEI))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = f0 * (sei_potential_storage - s3 * q_loss_sei_storage)
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f052 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3f, w5: %.3f, w6: %.3fe-6, p0: %.3f, p1: %.5f, "
                                "p2: %.1f, p3: %.2f, p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3fe-4, p8: %.0f, p9: %.3f, "
                                "p10: %.3f, p11: %.3f, p12: %.3f, p13: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, "
                                "c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f caused a RuntimeWarning (stor.)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4, w5, w6*1.0e6,
                                   p0, p1, p2, p3, p4, p5, p6, p7 * 1.0e4, p8, p9, p10, p11, p12, p13,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        # t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64) + p5 * c_rate_rel**2)
        t_internal_use = log_df[csv_label.T_CELL].astype(np.float64)

        # sei_potential = (
        #         s0
        #         * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
        #         * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        # )
        sei_potential = (  # when using final model in publication/diss -> calculate s0 to match T0/V0 instead _SEI
                s0
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0_SEI))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0_SEI))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?) -> already in cyclic wearout?

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))

        # p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))  # other T-dependency?
        d_temperature = (p2 + cfg.T0) - t_internal_use
        d_temperature[d_temperature < 0.0] = 0.0
        p0t = p0 + (p1 * d_temperature)**p3  # base film resistance
        p7t = p7 * t_func  # base plating rate

        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p13 * q_loss_plating_total)
                diff_a = f0 * (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = f0 * (cyclic_age_potential.iloc[i_start:i_end] * np.exp(w4 * max(q_loss_total - w5, 0))
                               - w6 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = f0 * (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017?
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - p0t.iloc[i_start:i_end] * np.exp(p4 * max(q_loss_total - p5, 0)) * c_rate_rel_roi)

                cond_v_plating_pos = (v_plating > -p12)
                v_plating[cond_v_plating_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * abs(c_rate_rel_roi)**w3
                dq_c = diff_c * t_func.iloc[i_start:i_end]
                # dq_d = ((p3 * abs(t_internal_use.iloc[i_start:i_end] - p4_abs_t))**p5 * abs(c_rate_rel_roi)**p6
                #         * np.exp(p7 * max(q_loss_total - p8, 0.0)))
                # cond_not_chg = (c_rate_rel_roi < 0.0)
                # dq_d[cond_v_plating_pos | cond_not_chg] = 0.0

                dq_d = (abs(v_plating) * p7t.iloc[i_start:i_end] * np.exp(p10 * max(q_loss_total - p11, 0.0))
                        * np.exp(p8 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0) * c_rate_rel_roi**p9))

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f052 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3f, w5: %.3f, w6: %.3fe-6, p0: %.3f, p1: %.5f,  "
                                "p2: %.1f, p3: %.2f,p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3fe-4, p8: %.0f, p9: %.3f, "
                                "p10: %.3f, p11: %.3f, p12: %.3f, p13: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, "
                                "c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4, w5, w6*1.0e6, p0, p1, p2,
                                   p3, p4, p5, p6, p7 * 1.0e4, p8, p9, p10, p11, p12, p13,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f052" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f052 END ===================================================================================================


# === model_f053 BEGIN =================================================================================================
def model_f053(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f053"
    (s0, s1, s2, s3, w0, w1, w2, w3, w4, w5, w6, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
     c0, c1, c2, c3, c4, f0, Vmid, Cn) = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Cn)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f053_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3f, "
                  "w5: %.3f, w6: %.3fe-6, p0: %.3f, p1: %.5f, p2: %.1f, p3: %.2f, p4: %.3f, p5: %.3f, p6: %.3f, "
                  "p7: %.3fe-4, p8: %.0f, p9: %.3f, p10: %.3f, p11: %.3f, p12: %.3f, p13: %.4f, c0: %.2fe-6, c1: %.0f, "
                  "c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3, w4, w5, w6*1.0e6, p0, p1, p2, p3, p4, p5, p6,
                     p7*1.0e4, p8, p9, p10, p11, p12, p13, c0*1.0e6, c1, c2, c3*1.0e6, c4, f0, Vmid, Cn))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("p5", "<br>p5"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f053_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        # if (task_queue is None) or task_queue.empty():
        #     time.sleep(5)  # thread exiting before queue is ready -> wait
        #     if (task_queue is None) or task_queue.empty():
        #         break  # no more files

        try:
            remaining_size = task_queue.qsize()
            # queue_entry = task_queue.get_nowait()
            queue_entry = task_queue.get(block=False)
            # except multiprocessing.queue.Empty:
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        # pid_index = queue_entry["pid_index"]
        (s0, s1, s2, s3, w0, w1, w2, w3, w4, w5, w6, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
         c0, c1, c2, c3, c4, f0, Vmid, Cn) = queue_entry["vars"]

        # if pid_index == 29:
        #     print("debug")

        # calculate delta of aged_cap f_age(...)
        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        # V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V0_SEI = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0
        T0_SEI = cfg.T0 + MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC

        df_length = log_df.shape[0]

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f053 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f053 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        # sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0)) * np.exp(s2 * (v_storage - V0))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0_SEI)) * np.exp(s2 * (v_storage - V0_SEI))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = f0 * (sei_potential_storage - s3 * q_loss_sei_storage)
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f053 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3f, w5: %.3f, w6: %.3fe-6, p0: %.3f, p1: %.5f, "
                                "p2: %.1f, p3: %.2f, p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3fe-4, p8: %.0f, p9: %.3f, "
                                "p10: %.3f, p11: %.3f, p12: %.3f, p13: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, "
                                "c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f caused a RuntimeWarning (stor.)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4, w5, w6*1.0e6,
                                   p0, p1, p2, p3, p4, p5, p6, p7 * 1.0e4, p8, p9, p10, p11, p12, p13,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # cond_chg -> if current >1/1000 C. For an 80 kWh battery: 80 W
        cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)
        # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        # t_internal_use = (log_df[csv_label.T_CELL].astype(np.float64) + p5 * c_rate_rel**2)
        t_internal_use = log_df[csv_label.T_CELL].astype(np.float64)

        # sei_potential = (
        #         s0
        #         * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
        #         * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        # )
        sei_potential = (  # when using final model in publication/diss -> calculate s0 to match T0/V0 instead _SEI
                s0
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0_SEI))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0_SEI))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        # corrosion_potential = (c0 * t_func  # c0 * dQ_chg
        #                        * np.exp(-c1 * (log_df[csv_label.V_CELL].astype(np.float64) - V0)))
        corrosion_potential = (c0  # c2 ~ internal resistance, c_rate_rel < 0 when discharging
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64) + c2 * c_rate_rel))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?) -> already in cyclic wearout?

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))

        # p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))  # other T-dependency?
        d_temperature = (p2 + cfg.T0) - t_internal_use
        d_temperature[d_temperature < 0.0] = 0.0
        p0t = p0 + (p1 * d_temperature)**p3  # base film resistance
        p7t = p7 * t_func  # base plating rate

        try:
            anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p13 * q_loss_plating_total)
                diff_a = f0 * (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = f0 * (cyclic_age_potential.iloc[i_start:i_end] * np.exp(w4 * max(q_loss_total - w5, 0))
                               - w6 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = f0 * (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017?
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - p0t.iloc[i_start:i_end] * np.exp(p4 * max(q_loss_total - p5, 0)) * c_rate_rel_roi)

                cond_v_plating_pos = (v_plating > -p12)
                v_plating[cond_v_plating_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * abs(c_rate_rel_roi)**w3
                dq_c = diff_c * t_func.iloc[i_start:i_end] * abs(c_rate_rel_roi)
                # dq_d = ((p3 * abs(t_internal_use.iloc[i_start:i_end] - p4_abs_t))**p5 * abs(c_rate_rel_roi)**p6
                #         * np.exp(p7 * max(q_loss_total - p8, 0.0)))
                # cond_not_chg = (c_rate_rel_roi < 0.0)
                # dq_d[cond_v_plating_pos | cond_not_chg] = 0.0

                dq_d = (abs(v_plating) * p7t.iloc[i_start:i_end] * np.exp(p10 * max(q_loss_total - p11, 0.0))
                        * np.exp(p8 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0) * c_rate_rel_roi**p9))

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f053 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3f, w5: %.3f, w6: %.3fe-6, p0: %.3f, p1: %.5f,  "
                                "p2: %.1f, p3: %.2f,p4: %.3f, p5: %.3f, p6: %.3f, p7: %.3fe-4, p8: %.0f, p9: %.3f, "
                                "p10: %.3f, p11: %.3f, p12: %.3f, p13: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, "
                                "c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4, w5, w6*1.0e6, p0, p1, p2,
                                   p3, p4, p5, p6, p7 * 1.0e4, p8, p9, p10, p11, p12, p13,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f053" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f053 END ===================================================================================================


# === model_f054 BEGIN =================================================================================================
def model_f054(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f054"
    (s0, s1, s2, s3, w0, w1, w2, w3, w4, w5, w6, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12,
     c0, c1, c2, c3, c4, f0, Vmid, Cn) = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Cn)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f054_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = rmse_param[pid_index] + delta_rmse_par
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = rmse_total + delta_rmse_tot
        num_rmse_points_total = num_rmse_points_total + delta_num_rmse_points_tot

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3f, "
                  "w5: %.3f, w6: %.3fe-6, p0: %.3f, p1: %.5f, p2: %.1f, p3: %.2f, p4: %.3f, p5: %.3f, p6: %.3fe-8, "
                  "p7: %.0f, p8: %.3f, p9: %.3f, p10: %.3f, p11: %.3f, p12: %.4f, c0: %.2fe-6, c1: %.0f, "
                  "c2: %.3f, c3: %.3fe-6, c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3, w4, w5, w6*1.0e6, p0, p1, p2, p3, p4, p5,
                     p6 * 1.0e8, p7, p8, p9, p10, p11, p12, c0*1.0e6, c1, c2, c3*1.0e6, c4, f0, Vmid, Cn))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("p5", "<br>p5"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f054_get_delta_age(task_queue, result_queue):
    time.sleep(2)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        try:
            remaining_size = task_queue.qsize()
            queue_entry = task_queue.get(block=False)
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        C_0 = queue_entry["C_0"]
        (s0, s1, s2, s3, w0, w1, w2, w3, w4, w5, w6, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12,
         c0, c1, c2, c3, c4, f0, Vmid, Cn) = queue_entry["vars"]

        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        # V0 = 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        V0_SEI = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        V_corr_ths = c4  # 3.2
        V_mid = Vmid  # 3.73  # 3.8 # 3.73  # 3.65  # 3.6  # 3.73  # 3.6
        T0 = cfg.T0 + 25.0
        T0_SEI = cfg.T0 + MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC

        df_length = log_df.shape[0]

        # How to determine ΔQ_loss_SEI = f0 * (SEI_potential - a_i * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f054 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f054 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        # sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0)) * np.exp(s2 * (v_storage - V0))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T0_SEI)) * np.exp(s2 * (v_storage - V0_SEI))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = f0 * (sei_potential_storage - s3 * q_loss_sei_storage)
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f054 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3f, w5: %.3f, w6: %.3fe-6, p0: %.3f, p1: %.5f, "
                                "p2: %.1f, p3: %.2f, p4: %.3f, p5: %.3f, p6: %.3fe-8, p7: %.0f, p8: %.3f, p9: %.3f, "
                                "p10: %.3f, p11: %.3f, p12: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, "
                                "c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f caused a RuntimeWarning (stor.)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4, w5, w6*1.0e6,
                                   p0, p1, p2, p3, p4, p5, p6 * 1.0e8, p7, p8, p9, p10, p11, p12,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        # only consider current if it is >1/500 C. For an 80 kWh battery: 160 W --> remove noise if I = ca. 0
        c_ths = cfg.CELL_CAPACITY_NOMINAL / 500.0
        cond_chg = (log_df[csv_label.I_CELL] > c_ths)
        cond_dischg = (log_df[csv_label.I_CELL] < -c_ths)

        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        c_rate_rel.loc[cond_chg | cond_dischg] = c_rate_rel

        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = log_df[csv_label.T_CELL].astype(np.float64)

        # sei_potential = (
        #         s0
        #         * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0))
        #         * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
        # )
        sei_potential = (  # when using final model in publication/diss -> calculate s0 to match T0/V0 instead _SEI
                s0
                * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T0_SEI))
                * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0_SEI))
        )
        cyclic_age_potential = (w0
                                * np.exp(w1 * (1.0 / t_internal_use - 1.0 / T0))
                                * (log_df[csv_label.V_CELL].astype(np.float64) - V_mid).abs() ** w2)
        corrosion_potential = (c0
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T0))
                               * (V_corr_ths - log_df[csv_label.V_CELL].astype(np.float64)))
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        # FIXME: consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?) -> already in cyclic wearout?

        # dq_cyclic_age_wearout = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_wearout = dq_cyclic_age_wearout.cumsum()

        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_cumsum = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        # dq_loss_plating = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))

        # p0t = p0 * np.exp(p1 * (1.0 / t_internal_use - 1.0 / T0))  # other T-dependency?
        d_temperature = (p2 + cfg.T0) - t_internal_use
        d_temperature[d_temperature < 0.0] = 0.0
        p0t = p0 + (p1 * d_temperature)**p3  # base film resistance
        p6t = p6 * t_func  # base plating rate

        try:
            # anode_potential = get_v_anode_from_v_terminal_df_v2(log_df[csv_label.V_CELL])
            # anode_potential = get_v_anode_from_v_terminal_df_v3(log_df[csv_label.V_CELL])
            anode_potential = get_v_anode_from_v_terminal_df_v4(log_df[csv_label.V_CELL])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = c_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)
                chg_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                sei_potential_use = sei_potential.iloc[i_start:i_end] * (1.0 + p12 * q_loss_plating_total)
                diff_a = f0 * (sei_potential_use - s3 * q_loss_sei_total)
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = f0 * (cyclic_age_potential.iloc[i_start:i_end] * np.exp(w4 * max(q_loss_total - w5, 0))
                               - w6 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = f0 * (corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total)
                diff_c[diff_c < 0] = 0.0
                # v_plating -> translates to the "lithium deposition potential (LDP)" in yangModelingLithiumPlating2017?
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - p0t.iloc[i_start:i_end] * np.exp(p4 * max(q_loss_total - p5, 0)) * chg_rate_rel_roi)

                cond_v_plating_pos = (v_plating > -p11)
                v_plating[cond_v_plating_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * abs(c_rate_rel_roi)**w3
                dq_c = diff_c * t_func.iloc[i_start:i_end] * abs(c_rate_rel_roi)**c2
                # dq_d = (abs(v_plating) * p6t.iloc[i_start:i_end] * np.exp(p9 * max(q_loss_total - p10, 0.0))
                #         * np.exp(p7 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0)
                #                  * abs(chg_rate_rel_roi)**p8))
                dq_d = (abs(v_plating) * p6t.iloc[i_start:i_end] * np.exp(p9 * max(q_loss_total - p10, 0.0))
                        * np.exp(p7 * (1.0 / t_internal_use.iloc[i_start:i_end] - 1.0 / T0))
                        * abs(chg_rate_rel_roi)**p8)

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f054 (???) call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, "
                                "w1: %.0f, w2: %.2f, w3: %.2f, w4: %.3f, w5: %.3f, w6: %.3fe-6, p0: %.3f, p1: %.5f,  "
                                "p2: %.1f, p3: %.2f,p4: %.3f, p5: %.3f, p6: %.3fe-8, p7: %.0f, p8: %.3f, p9: %.3f, "
                                "p10: %.3f, p11: %.3f, p12: %.4f, c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.3fe-6, "
                                "c4: %.3f, f0: %.3f, Vmid: %.3f, Cn: %.3f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3, w4, w5, w6*1.0e6, p0, p1, p2,
                                   p3, p4, p5, p6 * 1.0e8, p7, p8, p9, p10, p11, p12,
                                   c0 * 1.0e6, c1, c2, c3 * 1.0e6, c4, f0, Vmid, Cn))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f054" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f054 END ===================================================================================================


# === model_f055 BEGIN =================================================================================================
def model_f055(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f055"
    (s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, c0, c1, c2, c3, f0, Vm, Ci) = variable_list
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # === error band modeling ==========================================================================================
    if show_error_range and plot:
        # start modeling with +/- error_delta of the initial capacity to see error band of model
        # Append jobs
        result_manager_2 = multiprocessing.Manager()
        result_queue_2 = result_manager_2.Queue()
        C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
        model_min = [pd.Series(dtype=np.float64) for _ in range(0, n_pids)]
        model_max = [pd.Series(dtype=np.float64) for _ in range(0, n_pids)]
        for i_Ci in range(2):
            if i_Ci == 0:
                Ci_use = Ci * (1.0 - error_delta)
            else:
                Ci_use = Ci * (1.0 + error_delta)
            for pid_index in range(0, n_pids):
                for pnr_index in range(0, n_pnrs):
                    log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
                    log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                                 mdl_cap_delta_row, mdl_cap_row, Ci_use)
                    C_0[pid_index][pnr_index] = this_C_0
                    if skip:
                        continue
                    model_entry = {"log_df": log_df, "vars": variable_list,
                                   "pid_index": pid_index, "pnr_index": pnr_index, "C_0": this_C_0}
                    modeling_task_queue.put(model_entry)

        processes = []
        logging.log.info("Starting error band threads...")
        for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
            # === USER CODE 2 BEGIN ====================================================================================
            processes.append(multiprocessing.Process(target=model_f055_get_delta_age,
                                                     args=(modeling_task_queue, result_queue_2)))
            # === USER CODE 2 END ======================================================================================
        for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
            processes[processorNumber].start()
        for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
            processes[processorNumber].join()

        logging.log.debug("calculating and plotting error band ...")
        while True:
            if (result_queue_2 is None) or result_queue_2.empty():
                break  # no more reports
            try:
                this_result = result_queue_2.get_nowait()
            except multiprocessing.queues.Empty:
                break  # no more reports

            log_df = this_result["log_df"]
            pid_index, pnr_index, _, _, _, _, _ = this_result["others"]

            # convert timestamp to datetime to apply stuff
            log_df_roi = log_df[[csv_label.TIMESTAMP, mdl_cap_row]].copy()
            log_df_roi.loc[:, csv_label.TIMESTAMP] = pd.to_datetime(log_df_roi[csv_label.TIMESTAMP],
                                                                    unit="s", origin='unix')
            log_df_roi_rs = log_df_roi.set_index(csv_label.TIMESTAMP).resample('1D')
            # resample with time_resolution (use min/max)
            this_min = log_df_roi_rs.min().copy()
            this_max = log_df_roi_rs.max().copy()
            # this_min[csv_label.TIMESTAMP] = (this_min.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
            # this_max[csv_label.TIMESTAMP] = (this_max.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
            # this_min.reset_index(drop=True, inplace=True)
            # this_max.reset_index(drop=True, inplace=True)
            this_min.index = ((this_min.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")) / TIME_DIV
            this_max.index = this_min.index

            # if model_min empty -> use current as min, else find min
            if model_min[pid_index].shape[0] == 0:
                model_min[pid_index] = this_min
            else:
                model_group = pd.concat([model_min[pid_index], this_min]).groupby(level=0)
                model_min[pid_index] = model_group.min()
            # if model_max empty -> use current as max, else find max
            if model_max[pid_index].shape[0] == 0:
                model_max[pid_index] = this_max
            else:
                model_group = pd.concat([model_max[pid_index], this_max]).groupby(level=0)
                model_max[pid_index] = model_group.max()
        add_model_errorbands(mdl_cap_row, model_min, model_max, n_pids, fig_list, fig_and_sp_from_pid_arr)

    # === actual modeling ==============================================================================================
    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Ci)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting main threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f055_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = [rmse_param[pid_index][0] + delta_rmse_par[0],
                                 rmse_param[pid_index][1] + delta_rmse_par[1]]
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = [rmse_total[0] + delta_rmse_tot[0], rmse_total[1] + delta_rmse_tot[1]]
        num_rmse_points_total = [num_rmse_points_total[0] + delta_num_rmse_points_tot[0],
                                 num_rmse_points_total[1] + delta_num_rmse_points_tot[1]]

        if plot:
            add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                            fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9, w0: %.3fe-6, w1: %.3f, w2: %.3f, w3: %.3fe-6, "
                  "p0: %.4f, p1: %.4f, p2: %.1f, p3: %.3f, p4: %.3f, p5: %.3fe-9, p6: %.2f, "
                  "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.1fe-6, f0: %.3f, Vm: %.3f, Ci: %.4f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2, p3, p4, p5*1.0e9, p6,
                     c0*1.0e6, c1, c2, c3*1.0e6, f0, Vm, Ci))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("p3", "<br>p3"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f055_get_delta_age(task_queue, result_queue):
    time.sleep(1)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        try:
            remaining_size = task_queue.qsize()
            queue_entry = task_queue.get(block=False)
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        df_length = log_df.shape[0]
        C_0 = queue_entry["C_0"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, c0, c1, c2, c3, f0, Vm, Ci = queue_entry["vars"]

        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V_ref = Vm  # 3.73
        T_ref = cfg.T0 + 25.0

        # How to determine ΔQ_loss_SEI = f0 * (SEI_potential - a_i * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f055 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f055 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T_ref)) * np.exp(s2 * (v_storage - V_ref))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = sei_potential_storage - s3 * q_loss_sei_storage
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + f0 * diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f055 call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9,"
                                "w0: %.3fe-6, w1: %.3f, w2: %.3f, w3: %.3fe-6, "
                                "p0: %.4f, p1: %.4f, p2: %.1f, p3: %.3f, p4: %.3f, p5: %.3fe-9, p6: %.2f, "
                                "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.1fe-6, "
                                "f0: %.3f, Vm: %.3f, Ci: %.4f caused a RuntimeWarning (storage)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6,
                                   p0, p1, p2, p3, p4, p5 * 1.0e9, p6, c0 * 1.0e6, c1, c2, c3 * 1.0e6, f0, Vm, Ci))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        # only consider current if it is >1/500 C. For an 80 kWh battery: 160 W --> remove noise if I = ca. 0
        c_ths = cfg.CELL_CAPACITY_NOMINAL / 500.0
        cond_chg = (log_df[csv_label.I_CELL] > c_ths)
        cond_dischg = (log_df[csv_label.I_CELL] < -c_ths)

        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        c_rate_rel.loc[cond_chg | cond_dischg] = c_rate_rel

        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = log_df[csv_label.T_CELL].astype(np.float64)

        sei_potential = (s0
                         * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T_ref))
                         * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V_ref)))
        cyclic_age_potential = w0 * (log_df[csv_label.V_CELL].astype(np.float64) - V_ref).abs()
        corrosion_potential = (c0
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T_ref))
                               * (c2 - log_df[csv_label.V_CELL].astype(np.float64)))  # c2 = V_corr_ths
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))

        d_temperature = (p2 + cfg.T0) - t_internal_use
        d_temperature[d_temperature < 0.0] = 0.0
        p0t = p0 + (p1 * d_temperature)**p3  # base film resistance
        p5t = p5 * t_func  # base plating rate

        try:
            anode_potential = get_v_anode_from_v_terminal_df_v4(log_df[csv_label.V_CELL])

            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = c_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)
                chg_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                diff_a = sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] * np.exp(w1 * max(q_loss_total - w2, 0))
                          - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total
                diff_c[diff_c < 0] = 0.0
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - p0t.iloc[i_start:i_end] * np.exp(p4 * q_loss_total) * chg_rate_rel_roi)

                cond_v_plating_pos = (v_plating > 0.0)
                v_plating[cond_v_plating_pos] = 0.0

                dq_a = f0 * diff_a * t_func.iloc[i_start:i_end]
                dq_b = f0 * diff_b * t_func.iloc[i_start:i_end] * abs(c_rate_rel_roi)
                dq_c = f0 * diff_c * t_func.iloc[i_start:i_end] * abs(c_rate_rel_roi)
                dq_d = abs(v_plating) * p5t.iloc[i_start:i_end] * abs(chg_rate_rel_roi)**p6

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE
        except RuntimeWarning:
            logging.log.warning("model_f055 call with s0: %.3fe-9, s1: %.0f, s2: %.2f, s3: %.3fe-9,"
                                "w0: %.3fe-6, w1: %.3f, w2: %.3f, w3: %.3fe-6, "
                                "p0: %.4f, p1: %.4f, p2: %.1f, p3: %.3f, p4: %.3f, p5: %.3fe-9, p6: %.2f, "
                                "c0: %.2fe-6, c1: %.0f, c2: %.3f, c3: %.1fe-6, "
                                "f0: %.3f, Vm: %.3f, Ci: %.4f caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6,
                                   p0, p1, p2, p3, p4, p5 * 1.0e9, p6, c0 * 1.0e6, c1, c2, c3 * 1.0e6, f0, Vm, Ci))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f055" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row
        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f055 END ===================================================================================================


# === model_f056 BEGIN =================================================================================================
def model_f056(variable_list, log_age_dfs, mdl_id, use_cap_nom, use_t_prod, plot, fig_list, fig_and_sp_from_pid_arr):
    # === USER CODE 1 BEGIN ============================================================================================
    mdl_name = "model_f056"
    (s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, c0, c1, c2, c3, Vm, Ci) = variable_list
    error_delta_list = np.array(model_f056_error_var_dependency) * error_delta_param
    variable_list_high = list(np.array(variable_list) * (1.0 + error_delta_list))
    variable_list_low = list(np.array(variable_list) * (1.0 - error_delta_list))
    variable_list_high[-1] = Ci * (1.0 - error_delta)  # not sure if this is really needed
    variable_list_low[-1] = Ci * (1.0 + error_delta)  # lower aging - start with higher capacity
    # === USER CODE 1 END ==============================================================================================

    mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip = prepare_model(mdl_name, log_age_dfs)
    if skip:
        return 0  # error -> empty log_age_dfs

    # === error band modeling ==========================================================================================
    if show_error_range and plot:
        # start modeling with +/- error_delta of the initial capacity to see error band of model
        # Append jobs
        result_manager_2 = multiprocessing.Manager()
        result_queue_2 = result_manager_2.Queue()
        C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
        model_min = [pd.Series(dtype=np.float64) for _ in range(0, n_pids)]
        model_max = [pd.Series(dtype=np.float64) for _ in range(0, n_pids)]
        for i_Ci in range(2):
            if i_Ci == 0:
                Ci_use = Ci * (1.0 - error_delta)  # start with lower capacity
                variable_list_use = variable_list_high  # more aging
            else:
                Ci_use = Ci * (1.0 + error_delta)  # start with higher capacity
                variable_list_use = variable_list_low  # less aging
            for pid_index in range(0, n_pids):
                for pnr_index in range(0, n_pnrs):
                    log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
                    log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                                 mdl_cap_delta_row, mdl_cap_row, Ci_use)
                    C_0[pid_index][pnr_index] = this_C_0
                    if skip:
                        continue
                    model_entry = {"log_df": log_df, "vars": variable_list_use,
                                   "pid_index": pid_index, "pnr_index": pnr_index, "C_0": this_C_0}
                    modeling_task_queue.put(model_entry)

        processes = []
        logging.log.info("Starting error band threads...")
        for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
            # === USER CODE 2 BEGIN ====================================================================================
            processes.append(multiprocessing.Process(target=model_f056_get_delta_age,
                                                     args=(modeling_task_queue, result_queue_2)))
            # === USER CODE 2 END ======================================================================================
        for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
            processes[processorNumber].start()
        for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
            processes[processorNumber].join()

        logging.log.debug("calculating and plotting error band ...")
        while True:
            if (result_queue_2 is None) or result_queue_2.empty():
                break  # no more reports
            try:
                this_result = result_queue_2.get_nowait()
            except multiprocessing.queues.Empty:
                break  # no more reports

            log_df = this_result["log_df"]
            pid_index, pnr_index, _, _, _, _, _ = this_result["others"]

            # convert timestamp to datetime to apply stuff
            log_df_roi = log_df[[csv_label.TIMESTAMP, mdl_cap_row]].copy()
            log_df_roi.loc[:, csv_label.TIMESTAMP] = pd.to_datetime(log_df_roi[csv_label.TIMESTAMP],
                                                                    unit="s", origin='unix')
            # log_df_roi_rs = log_df_roi.set_index(csv_label.TIMESTAMP).resample('1D')
            # # resample with time_resolution (use min/max)
            # this_min = log_df_roi_rs.min().copy().dropna()
            # this_max = log_df_roi_rs.max().copy().dropna()
            # # this_min[csv_label.TIMESTAMP] = (this_min.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
            # # this_max[csv_label.TIMESTAMP] = (this_max.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
            # # this_min.reset_index(drop=True, inplace=True)
            # # this_max.reset_index(drop=True, inplace=True)
            # this_min.index = ((this_min.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")) / TIME_DIV
            # this_max.index = this_min.index
            #
            # # if model_min empty -> use current as min, else find min
            # if model_min[pid_index].shape[0] == 0:
            #     model_min[pid_index] = this_min
            # else:
            #     model_group = pd.concat([model_min[pid_index], this_min]).groupby(level=0)
            #     model_min[pid_index] = model_group.min()
            # # if model_max empty -> use current as max, else find max
            # if model_max[pid_index].shape[0] == 0:
            #     model_max[pid_index] = this_max
            # else:
            #     model_group = pd.concat([model_max[pid_index], this_max]).groupby(level=0)
            #     model_max[pid_index] = model_group.max()

            log_df_roi_rs = log_df_roi.set_index(csv_label.TIMESTAMP)
            log_df_roi_rs = log_df_roi_rs.resample('1D').interpolate("linear").ffill().bfill()
            log_df_roi_rs.index = ((log_df_roi_rs.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")) / TIME_DIV

            # if model_min empty -> use current as min, else find min of the two for each day
            if model_min[pid_index].shape[0] == 0:
                model_min[pid_index] = log_df_roi_rs
            else:
                model_group = pd.concat([model_min[pid_index], log_df_roi_rs]).groupby(level=0)
                model_min[pid_index] = model_group.min()
            # if model_max empty -> use current as max, else find max of the two for each day
            if model_max[pid_index].shape[0] == 0:
                model_max[pid_index] = log_df_roi_rs
            else:
                model_group = pd.concat([model_max[pid_index], log_df_roi_rs]).groupby(level=0)
                model_max[pid_index] = model_group.max()
        for pid_index in range(0, n_pids):
            model_min[pid_index] = model_min[pid_index].fillna(0.0)
            model_max[pid_index] = model_max[pid_index].fillna(0.0)
        add_model_errorbands(mdl_cap_row, model_min, model_max, n_pids, fig_list, fig_and_sp_from_pid_arr)

    # === actual modeling ==============================================================================================
    # Append jobs
    result_manager = multiprocessing.Manager()
    result_queue = result_manager.Queue()
    C_0 = [[np.nan for _ in range(0, n_pnrs)] for _ in range(0, n_pids)]
    for pid_index in range(0, n_pids):
        for pnr_index in range(0, n_pnrs):
            log_df: pd.DataFrame = log_age_dfs[pid_index][pnr_index]
            log_df, this_C_0, skip = prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod,
                                                         mdl_cap_delta_row, mdl_cap_row, Ci)
            C_0[pid_index][pnr_index] = this_C_0
            if skip:
                continue
            # [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
            # this_fig = fig_list[i_fig]
            model_entry = {"log_df": log_df, "vars": variable_list, "pid_index": pid_index, "pnr_index": pnr_index,
                           "C_0": this_C_0}  # , "plot": plot, "this_fig": this_fig, "i_row": i_row, "i_col": i_col
            modeling_task_queue.put(model_entry)

    # time.sleep(3)  # thread exiting before queue is ready -> wait
    processes = []
    logging.log.info("Starting main threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f056_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [[0.0, 0.0] for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = [0.0, 0.0], [0.0, 0.0]
    while True:
        if (result_queue is None) or result_queue.empty():
            break  # no more reports
        try:
            this_result = result_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        log_df = this_result["log_df"]
        # result = this_result["result"]
        # this_fig = this_result["this_fig"]
        pid_index, pnr_index, delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell \
            = this_result["others"]
        # [i_fig, _, _] = fig_and_sp_from_pid_arr[pid_index]
        # fig_list[i_fig] = this_fig
        rmse_param[pid_index] = [rmse_param[pid_index][0] + delta_rmse_par[0],
                                 rmse_param[pid_index][1] + delta_rmse_par[1]]
        num_rmse_points_param[pid_index] = num_rmse_points_param[pid_index] + delta_num_rmse_par
        rmse_total = [rmse_total[0] + delta_rmse_tot[0], rmse_total[1] + delta_rmse_tot[1]]
        num_rmse_points_total = [num_rmse_points_total[0] + delta_num_rmse_points_tot[0],
                                 num_rmse_points_total[1] + delta_num_rmse_points_tot[1]]

        if plot:
            if PLOT_AGING_TYPE and ("q_loss_type_df" in this_result):
                this_C_0 = C_0[pid_index][pnr_index]
                q_loss_type_df = this_result.get("q_loss_type_df")
                add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                fig_list, fig_and_sp_from_pid_arr, rmse_cell, C_0=this_C_0, q_loss_type=q_loss_type_df)
            else:
                add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index,
                                fig_list, fig_and_sp_from_pid_arr, rmse_cell)
    logging.log.debug("calculating RMSEs and plotting (2/2) ...")
    for pid_index in range(0, n_pids):
        if pid_index not in CELL_PARAMETERS_USE:
            continue
        this_rmse_p = rmse_param[pid_index]
        this_num_rmse_points_p = num_rmse_points_param[pid_index]
        rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p = calc_rmse_parameter(
            rmse_total, num_rmse_points_total, this_rmse_p, this_num_rmse_points_p)
        # rmse_param[pid_index] = this_rmse_p
        num_rmse_points_param[pid_index] = this_num_rmse_points_p
        if plot:
            add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, this_rmse_p)

    rmse_total, num_rmse_points_total = calc_rmse_total(rmse_total, num_rmse_points_total)
    var_string = ("s0: %.2fe-9, s1: %.0f, s2: %.2f, s3: %.2fe-9, w0: %.3fe-6, w1: %.3f, w2: %.3f, w3: %.3fe-6, "
                  "p0: %.3f, p1: %.4f, p2: %.1f, p3: %.2f, p4: %.3f, p5: %.1fe-9, p6: %.2f, "
                  "c0: %.1fe-6, c1: %.0f, c2: %.2f, c3: %.1fe-6, Vm: %.3f, Ci: %.3f"
                  % (s0*1.0e9, s1, s2, s3*1.0e9, w0*1.0e6, w1, w2, w3*1.0e6, p0, p1, p2, p3, p4, p5*1.0e9, p6,
                     c0*1.0e6, c1, c2, c3*1.0e6, Vm, Ci))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string.replace("p2", "<br>p2"))
    logging.log.debug("%s (%u) call with %s\n   results in rmse_total = %.10f %% (%.10f %%)"
                      % (mdl_name, mdl_id, var_string, rmse_total[0] * 100.0, rmse_total[1] * 100.0))
    return rmse_total


def model_f056_get_delta_age(task_queue, result_queue):
    time.sleep(1)  # sometimes the thread is called before task_queue is ready? wait a few seconds here.
    retry_counter = 0
    remaining_size = 1
    while True:
        try:
            remaining_size = task_queue.qsize()
            queue_entry = task_queue.get(block=False)
        except multiprocessing.queues.Empty:
            if remaining_size > 0:
                if retry_counter < 100:
                    retry_counter = retry_counter + 1
                    time.sleep(1)
                    continue
                else:
                    break
            else:
                break  # no more files

        if queue_entry is None:
            break  # no more files

        retry_counter = 0

        log_df = queue_entry["log_df"]
        df_length = log_df.shape[0]
        C_0 = queue_entry["C_0"]
        s0, s1, s2, s3, w0, w1, w2, w3, p0, p1, p2, p3, p4, p5, p6, c0, c1, c2, c3, Vm, Ci = queue_entry["vars"]

        t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

        V_ref = Vm  # 3.73
        T_ref = cfg.T0 + 25.0

        # How to determine ΔQ_loss_SEI = f0 * (SEI_potential - a_i * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 10) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)
                logging.log.warning("model_f056 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)
            logging.log.warning("model_f056 cannot read dt, t_func too short")

        # --- aging BEFORE the experiment ------------------------------------------------------------------------------
        t_storage = cfg.CELL_STORAGE_TEMPERATURE + cfg.T0
        v_storage = cfg.CELL_STORAGE_VOLTAGE
        time_production = cfg.CELL_PRODUCTION_TIMESTAMP
        time_experiment_start = cfg.EXPERIMENT_START_TIMESTAMP
        time_storage_steps = int(math.floor((time_experiment_start - time_production) / CHUNK_DURATION))
        sei_potential_storage = s0 * np.exp(s1 * (1.0 / t_storage - 1.0 / T_ref)) * np.exp(s2 * (v_storage - V_ref))
        q_loss_sei_storage = 0
        try:
            for i_step in range(time_storage_steps):
                diff_a = sei_potential_storage - s3 * q_loss_sei_storage
                if diff_a < 0.0:  # SEI layer can only be increased, not reduced
                    continue
                q_loss_sei_storage = q_loss_sei_storage + diff_a * CHUNK_DURATION
        except RuntimeWarning:
            logging.log.warning("model_f056 call with s0: %.2fe-9, s1: %.0f, s2: %.2f, s3: %.2fe-9,"
                                "w0: %.3fe-6, w1: %.3f, w2: %.3f, w3: %.3fe-6, "
                                "p0: %.3f, p1: %.4f, p2: %.1f, p3: %.2f, p4: %.3f, p5: %.1fe-9, p6: %.2f, "
                                "c0: %.1fe-6, c1: %.0f, c2: %.2f, c3: %.1fe-6, Vm: %.3f, Ci: %.3f "
                                "caused a RuntimeWarning (storage)"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6,
                                   p0, p1, p2, p3, p4, p5 * 1.0e9, p6, c0 * 1.0e6, c1, c2, c3 * 1.0e6, Vm, Ci))

        # --- aging DURING the experiment ------------------------------------------------------------------------------
        # only consider current if it is >1/500 C. For an 80 kWh battery: 160 W --> remove noise if I = ca. 0
        c_ths = cfg.CELL_CAPACITY_NOMINAL / 500.0
        cond_chg = (log_df[csv_label.I_CELL] > c_ths)
        cond_dischg = (log_df[csv_label.I_CELL] < -c_ths)

        c_rate_rel = log_df[csv_label.I_CELL].astype(np.float64) / cfg.CELL_CAPACITY_NOMINAL
        c_rate_rel.loc[cond_chg | cond_dischg] = c_rate_rel

        chg_rate_rel = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        chg_rate_rel.loc[cond_chg] = c_rate_rel
        t_internal_use = log_df[csv_label.T_CELL].astype(np.float64)

        sei_potential = (s0
                         * np.exp(s1 * (1.0 / t_internal_use - 1.0 / T_ref))
                         * np.exp(s2 * (log_df[csv_label.V_CELL].astype(np.float64) - V_ref)))
        cyclic_age_potential = w0 * (log_df[csv_label.V_CELL].astype(np.float64) - V_ref).abs()
        corrosion_potential = (c0
                               * np.exp(c1 * (1.0 / t_internal_use - 1.0 / T_ref))
                               * (c2 - log_df[csv_label.V_CELL].astype(np.float64)))  # c2 = V_corr_ths
        corrosion_potential[corrosion_potential < 0.0] = 0.0
        # current collector corrosion - see guoDegradationLithiumIon2021:
        # Guo, Thornton, Koronfel, et al.: "Degradation in lithium ion battery current collectors"

        q_loss_sei_total = q_loss_sei_storage
        q_loss_cyclic_total = 0
        q_loss_corrosion_total = 0
        q_loss_plating_total = 0
        q_loss_total = 0
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))

        # d_temperature = (p2 + cfg.T0) - t_internal_use
        d_temperature = p2 - t_internal_use
        d_temperature[d_temperature < 0.0] = 0.0
        p0t = p0 + (p1 * d_temperature)**p3  # base film resistance
        p5t = p5 * t_func  # base plating rate

        ix_max = (int(df_length / CHUNK_SIZE) + 1) * CHUNK_DURATION + 1
        q_loss_type_df = pd.DataFrame(np.nan, columns=LOSS_TYPES,
                                      index=range(0, ix_max, CHUNK_DURATION))

        try:
            i_lt = 0
            q_loss_type_df[LOSS_TYPE_SEI].iloc[i_lt] = q_loss_sei_total
            q_loss_type_df[LOSS_TYPE_CYC1].iloc[i_lt] = q_loss_cyclic_total
            q_loss_type_df[LOSS_TYPE_CYC2].iloc[i_lt] = q_loss_corrosion_total
            q_loss_type_df[LOSS_TYPE_PLATING].iloc[i_lt] = q_loss_plating_total

            anode_potential = get_v_anode_from_v_terminal_df_v4(log_df[csv_label.V_CELL])
            i_start = 0
            while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
                i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end not included when slicing df.iloc[i_start:i_end]

                c_rate_rel_roi = c_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)
                chg_rate_rel_roi = chg_rate_rel.iloc[i_start:i_end].astype(np.float64) / (1.0 - q_loss_total)

                diff_a = sei_potential.iloc[i_start:i_end] - s3 * q_loss_sei_total
                diff_a[diff_a < 0] = 0.0  # SEI layer can only be increased, not reduced

                diff_b = (cyclic_age_potential.iloc[i_start:i_end] * np.exp(w1 * max(q_loss_total - w2, 0))
                          - w3 * q_loss_cyclic_total)
                diff_b[diff_b < 0] = 0.0

                diff_c = corrosion_potential.iloc[i_start:i_end] - c3 * q_loss_corrosion_total
                diff_c[diff_c < 0] = 0.0
                v_plating = (anode_potential.iloc[i_start:i_end]  # V_a - C-rate * r_film(T, age)
                             - p0t.iloc[i_start:i_end] * np.exp(p4 * q_loss_total) * chg_rate_rel_roi)

                cond_v_plating_pos = (v_plating > 0.0)
                v_plating[cond_v_plating_pos] = 0.0

                dq_a = diff_a * t_func.iloc[i_start:i_end]
                dq_b = diff_b * t_func.iloc[i_start:i_end] * abs(c_rate_rel_roi)
                dq_c = diff_c * t_func.iloc[i_start:i_end] * abs(c_rate_rel_roi)
                dq_d = abs(v_plating) * p5t.iloc[i_start:i_end] * abs(chg_rate_rel_roi)**p6

                q_loss_sei_total = min(q_loss_sei_total + dq_a.sum(), 1.0)
                q_loss_cyclic_total = min(q_loss_cyclic_total + dq_b.sum(), 1.0)
                q_loss_corrosion_total = min(q_loss_corrosion_total + dq_c.sum(), 1.0)
                q_loss_plating_total = min(q_loss_plating_total + dq_d.sum(), 1.0)

                q_loss_total = min(q_loss_sei_total + q_loss_cyclic_total
                                   + q_loss_corrosion_total + q_loss_plating_total, 1.0)

                dq_loss.iloc[i_start:i_end] = dq_a + dq_b + dq_c + dq_d

                i_start = i_start + CHUNK_SIZE

                i_lt = i_lt + 1
                q_loss_type_df[LOSS_TYPE_SEI].iloc[i_lt] = q_loss_sei_total
                q_loss_type_df[LOSS_TYPE_CYC1].iloc[i_lt] = q_loss_cyclic_total
                q_loss_type_df[LOSS_TYPE_CYC2].iloc[i_lt] = q_loss_corrosion_total
                q_loss_type_df[LOSS_TYPE_PLATING].iloc[i_lt] = q_loss_plating_total

        except RuntimeWarning:
            logging.log.warning("model_f056 call with s0: %.2fe-9, s1: %.0f, s2: %.2f, s3: %.2fe-9, "
                                "w0: %.3fe-6, w1: %.3f, w2: %.3f, w3: %.3fe-6, "
                                "p0: %.3f, p1: %.4f, p2: %.1f, p3: %.2f, p4: %.3f, p5: %.1fe-9, p6: %.2f, "
                                "c0: %.1fe-6, c1: %.0f, c2: %.2f, c3: %.1fe-6, Vm: %.3f, Ci: %.3f "
                                "caused a RuntimeWarning"
                                % (s0 * 1.0e9, s1, s2, s3 * 1.0e9, w0 * 1.0e6, w1, w2, w3 * 1.0e6,
                                   p0, p1, p2, p3, p4, p5 * 1.0e9, p6, c0 * 1.0e6, c1, c2, c3 * 1.0e6, Vm, Ci))
        dq_loss.loc[0] = dq_loss[0] + q_loss_sei_storage
        result = (dq_loss * cfg.CELL_CAPACITY_NOMINAL).cumsum()

        mdl_cap_row = "model_f056" + DF_COL_AGE_MODEL_APPENDIX
        log_df_mdl_cap_row = C_0 - result
        cond_neg = (log_df_mdl_cap_row < 0.0)
        log_df_mdl_cap_row[cond_neg] = 0.0
        log_df.loc[:, mdl_cap_row] = log_df_mdl_cap_row

        q_loss_sei_total + q_loss_cyclic_total
        + q_loss_corrosion_total + q_loss_plating_total

        delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell = calc_rmse_cell_2(
            log_df, mdl_cap_row)

        pid_index = queue_entry["pid_index"]
        pnr_index = queue_entry["pnr_index"]

        result_entry = {"log_df": log_df, "q_loss_type_df": q_loss_type_df,
                        "others": [pid_index, pnr_index, delta_rmse_tot,
                                   delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell]}
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f056 END ===================================================================================================


# add the lines to the model result figure (if plotting is enabled)
def add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index, fig_list, fig_and_sp_from_pid_arr, rmse_cell,
                    C_0=None, q_loss_type=None):
    [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
    this_fig: go.Figure = fig_list[i_fig]

    if PLOT_AGING_TYPE and (q_loss_type is not None):
        q_loss_total = pd.Series(0.0, index=q_loss_type[LOSS_TYPES[0]].index)
        for i in range(0, len(LOSS_TYPES)):
            q_loss_total = q_loss_total + q_loss_type[LOSS_TYPES[i]]
        x_data_age_type = q_loss_type.index / TIME_DIV
        y_baseline = C_0 - q_loss_total * cfg.CELL_CAPACITY_NOMINAL
        stack_group = "aging_%u_%u" % (pid_index, pnr_index)
        this_fig.add_trace(go.Scatter(
            x=x_data_age_type, y=y_baseline, showlegend=False, mode='none', fillcolor="rgba(255,255,255,0)",
            line=dict(color="rgba(255,255,255,0)", width=TRACE_LINE_WIDTH), opacity=0.0, stackgroup=stack_group),
            row=(i_row + 1), col=(i_col + 1))
        for i in range(len(LOSS_TYPES)):
            loss_type = LOSS_TYPES[i]
            line_color = LOSS_TYPE_COLORS[i] % TRACE_OPACITY_AGE_TYPE
            fill_color = LOSS_TYPE_COLORS[i] % FILL_OPACITY_AGE_TYPE
            # x_data_age_type = q_loss_type.index / TIME_DIV
            y_data_age_type = q_loss_type[loss_type] * cfg.CELL_CAPACITY_NOMINAL
            this_fig.add_trace(go.Scatter(
                x=x_data_age_type, y=y_data_age_type, showlegend=False, mode='lines', fillcolor=fill_color,
                line=dict(color=line_color, width=TRACE_LINE_WIDTH), stackgroup=stack_group),
                row=(i_row + 1), col=(i_col + 1))

    # x_data = log_df.loc[::N_PLOT_DATA_DOWNSAMPLE, DF_COL_TIME_USE] / TIME_DIV
    x_data = log_df.loc[::N_PLOT_DATA_DOWNSAMPLE, csv_label.TIMESTAMP] / TIME_DIV
    y_data = log_df.loc[::N_PLOT_DATA_DOWNSAMPLE, mdl_cap_row]
    this_color = TRACE_COLORS[pnr_index]
    trace_name = TRACE_NAME_MDL % (mdl_name, pid_index, pnr_index)

    this_fig.add_trace(go.Scatter(
        x=x_data, y=y_data, showlegend=False, mode='lines', line=dict(color=this_color, width=TRACE_LINE_WIDTH),
        opacity=TRACE_OPACITY, name=trace_name, hovertemplate=PLOT_HOVER_TEMPLATE_MDL,
        fillcolor="rgba(255,255,255,0)"),
        row=(i_row + 1), col=(i_col + 1))
    # if LABEL_Q_LOSS_SEI in log_df.columns: --> doesn't really work? why? (also, this takes EXTREMELY long -> omit)
    #     this_hover_template = PLOT_HOVER_TEMPLATE_LOSS_TYPE_1 + "SEI" + PLOT_HOVER_TEMPLATE_LOSS_TYPE_2
    #     y_data_a = log_df.loc[::N_PLOT_DATA_DOWNSAMPLE, LABEL_Q_LOSS_SEI]
    #     this_fig.add_trace(go.Scatter(
    #         x=x_data, y=y_data_a, showlegend=False, mode='lines', opacity=TRACE_OPACITY, name=trace_name,
    #         line=dict(color=this_color, width=TRACE_LINE_WIDTH, dash="longdash"),
    #         hovertemplate=this_hover_template), row=(i_row + 1), col=(i_col + 1))
    #
    # if LABEL_Q_LOSS_wearout in log_df.columns:
    #     this_hover_template = PLOT_HOVER_TEMPLATE_LOSS_TYPE_1 + "wearout" + PLOT_HOVER_TEMPLATE_LOSS_TYPE_2
    #     y_data_b = log_df.loc[::N_PLOT_DATA_DOWNSAMPLE, LABEL_Q_LOSS_wearout]
    #     this_fig.add_trace(go.Scatter(
    #         x=x_data, y=y_data_b, showlegend=False, mode='lines', opacity=TRACE_OPACITY, name=trace_name,
    #         line=dict(color=this_color, width=TRACE_LINE_WIDTH, dash="dash"),
    #         hovertemplate=this_hover_template), row=(i_row + 1), col=(i_col + 1))
    #
    # if LABEL_Q_LOSS_PLATING in log_df.columns:
    #     this_hover_template = PLOT_HOVER_TEMPLATE_LOSS_TYPE_1 + "plating" + PLOT_HOVER_TEMPLATE_LOSS_TYPE_2
    #     y_data_c = log_df.loc[::N_PLOT_DATA_DOWNSAMPLE, LABEL_Q_LOSS_PLATING]
    #     this_fig.add_trace(go.Scatter(
    #         x=x_data, y=y_data_c, showlegend=False, mode='lines', opacity=TRACE_OPACITY, name=trace_name,
    #         line=dict(color=this_color, width=TRACE_LINE_WIDTH, dash="dot"),
    #         hovertemplate=this_hover_template), row=(i_row + 1), col=(i_col + 1))

    # add rmse_cell text
    i_pos = 0
    if len(TEXT_POS_RMSE_INDEX) > i_fig:
        if len(TEXT_POS_RMSE_INDEX[i_fig]) > i_row:
            if len(TEXT_POS_RMSE_INDEX[i_fig][i_row]) > i_col:
                i_pos = TEXT_POS_RMSE_INDEX[i_fig][i_row][i_col]

    text = TEXT_RMSE_CELL % (pnr_index + 1, rmse_cell[0] * 100.0, rmse_cell[1] * 100.0)
    y_fac = TEXT_POS_RMSE_DY_OFFSET[i_pos] + TEXT_POS_RMSE_DY_FACTOR[i_pos] * pnr_index
    # hotfix -> is this a rounding error or a bug in the annotation thingy or am I just stupid?
    if (i_pos == 0) and (pnr_index == 2):
        y_fac = y_fac - 0.65

    y_pos = (TEXT_POS_RMSE_Y_BASE[i_pos] + TEXT_POS_RMSE_DY * y_fac)
    x_pos = TEXT_POS_RMSE_X[i_pos]
    this_fig.add_annotation(xref="x domain", yref="y domain", x=x_pos, y=y_pos, showarrow=False,
                            opacity=ANNOTATION_OPACITY, bgcolor=BG_COLOR, text=text,
                            font=dict(size=ANNOTATION_FONT_SIZE, color=this_color),
                            row=(i_row + 1), col=(i_col + 1)
                            )


# add the model error bands to the model result figure (if plotting and uncertainty modeling is enabled)
def add_model_errorbands(mdl_cap_row, model_min, model_max, n_pids, fig_list, fig_and_sp_from_pid_arr):
    for pid_index in range(n_pids):
        [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
        this_fig: go.Figure = fig_list[i_fig]
        # trace_name = TRACE_NAME_MDL % (mdl_name, pid_index)
        # this_fig.add_trace(go.Scatter(
        #     x=list(model_min[pid_index].index), y=list(model_min[pid_index].values),
        #     showlegend=False, mode='lines', line=dict(color=FILL_COLOR_EDGE, width=TRACE_LINE_WIDTH, dash="dash"),
        #     opacity=FILL_OPACITY_EDGE, hoverinfo='skip', fill='tonexty', fillcolor=FILL_COLOR),
        #     row=(i_row + 1), col=(i_col + 1))
        # this_fig.add_trace(go.Scatter(
        #     x=list(model_max[pid_index].index), y=list(model_max[pid_index].values),
        #     showlegend=False, mode='lines', line=dict(color=FILL_COLOR_EDGE, width=TRACE_LINE_WIDTH, dash="dash"),
        #     opacity=FILL_OPACITY_EDGE, hoverinfo='skip', fill='tonexty', fillcolor=FILL_COLOR),
        #     row=(i_row + 1), col=(i_col + 1))
        # this_fig.add_trace(go.Scatter(
        #     x=list(model_min[pid_index].index), y=list(model_min[pid_index].values),
        #     showlegend=False, line=dict(color=FILL_COLOR_EDGE, width=TRACE_LINE_WIDTH, dash="dash"),
        #     opacity=FILL_OPACITY_EDGE),
        #     row=(i_row + 1), col=(i_col + 1))
        # this_fig.add_trace(go.Scatter(
        #     x=list(model_max[pid_index].index), y=list(model_max[pid_index].values),
        #     showlegend=False, line=dict(color=FILL_COLOR_EDGE, width=TRACE_LINE_WIDTH, dash="dash"),
        #     opacity=FILL_OPACITY_EDGE, fill='tonexty', fillcolor=FILL_COLOR),
        #     row=(i_row + 1), col=(i_col + 1))
        this_fig.add_trace(go.Scatter(
            x=list(model_min[pid_index].index), y=list(model_min[pid_index][mdl_cap_row].values), showlegend=False,
            hoverinfo='skip', line={"color": FILL_COLOR_EDGE, "width": TRACE_LINE_WIDTH, "dash": "dash"}),
            row=(i_row + 1), col=(i_col + 1))
        this_fig.add_trace(go.Scatter(
            x=list(model_max[pid_index].index), y=list(model_max[pid_index][mdl_cap_row].values), showlegend=False,
            hoverinfo='skip', line={"color": FILL_COLOR_EDGE, "width": TRACE_LINE_WIDTH, "dash": "dash"},
            fill='tonexty', fillcolor=FILL_COLOR),
            row=(i_row + 1), col=(i_col + 1))


# def add_model_trace_2(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index, this_fig, i_row, i_col, rmse_cell):
#     # x_data = log_df.loc[::N_PLOT_DATA_DOWNSAMPLE, DF_COL_TIME_USE] / TIME_DIV
#     x_data = log_df.loc[::N_PLOT_DATA_DOWNSAMPLE, csv_label.TIMESTAMP] / TIME_DIV
#     y_data = log_df.loc[::N_PLOT_DATA_DOWNSAMPLE, mdl_cap_row]
#     this_color = TRACE_COLORS[pnr_index]
#     trace_name = TRACE_NAME_MDL % (mdl_name, pid_index, pnr_index)
#     this_fig.add_trace(go.Scatter(
#         x=x_data, y=y_data, showlegend=False, mode='lines', line=dict(color=this_color, width=TRACE_LINE_WIDTH),
#         opacity=TRACE_OPACITY, name=trace_name, hovertemplate=PLOT_HOVER_TEMPLATE_MDL),
#         row=(i_row + 1), col=(i_col + 1))
#
#     # add rmse_cell text
#     text = TEXT_RMSE_CELL % (pnr_index + 1, rmse_cell * 100.0)
#     y_pos = TEXT_POS_RMSE_Y_BASE + TEXT_POS_RMSE_DY * (TEXT_POS_RMSE_DY_OFFSET + TEXT_POS_RMSE_DY_FACTOR * pnr_index)
#     x_pos = TEXT_POS_RMSE_X
#     this_fig.add_annotation(xref="x domain", yref="y domain", x=x_pos, y=y_pos, showarrow=False,
#                             opacity=TRACE_OPACITY, bgcolor=BG_COLOR, text=text, font=dict(size=10, color=this_color),
#                             row=(i_row + 1), col=(i_col + 1)
#                             )


# add RMSE to model result figure (if plotting is enabled)
def add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param):
    [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
    if fig_list[i_fig] is None:
        return  # skip
    this_fig: go.Figure = fig_list[i_fig]
    text = TEXT_RMSE_PARAM % (rmse_param[0] * 100.0, rmse_param[1] * 100.0)

    i_pos = 0
    if len(TEXT_POS_RMSE_INDEX) > i_fig:
        if len(TEXT_POS_RMSE_INDEX[i_fig]) > i_row:
            if len(TEXT_POS_RMSE_INDEX[i_fig][i_row]) > i_col:
                i_pos = TEXT_POS_RMSE_INDEX[i_fig][i_row][i_col]

    # hotfix -> is this a rounding error or a bug in the annotation thingy or am I just stupid?
    if i_pos == 1:
        y_pos = TEXT_POS_RMSE_Y_BASE[i_pos] + (TEXT_POS_RMSE_PARAM_OFFSET[i_pos] + 0.65) * TEXT_POS_RMSE_DY
    else:
        y_pos = TEXT_POS_RMSE_Y_BASE[i_pos] + TEXT_POS_RMSE_PARAM_OFFSET[i_pos] * TEXT_POS_RMSE_DY
    x_pos = TEXT_POS_RMSE_X[i_pos]
    this_fig.add_annotation(xref="x domain", yref="y domain", x=x_pos, y=y_pos, showarrow=False,
                            opacity=ANNOTATION_OPACITY, font=dict(size=ANNOTATION_FONT_SIZE, color=PARAM_RMSE_COLOR),
                            bgcolor=BG_COLOR, text=text,
                            row=(i_row + 1), col=(i_col + 1)
                            )
    # y_pos = TEXT_POS_RMSE_Y_BASE[i_pos] + TEXT_POS_RMSE_EXPLANATION_OFFSET[i_pos] * TEXT_POS_RMSE_DY
    # x_pos = TEXT_POS_RMSE_X[i_pos]
    # this_fig.add_annotation(xref="x domain", yref="y domain", x=x_pos, y=y_pos, showarrow=False,
    #                         opacity=ANNOTATION_OPACITY, font=dict(size=ANNOTATION_FONT_SIZE, color=PARAM_RMSE_COLOR),
    #                         bgcolor=BG_COLOR, text=TEXT_RMSE_EXPLAIN_DOUBLE,
    #                         row=(i_row + 1), col=(i_col + 1)
    #                         )


# plot the model result
def plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string):
    global iteration
    for i_fig in range(0, len(fig_list)):
        if fig_list[i_fig] is None:
            continue
        this_fig: go.Figure = fig_list[i_fig]
        title = this_fig['layout']['title']['text']
        rmse_text = TEXT_RMSE_TOTAL % (rmse_total[0] * 100.0, rmse_total[1] * 100.0)
        # title_pre = "<b>%s (%u): %s</b><br>" % (mdl_name, mdl_id, rmse_text)
        # this_fig['layout']['title']['text'] = title_pre + "<b>" + title + "</b>")
        # title_pre = "<b>%s (%u): %s</b>  -  " % (mdl_name, mdl_id, rmse_text)
        # this_fig['layout']['title']['text'] = title_pre + title + " - iteration %u<br>%s" % (iteration, var_string)
        this_fig['layout']['title']['text'] = PLOT_TITLE_RE % (mdl_name, rmse_text, title, var_string)

        age_type = cfg.age_type(i_fig)
        age_type_text = ""
        if age_type in AGE_TYPE_FILENAMES:
            age_type_text = AGE_TYPE_FILENAMES.get(age_type)

        run_timestring = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_base = PLOT_FILENAME_BASE % (mdl_name, mdl_id, age_type_text, run_timestring)

        if EXPORT_HTML or EXPORT_IMAGE:
            if not os.path.exists(cfg.IMAGE_OUTPUT_DIR):
                os.mkdir(cfg.IMAGE_OUTPUT_DIR)

            if EXPORT_HTML:
                filename = cfg.IMAGE_OUTPUT_DIR + filename_base + ".html"
                logging.log.debug("%s, age mode %s - saving figure as html\n    %s"
                                  % (mdl_name, age_type.name, filename))
                if os.path.isfile(filename):  # file already exists
                    file_exists = True
                else:
                    file_exists = False
                this_fig.write_html(filename, auto_open=SHOW_IN_BROWSER)
                if file_exists:
                    os.utime(filename)  # update "last modified date" -> can be pretty confusing otherwise

            if EXPORT_IMAGE:
                filename = cfg.IMAGE_OUTPUT_DIR + filename_base + "." + IMAGE_FORMAT
                logging.log.debug("%s, age mode %s - saving figure as image\n    %s"
                                  % (mdl_name, age_type.name, filename))
                if os.path.isfile(filename):  # file already exists
                    file_exists = True
                else:
                    file_exists = False
                this_fig.write_image(filename, format=IMAGE_FORMAT, engine=IMAGE_EXPORT_ENGINE,
                                     width=PLOT_WIDTH, height=this_fig['layout']['height'], scale=PLOT_SCALE_FACTOR)
                if file_exists:
                    os.utime(filename)  # update "last modified date" -> can be pretty confusing otherwise

        if SHOW_IN_BROWSER and (not EXPORT_HTML):
            logging.log.debug("%s, age mode %s - open figure in browser" % (mdl_name, age_type.name))
            this_fig.show(validate=False)


# return delta for sqrt(time series)
def get_t_func_sqrt(t_series):
    t_func = t_series.diff() / (t_series ** 0.5)
    t_func = t_func.fillna(0)
    cond = (t_func < 0)
    t_func.loc[cond] = 0  # set time function to 0 for t < 0 (e.g., if initial CU measurement is used for C_0
    #                                                         -> t_func = 0 before t_CU_0)
    return t_func


# return delta for time series
def get_t_func_diff(t_series):
    t_func = t_series.diff()
    t_func = t_func.fillna(0)
    return t_func


# calculate root-mean-square error (RMSE) for this cell - Version 1
def calc_rmse_cell(mdl_name, log_df, mdl_cap_row, rmse_tot, num_rmse_points_tot, rmse_par, num_rmse_points_par):
    # calculate RMSE at the data points (how? different approaches: per CU, per cell, per parameter)
    # filter valid cap_aged data points
    drop_cond = pd.isna(log_df[csv_label.CAP_CHARGED_EST])
    log_df_roi = log_df[~drop_cond]
    if log_df_roi.shape[0] == 0:
        logging.log.warning("%s - no valid cap_aged found! -> RMSE for cell not calculated" % mdl_name)
        rmse_cell = [0.0, 0.0]
    else:
        # actual RMSE of the cell
        cap_diff_all = (log_df_roi[csv_label.CAP_CHARGED_EST] - log_df_roi[mdl_cap_row]) / cfg.CELL_CAPACITY_NOMINAL
        se_cell_all = (cap_diff_all ** 2).sum()
        mse_cell_all = se_cell_all / cap_diff_all.shape[0]
        rmse_cell_all = (se_cell_all / cap_diff_all.shape[0])**0.5

        # actual RMSE of the cell
        flt = (log_df_roi[csv_label.CAP_CHARGED_EST] >= rmse_2_threshold)
        cap_diff_2 = ((log_df_roi[flt][csv_label.CAP_CHARGED_EST] - log_df_roi[flt][mdl_cap_row])
                      / cfg.CELL_CAPACITY_NOMINAL)
        se_cell_2 = (cap_diff_2 ** 2).sum()
        mse_cell_2 = se_cell_2 / cap_diff_2.shape[0]
        rmse_cell_2 = (se_cell_2 / cap_diff_2.shape[0])**0.5

        rmse_cell = [rmse_cell_all, rmse_cell_2]

        rmse_par_all, rmse_par_2 = rmse_par
        rmse_par_all = rmse_par_all + mse_cell_all
        rmse_par_2 = rmse_par_2 + mse_cell_2
        rmse_par = [rmse_par_all, rmse_par_2]
        num_rmse_points_par = num_rmse_points_par + 1

        if error_calc_method == error_calculation_method.RMSE_PER_CU:
            # rmse_tot = rmse_tot + cap_diff.abs().sum()

            rmse_tot_all, rmse_tot_2 = rmse_tot
            rmse_tot_all = rmse_tot_all + se_cell_all
            rmse_tot_2 = rmse_tot_2 + se_cell_2
            rmse_tot = [rmse_tot_all, rmse_tot_2]

            num_rmse_points_tot_all, num_rmse_points_tot_2 = num_rmse_points_tot
            num_rmse_points_tot_all = num_rmse_points_tot_all + cap_diff_all.shape[0]
            num_rmse_points_tot_2 = num_rmse_points_tot_2 + cap_diff_2.shape[0]
            num_rmse_points_tot = [num_rmse_points_tot_all, num_rmse_points_tot_2]

        elif error_calc_method == error_calculation_method.RMSE_PER_CELL:

            rmse_tot_all, rmse_tot_2 = rmse_tot
            rmse_tot_all = rmse_tot_all + mse_cell_all
            rmse_tot_2 = rmse_tot_2 + mse_cell_2
            rmse_tot = [rmse_tot_all, rmse_tot_2]

            num_rmse_points_tot = num_rmse_points_tot + 1
        elif error_calc_method == error_calculation_method.RMSE_PER_PARAMETER:
            # rmse_par = rmse_par + mse_cell
            # num_rmse_points_par = num_rmse_points_par + 1
            pass  # already calculated above
        else:
            logging.log.error("%s - RMSE calculation method not implemented: %s"
                              % (mdl_name, str(error_calc_method)))
    return rmse_tot, num_rmse_points_tot, rmse_par, num_rmse_points_par, rmse_cell


# calculate root-mean-square error (RMSE) for this cell - Version 2
def calc_rmse_cell_2(log_df, mdl_cap_row):
    # calculate RMSE at the data points (how? different approaches: per CU, per cell, per parameter)
    # filter valid cap_aged data points
    drop_cond = pd.isna(log_df[csv_label.CAP_CHARGED_EST])
    log_df_roi = log_df[~drop_cond]
    delta_rmse_par = [0.0, 0.0]
    delta_num_rmse_par = 0
    delta_rmse_tot = [0.0, 0.0]
    delta_num_rmse_points_tot = [0, 0]
    if log_df_roi.shape[0] == 0:
        logging.log.warning("no valid cap_aged found! -> RMSE for cell not calculated")
        rmse_cell = [0.0, 0.0]
    else:
        # actual RMSE of the cell
        cap_diff_all = (log_df_roi[csv_label.CAP_CHARGED_EST] - log_df_roi[mdl_cap_row]) / cfg.CELL_CAPACITY_NOMINAL
        se_cell_all = (cap_diff_all ** 2).sum()
        mse_cell_all = se_cell_all / cap_diff_all.shape[0]
        rmse_cell_all = (se_cell_all / cap_diff_all.shape[0]) ** 0.5

        # actual RMSE of the cell
        flt = (log_df_roi[csv_label.CAP_CHARGED_EST] >= rmse_2_threshold)
        cap_diff_2 = ((log_df_roi[flt][csv_label.CAP_CHARGED_EST] - log_df_roi[flt][mdl_cap_row])
                      / cfg.CELL_CAPACITY_NOMINAL)
        se_cell_2 = (cap_diff_2 ** 2).sum()
        mse_cell_2 = se_cell_2 / cap_diff_2.shape[0]
        rmse_cell_2 = (se_cell_2 / cap_diff_2.shape[0]) ** 0.5

        rmse_cell = [rmse_cell_all, rmse_cell_2]

        delta_rmse_par = [mse_cell_all, mse_cell_2]
        delta_num_rmse_par = 1

        if error_calc_method == error_calculation_method.RMSE_PER_PARAMETER:
            # rmse_par = rmse_par + mse_cell
            # num_rmse_points_par = num_rmse_points_par + 1
            pass  # already calculated above
        elif error_calc_method == error_calculation_method.RMSE_PER_CELL:
            delta_rmse_tot = [mse_cell_all, mse_cell_2]
            delta_num_rmse_points_tot = [1, 1]
        elif error_calc_method == error_calculation_method.RMSE_PER_CU:
            # rmse_tot = rmse_tot + cap_diff.abs().sum()
            delta_rmse_tot = [se_cell_all, se_cell_2]
            delta_num_rmse_points_tot = [cap_diff_all.shape[0], cap_diff_2.shape[0]]
        else:
            logging.log.error("RMSE calculation method not implemented: %s" % str(error_calc_method))
    return delta_rmse_tot, delta_num_rmse_points_tot, delta_rmse_par, delta_num_rmse_par, rmse_cell


# calculate root-mean-square error (RMSE) for this parameter
def calc_rmse_parameter(rmse_tot, num_rmse_points_tot, rmse_par, num_rmse_points_par):
    if num_rmse_points_par > 0:
        mse_par_all = rmse_par[0] / num_rmse_points_par
        mse_par_2 = rmse_par[1] / num_rmse_points_par
        rmse_par_all = mse_par_all**0.5
        rmse_par_2 = mse_par_2**0.5
        rmse_par = [rmse_par_all, rmse_par_2]
        if error_calc_method == error_calculation_method.RMSE_PER_PARAMETER:
            # rmse_tot = rmse_tot + ((rmse_par / num_rmse_points_par)**0.5)**2
            rmse_tot = [rmse_tot[0] + mse_par_all, rmse_tot[1] + mse_par_2]
            num_rmse_points_tot = [num_rmse_points_tot[0] + 1, num_rmse_points_tot[1] + 1]
    return rmse_tot, num_rmse_points_tot, rmse_par, num_rmse_points_par


# calculate total root-mean-square error (RMSE)
def calc_rmse_total(rmse_tot, num_rmse_points_tot):
    rmse_tot_all = rmse_tot[0]
    rmse_tot_2 = rmse_tot[1]
    num_rmse_points_tot_all = num_rmse_points_tot[0]
    num_rmse_points_tot_2 = num_rmse_points_tot[1]
    if num_rmse_points_tot_all > 0:
        rmse_tot_all = (rmse_tot_all / num_rmse_points_tot_all)**0.5
    if num_rmse_points_tot_2 > 0:
        rmse_tot_2 = (rmse_tot_2 / num_rmse_points_tot_2)**0.5
    rmse_tot = [rmse_tot_all, rmse_tot_2]
    num_rmse_points_tot = [num_rmse_points_tot_all, num_rmse_points_tot_2]
    return rmse_tot, num_rmse_points_tot


# initialize variables used for the models
def prepare_model(mdl_name, log_age_dfs):
    n_pids = len(log_age_dfs)
    if n_pids == 0:
        skip = True
        n_pnrs = 0
    else:
        skip = False
        n_pnrs = len(log_age_dfs[0])
    mdl_cap_delta_row = mdl_name + DF_COL_AGE_CAP_DELTA_APPENDIX
    mdl_cap_row = mdl_name + DF_COL_AGE_MODEL_APPENDIX
    return mdl_cap_delta_row, mdl_cap_row, n_pids, n_pnrs, skip


# prepare cell log data frame for aging model
def prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod, mdl_cap_delta_row, mdl_cap_row,
                        c_init=cfg.CELL_CAPACITY_NOMINAL):
    C_0 = c_init
    if (log_df is None) or (log_df.shape[0] == 0):
        # logging.log.warning("%s - log_df is empty -> skipping cell" % mdl_name)
        skip = True
        return log_df, C_0, skip

    log_df.loc[:, mdl_cap_delta_row] = DF_COL_AGE_CAP_DELTA_FILL
    log_df.loc[:, mdl_cap_row] = DF_COL_AGE_MODEL_FILL

    # prepare C_0 and time
    skip = False
    if use_t_prod:
        delta_time = cfg.EXPERIMENT_START_TIMESTAMP - cfg.CELL_PRODUCTION_TIMESTAMP
        log_df.loc[:, DF_COL_TIME_USE] = log_df[csv_label.TIMESTAMP] + delta_time
    else:
        log_df.loc[:, DF_COL_TIME_USE] = log_df[csv_label.TIMESTAMP]

    if use_cap_nom < 1.0:
        cap_aged = log_df[csv_label.CAP_CHARGED_EST][~pd.isna(log_df[csv_label.CAP_CHARGED_EST])]
        if cap_aged.shape[0] == 0:
            logging.log.warning("%s - no valid cap_aged found! -> skipping cell" % mdl_name)
            skip = True
        else:
            C_0 = use_cap_nom * C_0 + (1.0 - use_cap_nom) * cap_aged.iloc[0]
            i_C_0 = cap_aged.index[0]
            # log_df.loc[:, DF_COL_TIME_USE] = log_df[csv_label.TIMESTAMP] - log_df.loc[i_C_0, csv_label.TIMESTAMP]
            cond = log_df.index[log_df.index < i_C_0]
            log_df.loc[cond, DF_COL_TIME_USE] = np.nan
            # set everything before i_C_0 to NaN, so t_func will return 0
    return log_df, C_0, skip


# generate ALL base figures for all aging types, onto which all model results are plotted
def generate_base_figures(cell_log_csv, slave_cell_found, num_parameters, num_cells_per_parameter):
    # read all cfg files that are available and build up param_df
    param_df = generate_param_df_from_cfgs(cell_log_csv, slave_cell_found, num_parameters, num_cells_per_parameter)
    age_types = param_df[csv_label.AGE_TYPE].drop_duplicates().values

    fig_list = [None for _ in cfg.age_type]
    fig_and_sp_from_pid_arr = [[-1, -1, -1] for _ in range(0, num_parameters)]  # [i_fig in fig_list, i_row, i_col]
    if cfg.age_type.CALENDAR in age_types:
        # build up calendar aging figure:
        # columns: temperatures from lowest to highest
        # rows: SoC from lowest to highest
        age_type = cfg.age_type.CALENDAR
        pdf_age = param_df[(param_df[csv_label.AGE_TYPE] == age_type)
                           & (param_df[csv_label.CFG_CELL_USED] == cfg.cell_used.AUTO.value)]
        t_list = pdf_age[csv_label.AGE_TEMPERATURE].drop_duplicates().sort_values().values
        soc_list = pdf_age[csv_label.AGE_SOC].drop_duplicates().sort_values().values
        n_cols = len(t_list)
        n_rows = len(soc_list)
        subplot_titles = ['---' for _ in range(0, n_cols * n_rows)]
        for i_c in range(0, n_cols):
            t_text = ht.get_age_temp_text(t_list[i_c])
            for i_r in range(0, n_rows):
                soc_text = ht.get_age_soc_text_percent(soc_list[i_r])
                i_t = (i_r * n_cols) + i_c
                subplot_titles[i_t] = t_text + ", " + soc_text

        if MANUAL_CAL_AGING_EXPERIMENT_SHOW:
            if n_cols == 0:
                n_cols = 1
            pdf_age = param_df[param_df[csv_label.AGE_TYPE] == age_type]
            n_rows = n_rows + 1
            subplot_titles.extend(['---' for _ in range(0, n_cols)])
            i_t = ((n_rows - 1) * n_cols) + MANUAL_CAL_AGING_EXPERIMENT_SHOW_COL
            t_text = ht.get_age_temp_text(MANUAL_CAL_AGING_EXPERIMENT_T_USE_DEGC)
            soc_text = f"%.1f %%" % MANUAL_CAL_AGING_EXPERIMENT_SOC_USE_PERCENT
            subplot_titles[i_t] = t_text + ", " + soc_text
        plot_title = ""
        if age_type in AGE_TYPE_TITLES:
            plot_title = AGE_TYPE_TITLES.get(age_type)
        # noinspection PyTypeChecker
        fig_list[age_type] = generate_base_figure(n_rows, n_cols, plot_title, subplot_titles, age_type)
        for _, param_row in pdf_age.iterrows():
            pid = int(param_row[csv_label.PARAMETER_ID])
            if MANUAL_CAL_AGING_EXPERIMENT_SHOW and (pid == num_parameters):
                i_col = MANUAL_CAL_AGING_EXPERIMENT_SHOW_COL
                i_row = n_rows - 1
                fig_and_sp_from_pid_arr[pid - 1] = [age_type.value, i_row, i_col]
            else:
                i_col = np.where(t_list == param_row[csv_label.AGE_TEMPERATURE])[0][0]
                i_row = np.where(soc_list == param_row[csv_label.AGE_SOC])[0][0]
                fig_and_sp_from_pid_arr[pid - 1] = [age_type.value, i_row, i_col]

    if cfg.age_type.CYCLIC in age_types:
        # build up calendar aging figure:
        # columns: temperatures from lowest to highest
        # rows (outer loop): C-rate (unsorted - in my experiment: from lowest to highest)
        # rows (inner loop): SoC range (unsorted - in my experiment: from widest to narrowest)
        age_type = cfg.age_type.CYCLIC
        pdf_age = param_df[param_df[csv_label.AGE_TYPE] == age_type]
        t_list = pdf_age[csv_label.AGE_TEMPERATURE].drop_duplicates().sort_values().values
        c_rate_list = pdf_age[csv_label.AGE_C_RATES].drop_duplicates().values
        soc_range_list = pdf_age[csv_label.AGE_SOC_RANGE].drop_duplicates().values
        n_cols = len(t_list)
        n_c_rates = len(c_rate_list)
        n_soc_ranges = len(soc_range_list)
        n_rows = n_c_rates * n_soc_ranges
        subplot_titles = ['---' for _ in range(0, n_cols * n_rows)]
        for i_c in range(0, n_cols):
            t_text = ht.get_age_temp_text(t_list[i_c])
            for i_cr in range(0, n_c_rates):
                c_rate_text = c_rate_list[i_cr]
                for i_sr in range(0, n_soc_ranges):
                    i_r = (i_cr * n_soc_ranges) + i_sr
                    soc_range_text = soc_range_list[i_sr]
                    i_t = (i_r * n_cols) + i_c
                    subplot_titles[i_t] = t_text + ", " + soc_range_text + ",<br>" + c_rate_text
        plot_title = ""
        if age_type in AGE_TYPE_TITLES:
            plot_title = AGE_TYPE_TITLES.get(age_type)
        # noinspection PyTypeChecker
        fig_list[age_type] = generate_base_figure(n_rows, n_cols, plot_title, subplot_titles, age_type)
        for _, param_row in pdf_age.iterrows():
            pid = int(param_row[csv_label.PARAMETER_ID])
            i_col = np.where(t_list == param_row[csv_label.AGE_TEMPERATURE])[0][0]
            i_cr = np.where(c_rate_list == param_row[csv_label.AGE_C_RATES])[0][0]
            i_sr = np.where(soc_range_list == param_row[csv_label.AGE_SOC_RANGE])[0][0]
            i_row = (i_cr * n_soc_ranges) + i_sr
            fig_and_sp_from_pid_arr[pid - 1] = [age_type.value, i_row, i_col]

    if cfg.age_type.PROFILE in age_types:
        # build up profile aging figure:
        # columns: temperatures from lowest to highest
        # rows: profile number (rising)
        age_type = cfg.age_type.PROFILE
        pdf_age = param_df[param_df[csv_label.AGE_TYPE] == age_type]
        t_list = pdf_age[csv_label.AGE_TEMPERATURE].drop_duplicates().sort_values().values
        profile_list = pdf_age[csv_label.AGE_PROFILE].drop_duplicates().sort_values().values
        n_cols = len(t_list)
        n_rows = len(profile_list)
        subplot_titles = ['---' for _ in range(0, n_cols * n_rows)]
        for i_c in range(0, n_cols):
            t_text = ht.get_age_temp_text(t_list[i_c])
            for i_r in range(0, n_rows):
                profile_text = ht.get_age_profile_text(int(profile_list[i_r]))
                i_t = (i_r * n_cols) + i_c
                subplot_titles[i_t] = t_text + ",<br>" + profile_text
        plot_title = ""
        if age_type in AGE_TYPE_TITLES:
            plot_title = AGE_TYPE_TITLES.get(age_type)
        # noinspection PyTypeChecker
        fig_list[age_type] = generate_base_figure(n_rows, n_cols, plot_title, subplot_titles, age_type)
        for _, param_row in pdf_age.iterrows():
            pid = int(param_row[csv_label.PARAMETER_ID])
            i_col = np.where(t_list == param_row[csv_label.AGE_TEMPERATURE])[0][0]
            i_row = np.where(profile_list == param_row[csv_label.AGE_PROFILE])[0][0]
            fig_and_sp_from_pid_arr[pid - 1] = [age_type.value, i_row, i_col]

    return fig_list, fig_and_sp_from_pid_arr, param_df


# plot check-up measurement data onto plots
def fill_figures_with_measurements(fig_list, fig_and_sp_from_pid_arr, log_dfs, param_df):
    for i_param_id in range(0, len(log_dfs)):  # for each parameter ID
        [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[i_param_id]
        if (i_fig < 0) or (fig_list[i_fig] is None):
            continue
        this_fig: go.Figure = fig_list[i_fig]
        for i_param_nr in range(0, len(log_dfs[i_param_id])):  # for each parameter nr
            log_df: pd.DataFrame = log_dfs[i_param_id][i_param_nr]
            log_df_roi = log_df[~pd.isna(log_df[csv_label.CAP_CHARGED_EST])]
            x_data = log_df_roi[csv_label.TIMESTAMP] / TIME_DIV
            y_data = log_df_roi[csv_label.CAP_CHARGED_EST]
            data_range = range(0, x_data.shape[0])
            text_data = pd.Series("", index=data_range)
            for i in data_range:
                text_data.loc[i] = PLOT_TEXT_CU % (i + 1, x_data.iloc[i], log_df_roi[csv_label.EFC].iloc[i])
            # this_color = TRACE_COLORS[i_param_nr]
            this_marker_style = MARKER_STYLE.copy()
            this_marker_style["color"] = TRACE_COLORS[i_param_nr]
            pid = i_param_id + 1
            pnr = i_param_nr + 1
            # print(pid, pnr)
            slave_id = int(param_df[csv_label.SLAVE_ID][(param_df[csv_label.PARAMETER_ID] == pid)
                                                        & (param_df[csv_label.PARAMETER_NR] == pnr)].iloc[0])
            cell_id = int(param_df[csv_label.CELL_ID][(param_df[csv_label.PARAMETER_ID] == pid)
                                                      & (param_df[csv_label.PARAMETER_NR] == pnr)].iloc[0])
            trace_name = TRACE_NAME_CU % (pid, pnr, slave_id, cell_id)
            this_fig.add_trace(go.Scatter(
                x=x_data, y=y_data, showlegend=False, mode='markers', marker=this_marker_style,
                name=trace_name, text=text_data, hovertemplate=PLOT_HOVER_TEMPLATE_CU),
                row=(i_row + 1), col=(i_col + 1))  # line=dict(color=this_color),

    return fig_list


# generate ONE base figure for one of the aging type, onto which the model results are plotted
def generate_base_figure(n_rows, n_cols, plot_title, subplot_titles, age_type):
    subplot_h_spacing = (SUBPLOT_H_SPACING_REL / n_cols)
    subplot_v_spacing = (SUBPLOT_V_SPACING_REL / n_rows)
    plot_height = HEIGHT_PER_ROW * n_rows
    plot_title_y_pos = 1.0 - PLOT_TITLE_Y_POS_REL / plot_height

    fig = make_subplots(rows=n_rows, cols=n_cols, shared_xaxes="all",
                        horizontal_spacing=subplot_h_spacing, vertical_spacing=subplot_v_spacing,
                        subplot_titles=subplot_titles)

    fig.update_xaxes(ticks="outside",
                     tickcolor=MAJOR_GRID_COLOR,
                     ticklen=10,
                     showticklabels=True,
                     minor=dict(griddash='dot',
                                gridcolor=MINOR_GRID_COLOR,
                                ticklen=4)
                     )
    fig.update_yaxes(ticks="outside",
                     tickcolor=MAJOR_GRID_COLOR,
                     ticklen=10,
                     showticklabels=True,
                     minor=dict(griddash='dot',
                                gridcolor=MINOR_GRID_COLOR,
                                ticklen=4),
                     # title_text=Y_AXIS_TITLE,
                     # title={'text': Y_AXIS_TITLE, 'font': dict(size=AXIS_FONT_SIZE)},
                     )
    if USE_COMPACT_PLOT:
        for i_row in range(0, n_rows):
            fig.update_yaxes(title_text=Y_AXIS_TITLE, row=(i_row + 1), col=1)
    else:
        fig.update_yaxes(title_text=Y_AXIS_TITLE)

    if age_type == cfg.age_type.CALENDAR:
        if USE_MANUAL_PLOT_LIMITS_CAL:
            fig.update_yaxes(range=MANUAL_PLOT_LIMITS_Y_CAL)
    elif age_type == cfg.age_type.CYCLIC:
        if USE_MANUAL_PLOT_LIMITS_CYC:
            fig.update_yaxes(range=MANUAL_PLOT_LIMITS_Y_CYC)
    elif age_type == cfg.age_type.PROFILE:
        if USE_MANUAL_PLOT_LIMITS_PRF:
            fig.update_yaxes(range=MANUAL_PLOT_LIMITS_Y_PRF)

    for i_col in range(0, n_cols):
        fig.update_xaxes(title_text=X_AXIS_TITLE, row=n_rows, col=(i_col + 1))
        # fig.update_xaxes(title={'text': X_AXIS_TITLE, 'font': dict(size=AXIS_FONT_SIZE)},
        #                  row=n_rows, col=(i_col + 1))

    fig.update_layout(title={'text': plot_title,
                             'font': dict(size=TITLE_FONT_SIZE, color='black'),
                             'y': plot_title_y_pos,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template=FIGURE_TEMPLATE,
                      autosize=True,
                      height=plot_height,
                      width=PLOT_WIDTH,
                      legend=dict(x=0, y=0),
                      margin=dict(l=SUBPLOT_LR_MARGIN, r=SUBPLOT_LR_MARGIN,
                                  t=SUBPLOT_TOP_MARGIN, b=SUBPLOT_BOT_MARGIN,
                                  pad=SUBPLOT_PADDING)
                      )
    # fig.update_annotations(font={'size': SUBPLOT_TITLE_FONT_SIZE})  # subplot titles are annotations
    # fig.update_layout(hovermode='x unified')
    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    return fig


# generate parameter DataFrame from configuration files
def generate_param_df_from_cfgs(cell_log_csv, slave_cell_found, num_parameters, num_cells_per_parameter):
    cfg_data = np.full((num_parameters, num_cells_per_parameter), None)

    # I. collect cfg_data of ALL parameters in array[param_id][param_nr]
    last_valid_cfg_i_param_id = None
    last_valid_cfg_i_param_nr = None
    for i_csv in range(0, len(cell_log_csv)):
        cell = cell_log_csv[i_csv]
        slave_id = cell[csv_label.SLAVE_ID]
        cell_id = cell[csv_label.CELL_ID]
        if slave_cell_found[slave_id][cell_id] == "X":
            param_id = cell[csv_label.PARAMETER_ID]
            param_nr = cell[csv_label.PARAMETER_NR]
            i_param_id = param_id - 1
            i_param_nr = param_nr - 1
            filename_cfg = cell["cfg_filename"]

            # open and read relevant cfg data
            cfg_df = pd.read_csv(cfg.CSV_RESULT_DIR + filename_cfg, header=0, sep=cfg.CSV_SEP, engine="pyarrow")
            cfg_df = cfg_df.iloc[0, :].copy()

            age_type = cfg_df[csv_label.AGE_TYPE]

            cfg_df.loc[csv_label.CFG_AGE_V] = -1  # don't set to nan, so the cfg_df's can be compared with any/all()
            cfg_df.loc[csv_label.CFG_AGE_V_MIN] = -1
            cfg_df.loc[csv_label.CFG_AGE_V_MAX] = -1
            cfg_df.loc[csv_label.CFG_AGE_SOC_MIN] = -1
            cfg_df.loc[csv_label.CFG_AGE_SOC_MAX] = -1
            cfg_df.loc[csv_label.AGE_SOC_RANGE] = "?-?"
            cfg_df.loc[csv_label.AGE_V_RANGE] = "?-?"
            cfg_df.loc[csv_label.AGE_C_RATES] = "?-?"
            if age_type == cfg.age_type.CALENDAR:
                # add v idle to cfg
                cfg_df.loc[csv_label.CFG_AGE_V] = ht.get_v_idle_from_soc(cfg_df[csv_label.AGE_SOC], 0)
            elif age_type == cfg.age_type.CYCLIC:
                # add soc max and min to cfg
                v_min = cfg_df[csv_label.V_MIN_CYC]
                v_max = cfg_df[csv_label.V_MAX_CYC]
                cfg_df.loc[csv_label.CFG_AGE_V_MIN] = v_min
                cfg_df.loc[csv_label.CFG_AGE_V_MAX] = v_max
                soc_min = ht.get_soc_from_v_cell(v_min)
                soc_max = ht.get_soc_from_v_cell(v_max)
                cfg_df.loc[csv_label.CFG_AGE_SOC_MIN] = soc_min
                cfg_df.loc[csv_label.CFG_AGE_SOC_MAX] = soc_max
                cfg_df.loc[csv_label.AGE_SOC_RANGE] = csv_label.AGE_SOC_RANGE_RE % (soc_min, soc_max)
                cfg_df.loc[csv_label.AGE_V_RANGE] = csv_label.AGE_V_RANGE_RE % (v_min, v_max)
                cc = cfg_df[csv_label.AGE_CHG_RATE]
                cd = cfg_df[csv_label.AGE_DISCHG_RATE]
                cfg_df.loc[csv_label.AGE_C_RATES] = csv_label.AGE_C_RATES_RE % (cc, cd)
            elif age_type == cfg.age_type.PROFILE:
                # add soc max and min to cfg (and chg rate?)
                age_profile = int(cfg_df[csv_label.AGE_PROFILE])
                i_chg_rate = round(cfg_df[csv_label.I_CHG_MAX_CYC] / cfg.CELL_CAPACITY_NOMINAL, 2)
                cfg_df.loc[csv_label.AGE_CHG_RATE] = i_chg_rate
                soc_min = cfg.AGE_PROFILE_SOC_MIN[age_profile]
                soc_max = cfg.AGE_PROFILE_SOC_MAX[age_profile]
                cfg_df.loc[csv_label.CFG_AGE_SOC_MIN] = soc_min
                cfg_df.loc[csv_label.CFG_AGE_SOC_MAX] = soc_max
                v_min = ht.get_v_idle_from_soc(soc_min, -1)
                v_max = ht.get_v_idle_from_soc(soc_max, 1)
                cfg_df.loc[csv_label.CFG_AGE_V_MIN] = v_min
                cfg_df.loc[csv_label.CFG_AGE_V_MAX] = v_max
                cfg_df.loc[csv_label.AGE_SOC_RANGE] = csv_label.AGE_SOC_RANGE_RE % (soc_min, soc_max)
                cfg_df.loc[csv_label.AGE_V_RANGE] = f"%.2f-%.2f" % (v_min, v_max)

            cfg_data[i_param_id][i_param_nr] = cfg_df

            last_valid_cfg_i_param_id = i_param_id
            last_valid_cfg_i_param_nr = i_param_nr

    if (last_valid_cfg_i_param_id is None) or (last_valid_cfg_i_param_nr is None):
        logging.log.error("no valid config")
        return pd.DataFrame()

    if MANUAL_CAL_AGING_EXPERIMENT_SHOW:
        # noinspection PyTypeChecker
        cfg_manual_cal_aging_df: pd.Series = cfg_data[last_valid_cfg_i_param_id][last_valid_cfg_i_param_nr].copy()
        cfg_manual_cal_aging_df.loc[csv_label.CFG_CELL_USED] = cfg.cell_used.MANUAL
        cfg_manual_cal_aging_df.loc[csv_label.SD_BLOCK_ID] = 0
        cfg_manual_cal_aging_df.loc[csv_label.SLAVE_ID] = 20
        cfg_manual_cal_aging_df.loc[csv_label.PARAMETER_ID] = num_parameters
        cfg_manual_cal_aging_df.loc[csv_label.AGE_TYPE] = cfg.age_type.CALENDAR.value
        cfg_manual_cal_aging_df.loc[csv_label.AGE_SOC] = MANUAL_CAL_AGING_EXPERIMENT_SOC_USE_PERCENT
        cfg_manual_cal_aging_df.loc[csv_label.AGE_CHG_RATE] = 0.0
        cfg_manual_cal_aging_df.loc[csv_label.AGE_DISCHG_RATE] = 0.0
        cfg_manual_cal_aging_df.loc[csv_label.AGE_PROFILE] = 0.0
        cfg_manual_cal_aging_df.loc[csv_label.V_MAX_CYC] = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        cfg_manual_cal_aging_df.loc[csv_label.V_MIN_CYC] = MANUAL_CAL_AGING_EXPERIMENT_V_USE_V
        cfg_manual_cal_aging_df.loc[csv_label.I_CHG_MAX_CYC] = 1.0
        cfg_manual_cal_aging_df.loc[csv_label.I_DISCHG_MAX_CYC] = -1.0
        for i in range(0, num_cells_per_parameter):
            this_cfg_df = cfg_manual_cal_aging_df.copy()
            this_cfg_df.loc[csv_label.CELL_ID] = i
            this_cfg_df.loc[csv_label.PARAMETER_NR] = i + 1
            cfg_data[num_parameters - 1][i] = this_cfg_df

    # II. collect param_df
    cfg_param_df_drop_col = [csv_label.SD_BLOCK_ID]  # , csv_label.SLAVE_ID, csv_label.CELL_ID
    # noinspection PyTypeChecker
    cfg_a: pd.Series = cfg_data[last_valid_cfg_i_param_id][last_valid_cfg_i_param_nr].copy()
    cfg_a.drop(cfg_param_df_drop_col, inplace=True)
    param_df = pd.DataFrame(columns=cfg_a.index.tolist())
    for i_param_id in range(0, num_parameters):
        for i_param_nr in range(0, num_cells_per_parameter):
            if cfg_data[i_param_id][i_param_nr] is not None:
                # noinspection PyTypeChecker
                cfg_a: pd.Series = cfg_data[i_param_id][i_param_nr].copy()
                cfg_a.drop(cfg_param_df_drop_col, inplace=True)
                i_new = param_df.shape[0]
                param_df.loc[i_new, :] = cfg_a
    return param_df


# get key-value dictionary from string
def get_kv_dict(values):
    kv_dict = {}
    #  "a0: 0.0001, a1: 1.068, a2: 0.991"
    # for re_pat_float, also see https://stackoverflow.com/a/4703508/2738240
    re_pat_float = "\s*(\w+)\s*:\s*([-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)"
    rex_k_float = re.compile(re_pat_float)
    # kv_arr = "a0: 2.1, a1: 4, a2: -3, a3: -3.2134192, a4: .21, a5: -2.1e-3".split(",")
    kv_arr = values.split(",")
    for kv in kv_arr:
        # print("kv is '%s'" % kv)
        re_match = rex_k_float.fullmatch(kv)
        if re_match:
            k = re_match.group(1)
            v = float(re_match.group(2))
            kv_dict[k] = v
        # else:
        #     print("no match")
    return kv_dict


# estimate cell anode potential from SoC (input and output as float)
def get_v_anode(soc):
    # px / 394 * 0.7 = 0,0017766497461928934010152284264
    # v_a_0 = 0.54
    # v_a_7_5 = 0.235
    # v_a_25 = 0.151
    # v_a_75 = v_a_100 = 0.091
    # can be visualized using this tool: https://www.desmos.com/calculator?lang=de  => replace soc with x and v_a with y
    if soc > 1.0:
        # v_a = 0.091 + (soc - 1.0) * (0 - 0.091) / 0.25
        #     = 0.091 - (soc - 1.0) * 0.364
        #     = 0.091 - (soc * 0.364 - 0.364)
        #     = 0.455 - soc * 0.364
        return 0.455 - soc * 0.364
    elif soc > 0.75:
        # v_a = 0.091
        return 0.091  # 91 mV
    elif soc > 0.25:
        # v_a = 0.151 + (soc - 0.25) * (0.091 - 0.151) / 0.5
        #     = 0.151 - (soc - 0.25) * 0.12
        #     = 0.151 - (soc * 0.12 - 0.25 * 0.12)
        #     = 0.181 - soc * 0.12
        return 0.181 - soc * 0.12
    elif soc > 0.075:
        # v_a = 0.235 + (soc - 0.075) * (0.151 - 0.235) / 0.175
        #     = 0.235 - (soc - 0.075) * 0.48
        #     = 0.235 - (soc * 0.48 - 0.036)
        #     = 0.271 - soc * 0.48
        return 0.271 - soc * 0.48
    else:
        # v_a = 0.54 + soc * (0.235 - 0.54) / 0.075
        #     = 0.54 + soc * (0.235 - 0.54) / 0.075
        #     = 0.54 - soc * 4.067
        return 0.54 - soc * 4.067
    # test:
    # import matplotlib.pyplot as plt
    # x = np.array(range(-25, 125, 1)) / 100.0
    # y = np.full(len(x), np.nan)
    # plt.scatter(x, y, c="b")
    # plt.xlabel("SoC [100%]")
    # plt.ylabel("V_a")
    # plt.grid(True)
    # plt.show()


# estimate cell anode potential from terminal voltage (input and output as DataFrame) - Version 1
def get_v_anode_from_v_terminal_df(v_terminal_df):
    # # for an example for V_anode potential as a function of SoC, see:
    # #   schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
    # #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
    # return 0.0889 + 0.18 * (v_terminal_df - 4.2)**2  # estimation valid from 2.5 to 4.2 V (probably 2.0 to 4.3 V?)

    # better behavior towards high SoC:
    #   gallagherVolumeAveragedApproach2012: Gallagher K.G., Dees D.W., Jansen A.N., et al.:
    #   "A Volume Averaged Approach to the Numerical Modeling of Phase-Transition Intercalation Electrodes Presented
    #    for Li x C 6", J. Electrochem. Soc., vol. 159, no. 12, pp. A2029–A2037, 2012, DOI 10.1149/2.015301jes
    # I used Figure 3 (-> WebPlotDigitizer), set x=0.05 in Li_{x}C_{6} as SoC = 4 % and x = 0.96 as SoC = 100 %
    # for SoC < 4%, I used values from:
    #   schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
    #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
    # Then, using the OCV curve from measurements, I created a plot for V_anode(V_OCV)
    # Next, I used linear approximation functions to match the anode potential closely (particularly for  SoC > 5%)
    V1 = 4.076
    V2 = 3.832
    V3 = 3.668
    V4 = 3.536
    V5 = 3.372
    V6 = 3.259
    cond_a1 = (v_terminal_df > V1)
    cond_a2 = ((v_terminal_df > V2) & (v_terminal_df <= V1))
    cond_a3 = ((v_terminal_df > V3) & (v_terminal_df <= V2))
    cond_a4 = ((v_terminal_df > V4) & (v_terminal_df <= V3))
    cond_a5 = ((v_terminal_df > V5) & (v_terminal_df <= V4))
    cond_a6 = ((v_terminal_df > V6) & (v_terminal_df <= V5))
    cond_a7 = (v_terminal_df <= V6)
    v_anode = pd.Series(data=0.0, dtype=np.float64, index=v_terminal_df.index)
    v_anode[cond_a1] = 1.943616239 - 0.456980057 * v_terminal_df[cond_a1]
    v_anode[cond_a2] = 0.081  # reduce?? -> then also adjust V1 and V2 -> get_v_anode_from_v_terminal_df_v2
    v_anode[cond_a3] = 0.948586964 - 0.22641873 * v_terminal_df[cond_a3]
    v_anode[cond_a4] = 0.118
    v_anode[cond_a5] = 1.670224522 - 0.439009094 * v_terminal_df[cond_a5]
    v_anode[cond_a6] = 0.820011235 - 0.187182325 * v_terminal_df[cond_a6]
    v_anode[cond_a7] = 2.228263548 - 0.619305419 * v_terminal_df[cond_a7]
    return v_anode


# estimate cell anode potential from terminal voltage (input and output as DataFrame) - Version 2
def get_v_anode_from_v_terminal_df_v2(v_terminal_df):
    # # for an example for V_anode potential as a function of SoC, see:
    # #   schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
    # #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
    # return 0.0889 + 0.18 * (v_terminal_df - 4.2)**2  # estimation valid from 2.5 to 4.2 V (probably 2.0 to 4.3 V?)

    # better behavior towards high SoC:
    #   gallagherVolumeAveragedApproach2012: Gallagher K.G., Dees D.W., Jansen A.N., et al.:
    #   "A Volume Averaged Approach to the Numerical Modeling of Phase-Transition Intercalation Electrodes Presented
    #    for Li x C 6", J. Electrochem. Soc., vol. 159, no. 12, pp. A2029–A2037, 2012, DOI 10.1149/2.015301jes
    # I used Figure 3 (-> WebPlotDigitizer), set x=0.05 in Li_{x}C_{6} as SoC = 4 % and x = 0.96 as SoC = 100 %
    # for SoC < 4%, I used values from:
    #   schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
    #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
    # Then, using the OCV curve from measurements, I created a plot for V_anode(V_OCV)
    # Next, I used linear approximation functions to match the anode potential closely (particularly for  SoC > 5%)
    # use https://www.desmos.com/calculator to find limits
    V1 = 4.082
    # V1 = 4.055
    # V1 = 4.048
    V2 = 3.845
    # V2 = 3.858
    V3 = 3.668
    V4 = 3.536
    V5 = 3.372
    V6 = 3.259
    cond_a1 = (v_terminal_df > V1)
    cond_a2 = ((v_terminal_df > V2) & (v_terminal_df <= V1))
    cond_a3 = ((v_terminal_df > V3) & (v_terminal_df <= V2))
    cond_a4 = ((v_terminal_df > V4) & (v_terminal_df <= V3))
    cond_a5 = ((v_terminal_df > V5) & (v_terminal_df <= V4))
    cond_a6 = ((v_terminal_df > V6) & (v_terminal_df <= V5))
    cond_a7 = (v_terminal_df <= V6)
    v_anode = pd.Series(data=0.0, dtype=np.float64, index=v_terminal_df.index)
    v_anode[cond_a1] = 1.943616239 - 0.456980057 * v_terminal_df[cond_a1]
    # v_anode[cond_a1] = 1.785657673 - 0.42158516 * v_terminal_df[cond_a1]
    v_anode[cond_a2] = 0.078
    # v_anode[cond_a2] = 0.075
    v_anode[cond_a3] = 0.948586964 - 0.22641873 * v_terminal_df[cond_a3]
    v_anode[cond_a4] = 0.118
    v_anode[cond_a5] = 1.670224522 - 0.439009094 * v_terminal_df[cond_a5]
    v_anode[cond_a6] = 0.820011235 - 0.187182325 * v_terminal_df[cond_a6]
    v_anode[cond_a7] = 2.228263548 - 0.619305419 * v_terminal_df[cond_a7]
    return v_anode


# estimate cell anode potential from terminal voltage (input and output as DataFrame) - Version 3
def get_v_anode_from_v_terminal_df_v3(v_terminal_df):
    # # for an example for V_anode potential as a function of SoC, see:
    # #   schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
    # #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"

    # better behavior towards high SoC:
    #   gallagherVolumeAveragedApproach2012: Gallagher K.G., Dees D.W., Jansen A.N., et al.:
    #   "A Volume Averaged Approach to the Numerical Modeling of Phase-Transition Intercalation Electrodes Presented
    #    for Li x C 6", J. Electrochem. Soc., vol. 159, no. 12, pp. A2029–A2037, 2012, DOI 10.1149/2.015301jes
    # I used Figure 3 (-> WebPlotDigitizer), set x=0.05 in Li_{x}C_{6} as SoC = 0 % and x = 0.94 as SoC = 100 %
    # Then, using the OCV curve from measurements, I created a plot for V_anode(V_OCV)
    # Next, I used linear approximation functions to match the anode potential closely (particularly for  SoC > 10%)
    # use https://www.desmos.com/calculator to find limits
    V1 = 4.07
    V2 = 3.845
    V3 = 3.637
    V4 = 3.52
    V5 = 3.27
    cond_a1 = (v_terminal_df > V1)
    cond_a2 = ((v_terminal_df > V2) & (v_terminal_df <= V1))
    cond_a3 = ((v_terminal_df > V3) & (v_terminal_df <= V2))
    cond_a4 = ((v_terminal_df > V4) & (v_terminal_df <= V3))
    cond_a5 = ((v_terminal_df > V5) & (v_terminal_df <= V4))
    cond_a6 = (v_terminal_df <= V5)
    v_anode = pd.Series(data=0.0, dtype=np.float64, index=v_terminal_df.index)
    v_anode[cond_a1] = 1.094369231 - 0.249230769 * v_terminal_df[cond_a1]
    v_anode[cond_a2] = 0.08
    v_anode[cond_a3] = 0.76533838 - 0.177986907 * v_terminal_df[cond_a3]
    v_anode[cond_a4] = 0.118
    v_anode[cond_a5] = 1.229578947 - 0.315789474 * v_terminal_df[cond_a5]
    v_anode[cond_a6] = 0.606935065 - 0.125974026 * v_terminal_df[cond_a6]  # v_anode is underestimated vor V_cell < 2.5V
    return v_anode


# estimate cell anode potential from terminal voltage (input and output as DataFrame) - Version 4
def get_v_anode_from_v_terminal_df_v4(v_terminal_df):
    # # for an example for V_anode potential as a function of SoC, see:
    # #   schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
    # #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
    # return 0.0889 + 0.18 * (v_terminal_df - 4.2)**2  # estimation valid from 2.5 to 4.2 V (probably 2.0 to 4.3 V?)

    # better behavior towards high SoC:
    #   gallagherVolumeAveragedApproach2012: Gallagher K.G., Dees D.W., Jansen A.N., et al.:
    #   "A Volume Averaged Approach to the Numerical Modeling of Phase-Transition Intercalation Electrodes Presented
    #    for Li x C 6", J. Electrochem. Soc., vol. 159, no. 12, pp. A2029–A2037, 2012, DOI 10.1149/2.015301jes
    # I used Figure 3 (-> WebPlotDigitizer), set x=0.01 in Li_{x}C_{6} as SoC = 0 % and x = 0.96 as SoC = 100 %
    # for x < 0.04, I used values from:
    #   schimpeComprehensiveModelingTemperatureDependent2018: Schimpe, von Kuepach, Naumann, et al.:
    #   "Comprehensive Modeling of TemperatureDependent Degradation Mechanisms in Lithium Iron Phosphate Batteries"
    # Then, using the OCV curve from measurements, I created a plot for V_anode(V_OCV)
    # Next, I used linear approximation functions to match the anode potential closely (particularly for  SoC > 25%)
    V1 = 4.095
    V2 = 3.83
    V3 = 3.65
    V4 = 3.5
    cond_a1 = (v_terminal_df > V1)
    cond_a2 = ((v_terminal_df > V2) & (v_terminal_df <= V1))
    cond_a3 = ((v_terminal_df > V3) & (v_terminal_df <= V2))
    cond_a4 = ((v_terminal_df > V4) & (v_terminal_df <= V3))
    cond_a5 = (v_terminal_df <= V4)
    v_anode = pd.Series(data=0.0, dtype=np.float64, index=v_terminal_df.index)
    v_anode[cond_a1] = 2.5 - 0.59047619 * v_terminal_df[cond_a1]
    v_anode[cond_a2] = 0.082
    v_anode[cond_a3] = 0.890555556 - 0.211111111 * v_terminal_df[cond_a3]
    v_anode[cond_a4] = 0.12
    v_anode[cond_a5] = 2.15 - 0.58 * v_terminal_df[cond_a5]  # estimation not good for SoC < 25%
    return v_anode


if __name__ == "__main__":
    run()
