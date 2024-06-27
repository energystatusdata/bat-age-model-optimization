# Automatic optimization of battery aging models according to aging data, using the scipy library, as described in:
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
# They were derived with the manual/semi-automatic "optimize_models_manual.py".
# The model and the variables are explained in Chapter 7.2 of the dissertation.
# Note: Only models 1-14 are implemented in the "optimize_models_automatic.py" file.
#
# ToDo: to start, you need to download the CFG and the LOG_AGE .csv files from RADAR4KIT (see above), e.g.:
#   cell_log_age_30s_P012_3_S14_C11.csv --> found in the "cell_log_age.7z" (e.g., use 7-Zip to unzip)
#   cell_cfg_P012_3_S14_C11.csv --> found in the "cfg.zip"


import time
# import gc
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
from scipy import optimize
import math
# import pyarrow as pa
from enum import IntEnum
import config_labels as csv_label
import config_logging  # as logging


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
error_calc_method = error_calculation_method.RMSE_PER_PARAMETER  # Recommended: RMSE_PER_PARAMETER

# ToDo: which aging types from the experiment shall be considered in optimization?
CELL_PARAMETERS_USE = list(range(0, 17))  # all calendar aging cells
# CELL_PARAMETERS_USE = list(range(0, 65))  # all calendar + cyclic aging cells
# CELL_PARAMETERS_USE = list(range(17, 65))  # all cyclic aging cells
# CELL_PARAMETERS_USE = list(range(65, 77))  # all profile aging cells
# CELL_PARAMETERS_USE = list(range(0, 77))  # all calendar + cyclic + profile aging cells

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

CHUNK_DURATION = 2 * 24 * 60 * 60  # 2 days, process aging data in chunks of this duration if aging is iterative
# for models 7 -14, some data is processed in "chunks" -> speeds up simulation. Too large chunks reduce model accuracy.

# --- plot output format ---
SHOW_IN_BROWSER = True  # open interactive plots in the browser NOW
EXPORT_HTML = True  # save interactive plots as html
EXPORT_IMAGE = True  # static plot images instead of interactive plots
REQUIRE_CFG = True  # probably can also be set to False if neither SHOW_IN_BROWSER / EXPORT_HTML / EXPORT_IMAGE enabled.
#                     if you use this script to fit other data: generate a CFG file for them, or set REQUIRE_CFG = False
# IMAGE_FORMAT = "jpg"  # -> tends to be larger than png
IMAGE_FORMAT = "png"  # -> easiest to view
# IMAGE_FORMAT = "svg"  # for EOC plots, this is smaller than png, and crisper of course
# IMAGE_FORMAT = "pdf"  # for EOC plots, this is by far the smallest, HOWEVER, there are tiny but annoying grid lines

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
SUBPLOT_H_SPACING_REL = 0.25  # 0.2  # 0.12  # 0.03, was 0.04
SUBPLOT_V_SPACING_REL = 0.3  # 0.35  # 0.21  # was 0.035
SUBPLOT_LR_MARGIN = 10
SUBPLOT_TOP_MARGIN = 120  # 0
SUBPLOT_BOT_MARGIN = 0
SUBPLOT_PADDING = 0

HEIGHT_PER_ROW = 230  # in px
PLOT_WIDTH = 1850  # in px
# PLOT_HEIGHT = HEIGHT_PER_ROW * SUBPLOT_ROWS -> we need to figure this out dynamically for each plot

PLOT_TITLE_Y_POS_REL = 30.0
AGE_TYPE_TITLES = {cfg.age_type.CALENDAR: "calendar aging",
                   cfg.age_type.CYCLIC: "cyclic aging",
                   cfg.age_type.PROFILE: "profile aging"
                   }

TIME_DIV = 24.0 * 60.0 * 60.0
TIME_UNIT = "days"
X_AXIS_TITLE = 'Time [' + TIME_UNIT + ']'
Y_AXIS_TITLE = "usable dischg. capacity [Ah]"
TRACE_NAME_CU = f"P%03u-%01u (S%02u:C%02u)"
TRACE_NAME_MDL = f"%s for P%03u-%01u"


# figure settings
FIGURE_TEMPLATE = "custom_theme"  # "custom_theme" "plotly_white" "plotly" "none"
PLOT_LINE_OPACITY = 0.8
# PLOT_HOVER_TEMPLATE_1 = "<b>%s</b><br>"
# PLOT_HOVER_TEMPLATE_2 = "X %{x:.2f}, Y: %{y:.2f}<br><extra></extra>"
PLOT_HOVER_TEMPLATE_CU = "<b>%{text}</b><br>Remaining usable discharge capacity: %{y:.4f} Ah<br><extra></extra>"
PLOT_HOVER_TEMPLATE_MDL = "<b>after %{x:.1f} " + TIME_UNIT + ("</b><br>Remaining usable discharge capacity:"
                                                                   " %{y:.4f} Ah<br><extra></extra>")
PLOT_TEXT_CU = f"CU #%u after %.1f " + TIME_UNIT + ", %.2f EFCs"
PLOT_TEXT_MDL = f"after %.1f " + TIME_UNIT + ", %.2f EFCs"
TEXT_RMSE = f"%.4f %%"
TEXT_RMSE_CELL = f"RMSE cell %u: " + TEXT_RMSE
TEXT_RMSE_PARAM = f"RMSE parameter: " + TEXT_RMSE
TEXT_RMSE_TOTAL = f"RMSE total: " + TEXT_RMSE
TEXT_POS_RMSE_X = 0.92  # 0.02
TEXT_POS_RMSE_Y_BASE = 0.95  # 0.02
TEXT_POS_RMSE_DY = 0.08  # 0.08
TEXT_POS_RMSE_DY_OFFSET = -1  # 3
TEXT_POS_RMSE_DY_FACTOR = -1  # ?

BG_COLOR = '#fff'
MAJOR_GRID_COLOR = '#bbb'
MINOR_GRID_COLOR = '#e8e8e8'  # '#ddd'

TRACE_OPACITY = 0.4
TRACE_LINE_WIDTH = 1.5
TRACE_COLORS = ['rgb(239,65,54)', 'rgb(1,147,70)', 'rgb(43,111,183)']  # for parameter_nr 1, 2, 3 (here: R, G, B)
MARKER_OPACITY = 0.8  # 75
MARKER_STYLE = dict(size=5, opacity=MARKER_OPACITY, line=None, symbol='circle')

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
pio.templates['custom_theme']['layout']['yaxis']['title']['standoff'] = 10


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
# ToDo: select numbers of CPU cors used for optimization (to run parallel instances of optimization, e.g., to compare
#  different models or optimizers)
NUMBER_OF_PROCESSORS_TO_USE = 1  # only use one core
# NUMBER_OF_PROCESSORS_TO_USE = math.ceil(multiprocessing.cpu_count() / 4)  # use 25% of the cores
# NUMBER_OF_PROCESSORS_TO_USE = math.ceil(multiprocessing.cpu_count() / 2)  # use 50% of the cores
# NUMBER_OF_PROCESSORS_TO_USE = max(multiprocessing.cpu_count() - 1, 1)  # leave one free
# NUMBER_OF_PROCESSORS_TO_USE = multiprocessing.cpu_count()  # use all cores

# select numbers of CPU cors used for optimizing AN INDIVIDUAL MODEL (only model_f014, but may be adopted to others)
# NUMBER_OF_PROCESSORS_TO_USE_MDL = max(multiprocessing.cpu_count() - 1, 1)  # leave one free
NUMBER_OF_PROCESSORS_TO_USE_MDL = NUMBER_OF_PROCESSORS_TO_USE

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
DF_COL_AGE_CAP_DELTA_FILL = 0
DF_COL_AGE_MODEL_FILL = np.nan

# run_timestring = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

AGE_TYPE_FILENAMES = {cfg.age_type.CALENDAR: 'CAL',
                      cfg.age_type.CYCLIC: 'CYC',
                      cfg.age_type.PROFILE: 'PRF'}

LABEL_C_CHG = csv_label.I_CELL + "_chg_rel"

logging_filename = "09_log.txt"
logging = config_logging.bat_data_logger(cfg.LOG_DIR + logging_filename)

# tweaks to make it more likely that large image exports work:
pio.orca.config.executable = cfg.ORCA_PATH
# pio.kaleido.scope.mathjax = None
# pio.kaleido.scope.chromium_args += ("--single-process",)
pio.orca.config.timeout = 600  # increase timeout from 30 seconds to 10 minutes

modeling_task_queue = multiprocessing.Queue()
optimizer_task_queue = multiprocessing.Queue()


# exceptions
class ProcessingFailure(Exception):
    pass


# main function
def run():
    start_timestamp = datetime.now()
    logging.log.info(os.path.basename(__file__))
    # report_queue = multiprocessing.Queue()
    report_manager = multiprocessing.Manager()
    report_queue = report_manager.Queue()

    optimizer_main(report_queue)  # <-- this is the "actual" function doing the work

    logging.log.info("\n\n========== All tasks ended - summary ==========\n")

    while True:
        if (report_queue is None) or report_queue.empty():
            break  # no more reports

        try:
            slave_report = report_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break  # no more reports

        if slave_report is None:
            break  # no more reports

        report_msg = slave_report["msg"]
        report_level = slave_report["level"]

        if report_level == config_logging.ERROR:
            logging.log.error(report_msg)
        elif report_level == config_logging.WARNING:
            logging.log.warning(report_msg)
        elif report_level == config_logging.INFO:
            logging.log.info(report_msg)
        elif report_level == config_logging.DEBUG:
            logging.log.debug(report_msg)
        elif report_level == config_logging.CRITICAL:
            logging.log.critical(report_msg)

    stop_timestamp = datetime.now()
    logging.log.info("\nScript runtime: %s h:mm:ss.ms" % str(stop_timestamp - start_timestamp))


# main optimization function
def optimizer_main(report_queue):
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

    # List found slaves/cells for user
    pre_text = ("Found the following files:\n"
                "' ' = no file found, 'l' = LOG_AGE found, 'c' = config found, 'b' = both found,\n"
                "'X' = LOG_AGE and config of interest found & matching -> added\n")
    logging.log.info(ht.get_found_cells_text(slave_cell_found, pre_text))

    # read files and store in df
    logging.log.info("Reading LOG_AGE files...")
    log_dfs = [[pd.DataFrame() for _ in range(num_cells_per_parameter)] for _ in range(num_parameters)]
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

            log_dfs[param_id - 1][param_nr - 1] = log_df
    logging.log.info("...done reading LOG_AGE files")

    # generate base plots
    logging.log.info("generating plot templates")
    fig_list, fig_and_sp_from_pid_arr, param_df = generate_base_figures(
        cell_log_csv, slave_cell_found, num_parameters, num_cells_per_parameter)
    fig_list = fill_figures_with_measurements(fig_list, fig_and_sp_from_pid_arr, log_dfs, param_df)

    # # for debugging:
    # age_type_show = [cfg.age_type.CALENDAR, cfg.age_type.CYCLIC, cfg.age_type.PROFILE]
    # for age_type in age_type_show:
    #     if fig_list[age_type] is not None:
    #         # noinspection PyTypeChecker
    #         this_fig: go.Figure = fig_list[age_type]
    #         this_fig.show()

    # ToDo: here you can select one (or multiple) optimization methods supported by scipy - you can also leave it as is
    # 'L-BFGS-B' seems to be particularly fast?
    # 'Nelder-Mead' seems also ok
    # 'Powell' is a bit lost?
    # 'TNC' also not so fast
    # 'SLSQP' failed: "Inequality constraints incompatible"
    # 'trust-constr' seems to ignore constraints?
    # https://stackoverflow.com/a/64004381/2738240 : "If you have noisy measurements: Use Nelder-Mead or Powell."
    # opt_methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']
    # 'CG', 'BFGS', 'Newton-CG', 'COBYLA', 'trust-ncg', 'dogleg', 'trust-exact', 'trust-krylov' don't work with bounds
    # model_f001: 'Nelder-Mead', 'Powell' -> work well
    # model_f002: 'Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP' -> not really fitted
    # opt_methods = ['Nelder-Mead', 'L-BFGS-B']
    # 0 'Nelder-Mead': 1559 + 2480 + 2995 + 3108 + 5953 + 6805 = 22900 -> 6,4 h
    # 1 'L-BFGS-B': 2316 + 3154 + 3701 + 4650 + 4482 + 4817 = 23120 -> 6,4 h

    # === ToDo: Add optimizing functions to task queue =================================================================
    # optimizer_entry[OPT_FUNC_ID] = 1                         # running number, unique identifier
    # optimizer_entry[OPT_FUNC_NAME] = model_f001              # python function name -> you can also implement own ones
    # optimizer_entry[OPT_VARS] = {"a0": 1, "a1": 1, "a2": 1}  # variables and their initial values
    # optimizer_entry[OPT_VAR_LIMS] = {"a0": (0.5, 1.5), ...}  # boundaries for variables
    # optimizer_entry[OPT_METHOD] = 'Nelder-Mead'              # optimization method, see above

    optimizer_entry = {OPT_FUNC_ID: 1400, OPT_FUNC_NAME: model_f014,
                       OPT_VARS: {"a0": 1.0e-9,
                                  "a1": -2500,
                                  "a2": 2.0,
                                  "a3": 0.5e-7,
                                  "a4": 3.0e-7,
                                  "a5": 4.0e-6,
                                  "a6": 0.12,
                                  "a7": 0.052,
                                  "a8": 3.0e-7
                                  },
                       OPT_VAR_LIMS: {"a0": (0.8e-9, 1.2e-9),
                                      "a1": (-3000, -2000),
                                      "a2": (1.5, 2.5),
                                      "a3": (0.3e-7, 0.7e-7),
                                      "a4": (2.0e-7, 4.0e-7),
                                      "a5": (3.0e-6, 5.0e-6),
                                      "a6": (0.06, 0.24),
                                      "a7": (0.02, 0.08),
                                      "a8": (1.0e-7, 1.0e-6),
                                      },
                       OPT_METHOD: 'Nelder-Mead',
                       OPT_USE_CAP_NOM: -0.1,
                       }
    optimizer_task_queue.put(optimizer_entry)

    # You can add others if you want.
    pass

    # add others...
    pass

    # You can also generate the optimizer_task_queue entries programmatically. For example, I used these in the past:
    # opt_methods = ['Nelder-Mead']
    # opt_use_cap_nom_arr = [0, 1, 0, 1, 0.5]
    # opt_use_t_prod__arr = [True, True, False, False, False]
    # # opt_use_cap_nom_arr = [False, True]
    # # opt_use_t_prod__arr = [True, True]

    # # opt_use_cap_nom_only_arr = [0, 1, 0.5]
    # opt_use_cap_nom_only_arr = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]

    # for i in range(len(opt_methods)):
    #     for j in range(len(opt_use_cap_nom_only_arr)):
    #         opt_a = opt_use_cap_nom_only_arr[j]
    #         optimizer_entry = {OPT_FUNC_ID: 6000 + j * 10 + i, OPT_FUNC_NAME: model_f006,
    #                            OPT_VARS: {"a0": 0.75e-8, "a1": -11000, "a2": 0.15, "a3": 1.95e-8, "a4": 7.0e-7},
    #                            OPT_VAR_LIMS: {"a0": (0.25e-8, 1.25e-8),
    #                                           "a1": (-20000, -5000),
    #                                           "a2": (0.1, 0.2),
    #                                           "a3": (1e-8, 3e-8),
    #                                           "a4": (1.0e-7, 1.5e-6)},
    #                            OPT_METHOD: opt_methods[i], OPT_USE_CAP_NOM: opt_a,
    #                            }
    #         optimizer_task_queue.put(optimizer_entry)

    # for i in range(len(opt_methods)):
    #     for j in range(len(opt_use_cap_nom_only_arr)):
    #         opt_a = opt_use_cap_nom_only_arr[j]
    #         optimizer_entry = {OPT_FUNC_ID: 650 + j * 10 + i, OPT_FUNC_NAME: model_f005,
    #                            OPT_VARS: {"a00": 0.000000017, "a4": 0.00000006},
    #                            OPT_VAR_LIMS: {"a0": (0.00000001, 0.00000003), "a4": (0.00000001, 0.0000002)},
    #                            OPT_METHOD: opt_methods[i], OPT_USE_CAP_NOM: opt_a,
    #                            }
    #         optimizer_task_queue.put(optimizer_entry)

    # for i in range(len(opt_methods)):
    #     for j in range(len(opt_use_cap_nom_only_arr)):
    #         opt_a = opt_use_cap_nom_only_arr[j]
    #         optimizer_entry = {OPT_FUNC_ID: 600 + j * 10 + i, OPT_FUNC_NAME: model_f004,
    #                            OPT_VARS: {"a0": 0.00000001, "a1": 0.01, "a2": 0.5, "a3": 0.0001, "a4": 0.004},
    #                            OPT_VAR_LIMS: {"a0": (0.000000001, 0.0001),
    #                                           "a1": (0.001, 100000.0), "a2": (0.01, 10.0), "a3": (0.00001, 10.0),
    #                                           "a4": (0.0000001, 0.1)},
    #                            OPT_METHOD: opt_methods[i], OPT_USE_CAP_NOM: opt_a,
    #                            }
    #         optimizer_task_queue.put(optimizer_entry)
    #
    # for i in range(len(opt_methods)):
    #     for j in range(len(opt_use_cap_nom_only_arr)):
    #         opt_a = opt_use_cap_nom_only_arr[j]
    #         optimizer_entry = {OPT_FUNC_ID: 550 + j * 10 + i, OPT_FUNC_NAME: model_f005,
    #                            OPT_VARS: {"a00": 0.00000001, "a4": 0.004},
    #                            OPT_VAR_LIMS: {"a00": (0.000000001, 0.1), "a4": (0.0000001, 0.1)},
    #                            OPT_METHOD: opt_methods[i], OPT_USE_CAP_NOM: opt_a,
    #                            }
    #         optimizer_task_queue.put(optimizer_entry)

    # # a_0*exp(a_1*T)*exp(a_2*V)
    # for i in range(len(opt_methods)):
    #     for j in range(len(opt_use_cap_nom_arr)):
    #         opt_a = opt_use_cap_nom_arr[j]
    #         opt_b = opt_use_t_prod__arr[j]
    #         optimizer_entry = {OPT_FUNC_ID: 300 + j * 10 + i, OPT_FUNC_NAME: model_f003,
    #                            OPT_VARS: {"a0": 0.00005, "a1": 0.3, "a2": 0.0001},
    #                            OPT_VAR_LIMS: {"a0": (0.000001, 10.0), "a1": (0.01, 0.6), "a2": (0.00001, 10.0)},
    #                            OPT_METHOD: opt_methods[i], OPT_USE_CAP_NOM: opt_a, OPT_USE_T_PRODUCTION: opt_b,
    #                            }
    #         optimizer_task_queue.put(optimizer_entry)
    #         # 300 ('Nelder-Mead'):  FAIL  984 seconds
    #         # 301 ('Powell'):       x
    #         # 302 ('L-BFGS-B'):     FAIL  1097 seconds
    #         # 303 ('TNC'):          FAIL  5307 seconds
    #         # 304 ('SLSQP'):        FAIL  591 seconds
    #         # 305 ('trust-constr'): x
    #
    # # eckerDevelopmentLifetimePrediction2012 - original form:
    # # Ecker et al.: "Development of a lifetime prediction model for lithium-ion batteries based on extended
    # #                accelerated aging test data" (2012)
    # for i in range(len(opt_methods)):
    #     for j in range(len(opt_use_cap_nom_arr)):
    #         opt_a = opt_use_cap_nom_arr[j]
    #         opt_b = opt_use_t_prod__arr[j]
    #         optimizer_entry = {OPT_FUNC_ID: 200 + j * 10 + i, OPT_FUNC_NAME: model_f002,
    #                            OPT_VARS: {"a0": 0.00005, "a1": 1.046, "a2": 1.134},
    #                            # OPT_VAR_LIMS: {"a0": (0.0001, 0.1), "a1": (0.5, 2.0), "a2": (0.5, 2.0)},
    #                            OPT_VAR_LIMS: {"a0": (0.00001, 0.1), "a1": (0.7, 2.0), "a2": (0.9, 4.0)},
    #                            OPT_METHOD: opt_methods[i], OPT_USE_CAP_NOM: opt_a, OPT_USE_T_PRODUCTION: opt_b,
    #                            }
    #         optimizer_task_queue.put(optimizer_entry)
    #         # 200 ('Nelder-Mead'):  FAIL  984 seconds
    #         # 201 ('Powell'):       x
    #         # 202 ('L-BFGS-B'):     FAIL  1097 seconds
    #         # 203 ('TNC'):          FAIL  5307 seconds
    #         # 204 ('SLSQP'):        FAIL  591 seconds
    #         # 205 ('trust-constr'): x
    #
    # # eckerDevelopmentLifetimePrediction2012 - generic form:
    # # Ecker et al.: "Development of a lifetime prediction model for lithium-ion batteries based on extended
    # #                accelerated aging test data" (2012)
    # for i in range(len(opt_methods)):
    #     for j in range(len(opt_use_cap_nom_arr)):
    #         opt_a = opt_use_cap_nom_arr[j]
    #         opt_b = opt_use_t_prod__arr[j]
    #         optimizer_entry = {OPT_FUNC_ID: 100 + j * 10 + i, OPT_FUNC_NAME: model_f001,
    #                            OPT_VARS: {"a0": 0.0001, "a1": 1.068, "a2": 0.991},
    #                            OPT_VAR_LIMS: {"a0": (0.00001, 1.0), "a1": (0.7, 1.3), "a2": (0.96, 1.1)},
    #                            OPT_METHOD: opt_methods[i], OPT_USE_CAP_NOM: opt_a, OPT_USE_T_PRODUCTION: opt_b,
    #                            }  # take care: both 0 and 1 will fail!
    #         optimizer_task_queue.put(optimizer_entry)
    #         # 10? (default):        OK  5558-5975 seconds
    #         # 100 ('Nelder-Mead'):  x
    #         # 101 ('Powell'):       x
    #         # 102 ('L-BFGS-B'):     x
    #         # 103 ('TNC'):          x
    #         # 104 ('SLSQP'):        x
    #         # 105 ('trust-constr'): x

    total_queue_size = optimizer_task_queue.qsize()

    # Create processes
    processes = []
    logging.log.info("Starting processes to optimize...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        logging.log.debug("  Starting process %u" % processorNumber)
        processes.append(multiprocessing.Process(target=optimizer_thread,
                                                 args=(processorNumber, optimizer_task_queue, log_dfs, report_queue,
                                                       total_queue_size, fig_list, fig_and_sp_from_pid_arr)))
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE):
        processes[processorNumber].join()
        logging.log.debug("Joined process %u" % processorNumber)


def optimizer_thread(processor_number, task_queue, log_dfs, thread_report_queue, total_queue_size,
                     fig_list, fig_and_sp_from_pid_arr):
    while True:
        if (task_queue is None) or task_queue.empty():
            break  # no more files

        try:
            remaining_size = task_queue.qsize()
            queue_entry = task_queue.get_nowait()
        except multiprocessing.queues.Empty:  # except multiprocessing.queue.Empty:
            break  # no more files

        if queue_entry is None:
            break  # no more files

        mdl_id = queue_entry[OPT_FUNC_ID]
        mdl_name = queue_entry[OPT_FUNC_NAME]
        variables = queue_entry[OPT_VARS]
        variable_bounds = queue_entry[OPT_VAR_LIMS]
        opt_method = None
        if OPT_METHOD in queue_entry:
            opt_method = queue_entry[OPT_METHOD]
        use_cap_nom = USE_NOMINAL_CAPACITY_DEFAULT
        if OPT_USE_CAP_NOM in queue_entry:
            use_cap_nom = queue_entry[OPT_USE_CAP_NOM]
        use_t_prod = USE_T_PRODUCTION_DEFAULT
        if OPT_USE_T_PRODUCTION in queue_entry:
            use_t_prod = queue_entry[OPT_USE_T_PRODUCTION]

        # # if not (mdl_id == 2):
        # if ((mdl_id == 100) or (mdl_id == 101) or (mdl_id == 110) or (mdl_id == 111)
        #         or (mdl_id == 200) or (mdl_id == 201) or (mdl_id == 210) or (mdl_id == 211)
        #         or (mdl_id == 300) or (mdl_id == 301) or (mdl_id == 310) or (mdl_id == 311)
        #         or (mdl_id == 120) or (mdl_id == 130) or (mdl_id == 220) or (mdl_id == 230)
        #         or (mdl_id == 320) or (mdl_id == 330)
        #         or (mdl_id == 240) or (mdl_id == 340)):
        #     continue  # for debugging individual optimizing functions

        progress = 0.0
        if total_queue_size > 0:
            progress = (1.0 - remaining_size / total_queue_size) * 100.0
        logging.log.info("Thread %u model function ID %u - start optimizing (progress: %.1f %%)"
                         % (processor_number, mdl_id, progress))

        num_infos = 0
        num_warnings = 0
        num_errors = 0
        rmse_total = np.nan
        result_filename = "no result file!"
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

            # prepare optimization
            # model = Model(function_name)
            # params = model.make_params(**variables)

            # run optimization
            # result = model.fit(0, params, log_age_dfs=log_dfs)
            # variables = result.params.valuesdict()
            t_read_start = datetime.now()
            res = optimize.minimize(mdl_name, np.array(list(variables.values())), method=opt_method,
                                    bounds=np.array(list(variable_bounds.values())),
                                    args=(log_dfs, mdl_id, use_cap_nom, use_t_prod, False, None, None))
            t_read_stop = datetime.now()
            dt = t_read_stop - t_read_start
            dts = dt.total_seconds()

            result_variables = res.x
            status = res.status
            msg = res.message
            if res.success:
                write_msg = "successfully optimized after %u seconds with code %u: %s" % (dts, status, msg)
                logging.log.info("Thread %u model function ID %u %s" % (processor_number, mdl_id, write_msg))
            else:
                write_msg = "optimization failed after %u seconds with code %u: %s" % (dts, status, msg)
                logging.log.warning("Thread %u model function ID %u %s" % (processor_number, mdl_id, write_msg))

            for i, key in enumerate(variables):
                variables[key] = result_variables[i]

            # write results to plot and (optionally) show
            rmse_total = mdl_name(np.array(list(variables.values())), log_dfs, mdl_id, use_cap_nom, use_t_prod,
                                  True, fig_list_copy, fig_and_sp_from_pid_arr)

            # write results to file...
            algo = "default ('none')"
            if opt_method is not None:
                algo = f"'%s'" % opt_method
            settings_string = (f"using %s algorithm\nbounds: %s\nuse_cap_nom: %.2f\tuse_t_prod: %u\n\n"
                               f"RMSE_total: %.4f %%\n\n"
                               % (algo, str(variable_bounds), use_cap_nom, int(use_t_prod), rmse_total * 100.0))
            run_timestring = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # use unique string so we don't overwrite
            result_filename = cfg.MODEL_RESULT_FILENAME % (mdl_name.__name__, mdl_id, run_timestring)
            if not os.path.exists(cfg.MODEL_OUTPUT_DIR):
                os.mkdir(cfg.MODEL_OUTPUT_DIR)
            with open(cfg.MODEL_OUTPUT_DIR + result_filename, "w+") as result_file:
                result_file.write("%s (%u)\n" % (mdl_name.__name__, mdl_id))
                result_file.write(write_msg + "\n")
                result_file.write(settings_string + "\n")
                for var_name, value in variables.items():
                    result_file.write("%s\t%.24f\n" % (var_name, value))
                result_file.close()

            logging.log.debug("Thread %u model function ID %u - optimizing complete (%.0f seconds)"
                              % (processor_number, mdl_id, dts))

        except ProcessingFailure:
            # logging.log.warning("Thread %u model function ID %u - optimizing failed!"
            #                     % (processor_number, function_id))
            pass
        except Exception:
            logging.log.error("Thread %u model function ID %u - Python Error:\n%s"
                              % (processor_number, mdl_id, traceback.format_exc()))

        # we land here on success or any error

        # reporting to main thread
        report_msg = (f"%s - model function ID %u - optimizing finished: %u infos, %u warnings, %u errors - RMSE: "
                      f"%.4f %%" % (result_filename, mdl_id, num_infos, num_warnings, num_errors, rmse_total * 100.0))
        report_level = config_logging.INFO
        if num_errors > 0:
            report_level = config_logging.ERROR
        elif num_warnings > 0:
            report_level = config_logging.WARNING

        cell_report = {"msg": report_msg, "level": report_level}
        thread_report_queue.put(cell_report)

    task_queue.close()
    # thread_report_queue.close()
    logging.log.info("Thread %u - no more tasks - exiting" % processor_number)


# ToDo: here comes a long list with all models tested...
#   model_f... is called by the optimizer
#   model_f..._get_delta_age is called for each parameter set id (operating condition) and nr (cells tested with same
#      parameter id). This is also, where the actual aging function is implemented.
#   prepare_model --> same for all: initialize variables
#   prepare_cell_log_df --> same for all: prepare cell log data frame
#   add_model_trace --> same for all: add the lines to the figure (if plotting is enabled)
#   add_model_param_rmse --> same for all: add RMSE to figure (if plotting is enabled)
#   plot_model --> same for all: plot the model result
#   ...
#
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

    rmse_total, num_rmse_points_total = 0, 0
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
    var_string = ("a0 = %.24f, a1 = %.24f, a2 = %.24f" % (a0, a1, a2))
    if plot:
        plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string)
    logging.log.debug("%s (%u) call with %s results in rmse_total = %.10f %%"
                      % (mdl_name, mdl_id, var_string, rmse_total * 100.0))
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

    rmse_total, num_rmse_points_total = 0, 0
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
    logging.log.debug("%s (%u) call with %s results in rmse_total = %.10f %%"
                      % (mdl_name, mdl_id, var_string, rmse_total * 100.0))
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

    rmse_total, num_rmse_points_total = 0, 0
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
    logging.log.debug("%s (%u) call with %s results in rmse_total = %.10f %%"
                      % (mdl_name, mdl_id, var_string, rmse_total * 100.0))
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

    rmse_total, num_rmse_points_total = 0, 0
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
    logging.log.debug("%s (%u) call with %s results in "
                      "rmse_total = %.10f %%" % (mdl_name, mdl_id, var_string, rmse_total * 100.0))
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

    rmse_total, num_rmse_points_total = 0, 0
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
    logging.log.debug("%s (%u) call with %s results in "
                      "rmse_total = %.10f %%" % (mdl_name, mdl_id, var_string, rmse_total * 100.0))
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

    rmse_total, num_rmse_points_total = 0, 0
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
    logging.log.debug("%s (%u) call with %s results in "
                      "rmse_total = %.10f %%" % (mdl_name, mdl_id, var_string, rmse_total * 100.0))
    return rmse_total


def model_f006_get_delta_age(log_df, a0, a1, a2, a3, a4):
    # calculate delta of aged_cap f_age(...)
    t_func = get_t_func_diff(log_df[DF_COL_TIME_USE])  # delta_t

    V0 = 3.6
    T0 = cfg.T0 + 25.0
    label_I_chg = csv_label.I_CELL + "_chg"

    log_df[label_I_chg] = 0.0
    cond_chg = (log_df[csv_label.I_CELL] > cfg.CELL_CAPACITY_NOMINAL / 1000.0)  # if current >1/1000 C. 80 kWh bat: 80 W
    # cond_chg = (log_df[csv_label.I_CELL] > 0)  # if current > 0
    log_df.loc[cond_chg, label_I_chg] = log_df[csv_label.I_CELL]

    sei_potential = (  # log_df["model_f006_SEI_potential"] = (
            a0
            * np.exp(a1 * (1.0/log_df[csv_label.T_CELL].astype(np.float64) - 1.0/T0))
            * np.exp(a2 * (log_df[csv_label.V_CELL].astype(np.float64) - V0))
            + a3 * log_df[label_I_chg] * t_func  # a3 * dQ_chg
            # * np.exp(a3 * log_df[label_I_chg])
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
        logging.log.warning("model_f006 (???) call with a0 = %.12f, a1 = %.12f, a2 = %.12f, a3 = %.12f, a4 = %.12f "
                            "caused a RuntimeWarning!" % (a0, a1, a2, a3, a4))
        dq_loss_sei_np = sei_potential_np * t_func_np

    return dq_loss_sei_np
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

    rmse_total, num_rmse_points_total = 0, 0
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
    logging.log.debug("%s (%u) call with %s results"
                      " in rmse_total = %.10f %%" % (mdl_name, mdl_id, var_string, rmse_total * 100.0))
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
        CHUNK_SIZE = int(CHUNK_DURATION / 30)  # -> equals 24 hours for 30 second resolution -> 1x per day
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

    rmse_total, num_rmse_points_total = 0, 0
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
    logging.log.debug("%s (%u) call with %s results in "
                      "rmse_total = %.10f %%" % (mdl_name, mdl_id, var_string, rmse_total * 100.0))
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
        CHUNK_SIZE = int(CHUNK_DURATION / 30)  # -> equals 24 hours for 30 second resolution -> 1x per day
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

    rmse_total, num_rmse_points_total = 0, 0
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
    logging.log.debug("%s (%u) call with %s\n   results in "
                      "rmse_total = %.10f %%" % (mdl_name, mdl_id, var_string, rmse_total * 100.0))
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
        CHUNK_SIZE = int(CHUNK_DURATION / 30)  # -> equals 24 hours for 30 second resolution -> 1x per day
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

    rmse_total, num_rmse_points_total = 0, 0
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
    logging.log.debug("%s (%u) call with %s\n   results in "
                      "rmse_total = %.10f %%" % (mdl_name, mdl_id, var_string, rmse_total * 100.0))
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

    cyclic_age_weardown = a6 * log_df[LABEL_C_CHG] * t_func  # a6 * dQ_chg

    # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
    # it is a differential equation, I don't know a way how to solve it with pure pandas.
    # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
    # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
    try:
        # faster accumulation using DataFrame chunks
        CHUNK_SIZE = int(CHUNK_DURATION / 30)  # -> equals 24 hours for 30 second resolution -> 1x per day
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

    return (dq_loss + cyclic_age_weardown) * cfg.CELL_CAPACITY_NOMINAL
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

    rmse_total, num_rmse_points_total = 0, 0
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
    logging.log.debug("%s (%u) call with %s\n   results in "
                      "rmse_total = %.10f %%" % (mdl_name, mdl_id, var_string, rmse_total * 100.0))
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

    cyclic_age_weardown = a6 * log_df[LABEL_C_CHG] * t_func  # a6 * dQ_chg

    # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
    # it is a differential equation, I don't know a way how to solve it with pure pandas.
    # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
    # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
    try:
        # faster accumulation using DataFrame chunks
        CHUNK_SIZE = int(CHUNK_DURATION / 30)  # -> equals 24 hours for 30 second resolution -> 1x per day
        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        df_length = sei_potential.shape[0]
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        i_start = 0
        while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
            i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end is not included when slicing df.iloc[i_start:i_end]
            # diff_a = (sei_potential.iloc[i_start:i_end] - a4 * q_loss_sei_total)
            diff_a = (sei_potential.iloc[i_start:i_end]
                      - a4 * q_loss_sei_total / (1.0 + cyclic_age_weardown.iloc[i_start:i_end]))
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

    return (dq_loss + cyclic_age_weardown) * cfg.CELL_CAPACITY_NOMINAL
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

    rmse_total, num_rmse_points_total = 0, 0
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
    logging.log.debug("%s (%u) call with %s\n   results in "
                      "rmse_total = %.10f %%" % (mdl_name, mdl_id, var_string, rmse_total * 100.0))
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

    dq_cyclic_age_weardown = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
    q_cyclic_age_weardown = dq_cyclic_age_weardown.cumsum()

    # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
    # it is a differential equation, I don't know a way how to solve it with pure pandas.
    # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
    # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
    try:
        # faster accumulation using DataFrame chunks
        CHUNK_SIZE = int(CHUNK_DURATION / 30)  # -> equals 24 hours for 30 second resolution -> 1x per day
        q_loss_sei_total = 0
        q_loss_cyclic_total = 0
        df_length = sei_potential.shape[0]
        dq_loss = pd.Series(data=0.0, dtype=np.float64, index=range(0, df_length))
        i_start = 0
        while i_start < (df_length - 1):  # df.iloc[i_start:i_end] -> i_end should be at least i_start + 1
            i_end = min(i_start + CHUNK_SIZE, df_length)  # i_end is not included when slicing df.iloc[i_start:i_end]
            # diff_a = (sei_potential.iloc[i_start:i_end] - a3 * q_loss_sei_total)
            diff_a = (sei_potential.iloc[i_start:i_end]
                      - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_weardown.iloc[i_start:i_end]))
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

    return (dq_loss + dq_cyclic_age_weardown) * cfg.CELL_CAPACITY_NOMINAL
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

    rmse_total, num_rmse_points_total = 0, 0
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
    logging.log.debug("%s (%u) call with %s\n   results in "
                      "rmse_total = %.10f %%" % (mdl_name, mdl_id, var_string, rmse_total * 100.0))
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

    # FIXME: consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
    #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current collector
    #  corrosion? structural disordering? loss of electrical contact?)

    # dq_cyclic_age_weardown = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
    # q_cyclic_age_weardown = dq_cyclic_age_weardown.cumsum()

    # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
    # it is a differential equation, I don't know a way how to solve it with pure pandas.
    # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
    # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
    # faster accumulation using DataFrame chunks
    if t_func.shape[0] > 1:
        dt = t_func.max()
        if (dt == 2) or (dt == 30):
            CHUNK_SIZE = int(CHUNK_DURATION / dt)  # -> equals 24 hours -> 1x per day
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)  # -> equals 24 hours for 30 second resolution -> 1x per day
            logging.log.warning("model_f013 unexpected dt = %u" % dt)
    else:
        CHUNK_SIZE = int(CHUNK_DURATION / 30)  # -> equals 24 hours for 30 second resolution -> 1x per day
        logging.log.warning("model_f013 cannot read dt, t_func too short")

    # # store individual aging cumsum() to plot them later
    # log_df.loc[:, LABEL_Q_LOSS_SEI] = 0.0
    # log_df.loc[:, LABEL_Q_LOSS_WEARDOWN] = 0.0
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
            #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_weardown.iloc[i_start:i_end]))
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
            # log_df.loc[:, LABEL_Q_LOSS_WEARDOWN] = dq_b.cumsum()
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

    # return (dq_loss + dq_cyclic_age_weardown) * cfg.CELL_CAPACITY_NOMINAL
    return dq_loss * cfg.CELL_CAPACITY_NOMINAL
# === model_f013 END ===================================================================================================


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
    # logging.log.info("Starting threads...")
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE_MDL):
        # logging.log.debug("  Starting process %u" % processorNumber)
        # === USER CODE 2 BEGIN ========================================================================================
        processes.append(multiprocessing.Process(target=model_f014_get_delta_age,
                                                 args=(modeling_task_queue, result_queue)))
        # === USER CODE 2 END ==========================================================================================
    # time.sleep(3)  # thread exiting before queue is ready -> wait
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE_MDL):
        processes[processorNumber].start()
    for processorNumber in range(0, NUMBER_OF_PROCESSORS_TO_USE_MDL):
        processes[processorNumber].join()
        # logging.log.debug("Joined process %u" % processorNumber)

    # logging.log.debug("calculating RMSEs and plotting (1/2) ...")
    rmse_param = [0.0 for _ in range(0, n_pids)]
    num_rmse_points_param = [0.0 for _ in range(0, n_pids)]
    rmse_total, num_rmse_points_total = 0, 0
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

    # logging.log.debug("calculating RMSEs and plotting (2/2) ...")
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
    logging.log.debug("%s (%u) call with %s\n   results in "
                      "rmse_total = %.10f %%" % (mdl_name, mdl_id, var_string, rmse_total * 100.0))
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

        # FIXME: consider introducing discharge current related aging (gas generation?? SEI cracking? electrolyte
        #  decomposition? SEI growth as well? graphite exfoliation? electrode particle fracture/cracking? current
        #  collector corrosion? structural disordering? loss of electrical contact?)

        # dq_cyclic_age_weardown = a7 * log_df[LABEL_C_CHG] * t_func  # a7 * dQ_chg
        # q_cyclic_age_weardown = dq_cyclic_age_weardown.cumsum()

        # How to determine ΔQ_loss_SEI = (SEI_potential - a_4 * Q_loss_SEI) * Δt?
        # it is a differential equation, I don't know a way how to solve it with pure pandas.
        # https://ryxcommar.com/2020/01/15/for-the-love-of-god-stop-using-iterrows/
        # it seems like all iloc/loc/iterrows operations are slow, but for loops in general not too bad
        # faster accumulation using DataFrame chunks
        if t_func.shape[0] > 1:
            dt = t_func.max()
            if (dt == 2) or (dt == 30):
                CHUNK_SIZE = int(CHUNK_DURATION / dt)  # -> equals 24 hours -> 1x per day
            else:
                CHUNK_SIZE = int(CHUNK_DURATION / 30)  # -> equals 24 hours for 30 second resolution -> 1x per day
                logging.log.warning("model_f014 unexpected dt = %u" % dt)
        else:
            CHUNK_SIZE = int(CHUNK_DURATION / 30)  # -> equals 24 hours for 30 second resolution -> 1x per day
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
                #           - a3 * q_loss_sei_total / (1.0 + a4 * q_cyclic_age_weardown.iloc[i_start:i_end]))
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

        # return (dq_loss + dq_cyclic_age_weardown) * cfg.CELL_CAPACITY_NOMINAL
        result_entry = queue_entry
        result_entry["result"] = dq_loss * cfg.CELL_CAPACITY_NOMINAL
        result_queue.put(result_entry)

    task_queue.close()
    # logging.log.debug("exiting thread")
# === model_f014 END ===================================================================================================


# add the lines to the model result figure (if plotting is enabled)
def add_model_trace(mdl_name, log_df, mdl_cap_row, pid_index, pnr_index, fig_list, fig_and_sp_from_pid_arr, rmse_cell):
    [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
    this_fig: go.Figure = fig_list[i_fig]
    # x_data = log_df.loc[::N_PLOT_DATA_DOWNSAMPLE, DF_COL_TIME_USE] / TIME_DIV
    x_data = log_df.loc[::N_PLOT_DATA_DOWNSAMPLE, csv_label.TIMESTAMP] / TIME_DIV
    y_data = log_df.loc[::N_PLOT_DATA_DOWNSAMPLE, mdl_cap_row]
    this_color = TRACE_COLORS[pnr_index]
    trace_name = TRACE_NAME_MDL % (mdl_name, pid_index, pnr_index)
    this_fig.add_trace(go.Scatter(
        x=x_data, y=y_data, showlegend=False, mode='lines', line=dict(color=this_color, width=TRACE_LINE_WIDTH),
        opacity=TRACE_OPACITY, name=trace_name, hovertemplate=PLOT_HOVER_TEMPLATE_MDL),
        row=(i_row + 1), col=(i_col + 1))

    # add rmse_cell text
    text = TEXT_RMSE_CELL % (pnr_index + 1, rmse_cell * 100.0)
    y_pos = TEXT_POS_RMSE_Y_BASE + TEXT_POS_RMSE_DY * (TEXT_POS_RMSE_DY_OFFSET + TEXT_POS_RMSE_DY_FACTOR * pnr_index)
    x_pos = TEXT_POS_RMSE_X
    this_fig.add_annotation(xref="x domain", yref="y domain", x=x_pos, y=y_pos, showarrow=False,
                            opacity=TRACE_OPACITY, bgcolor=BG_COLOR, text=text, font=dict(size=10, color=this_color),
                            row=(i_row + 1), col=(i_col + 1)
                            )


# add RMSE to model result figure (if plotting is enabled)
def add_model_param_rmse(pid_index, fig_list, fig_and_sp_from_pid_arr, rmse_param):
    [i_fig, i_row, i_col] = fig_and_sp_from_pid_arr[pid_index]
    this_fig: go.Figure = fig_list[i_fig]
    text = TEXT_RMSE_PARAM % (rmse_param * 100.0)
    y_pos = TEXT_POS_RMSE_Y_BASE
    x_pos = TEXT_POS_RMSE_X
    this_fig.add_annotation(xref="x domain", yref="y domain", x=x_pos, y=y_pos, showarrow=False,
                            opacity=TRACE_OPACITY, bgcolor=BG_COLOR, text=text, font=dict(size=10, color='gray'),
                            row=(i_row + 1), col=(i_col + 1)
                            )


# plot the model result
def plot_model(mdl_name, mdl_id, fig_list, rmse_total, var_string):
    for i_fig in range(0, len(fig_list)):
        if fig_list[i_fig] is None:
            continue
        this_fig: go.Figure = fig_list[i_fig]
        title = this_fig['layout']['title']['text']
        rmse_text = TEXT_RMSE_TOTAL % (rmse_total * 100.0)
        # title_pre = "<b>%s (%u): %s</b><br>" % (mdl_name, mdl_id, rmse_text)
        # this_fig['layout']['title']['text'] = title_pre + "<b>" + title + "</b>")
        title_pre = "<b>%s (%u): %s</b>  -  " % (mdl_name, mdl_id, rmse_text)
        this_fig['layout']['title']['text'] = title_pre + title + "<br>%s" % var_string

        age_type = cfg.age_type(i_fig)
        age_type_text = ""
        if age_type in AGE_TYPE_FILENAMES:
            age_type_text = AGE_TYPE_FILENAMES.get(age_type)

        run_timestring = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_base = PLOT_FILENAME_BASE % (mdl_name, mdl_id, age_type_text, run_timestring)

        if SHOW_IN_BROWSER:
            logging.log.debug("%s, age mode %s - open figure in browser" % (mdl_name, age_type.name))
            this_fig.show()

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
                this_fig.write_html(filename)
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


# calculate root-mean-square error (RMSE) for this cell
def calc_rmse_cell(mdl_name, log_df, mdl_cap_row, rmse_tot, num_rmse_points_tot, rmse_par, num_rmse_points_par):
    # calculate RMSE at the data points (how? different approaches: per CU, per cell, per parameter)
    # filter valid cap_aged data points
    drop_cond = pd.isna(log_df[csv_label.CAP_CHARGED_EST])
    log_df_roi = log_df[~drop_cond]
    if log_df_roi.shape[0] == 0:
        logging.log.warning("%s - no valid cap_aged found! -> RMSE for cell not calculated" % mdl_name)
        rmse_cell = 0.0
    else:
        cap_diff = log_df_roi[csv_label.CAP_CHARGED_EST] - log_df_roi[mdl_cap_row]

        se_cell = (cap_diff ** 2).sum()
        mse_cell = se_cell / cap_diff.shape[0]
        rmse_cell = (se_cell / cap_diff.shape[0])**0.5  # actual RMSE of the cell

        rmse_par = rmse_par + mse_cell
        num_rmse_points_par = num_rmse_points_par + 1

        if error_calc_method == error_calculation_method.RMSE_PER_CU:
            # rmse_tot = rmse_tot + cap_diff.abs().sum()
            rmse_tot = rmse_tot + se_cell
            num_rmse_points_tot = num_rmse_points_tot + cap_diff.shape[0]
        elif error_calc_method == error_calculation_method.RMSE_PER_CELL:
            rmse_tot = rmse_tot + mse_cell
            num_rmse_points_tot = num_rmse_points_tot + 1
        elif error_calc_method == error_calculation_method.RMSE_PER_PARAMETER:
            # rmse_par = rmse_par + mse_cell
            # num_rmse_points_par = num_rmse_points_par + 1
            pass  # already calculated above
        else:
            logging.log.error("%s - RMSE calculation method not implemented: %s"
                              % (mdl_name, str(error_calc_method)))
    return rmse_tot, num_rmse_points_tot, rmse_par, num_rmse_points_par, rmse_cell


# calculate root-mean-square error (RMSE) for this parameter
def calc_rmse_parameter(rmse_tot, num_rmse_points_tot, rmse_par, num_rmse_points_par):
    if num_rmse_points_par > 0:
        mse_par = rmse_par / num_rmse_points_par
        rmse_par = mse_par**0.5
        if error_calc_method == error_calculation_method.RMSE_PER_PARAMETER:
            # rmse_tot = rmse_tot + ((rmse_par / num_rmse_points_par)**0.5)**2
            rmse_tot = rmse_tot + mse_par
            num_rmse_points_tot = num_rmse_points_tot + 1
    return rmse_tot, num_rmse_points_tot, rmse_par, num_rmse_points_par


# calculate total root-mean-square error (RMSE)
def calc_rmse_total(rmse_tot, num_rmse_points_tot):
    if num_rmse_points_tot > 0:
        rmse_tot = (rmse_tot / num_rmse_points_tot)**0.5
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
def prepare_cell_log_df(log_df, mdl_name, use_cap_nom, use_t_prod, mdl_cap_delta_row, mdl_cap_row):
    C_0 = cfg.CELL_CAPACITY_NOMINAL

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
            C_0 = use_cap_nom * cfg.CELL_CAPACITY_NOMINAL + (1.0 - use_cap_nom) * cap_aged.iloc[0]
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
        pdf_age = param_df[param_df[csv_label.AGE_TYPE] == age_type]
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
        plot_title = ""
        if age_type in AGE_TYPE_TITLES:
            plot_title = AGE_TYPE_TITLES.get(age_type)
        # noinspection PyTypeChecker
        fig_list[age_type] = generate_base_figure(n_rows, n_cols, plot_title, subplot_titles)
        for _, param_row in pdf_age.iterrows():
            pid = int(param_row[csv_label.PARAMETER_ID])
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
        fig_list[age_type] = generate_base_figure(n_rows, n_cols, plot_title, subplot_titles)
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
                profile_text = ht.get_age_profile_text(profile_list[i_r])
                i_t = (i_r * n_cols) + i_c
                subplot_titles[i_t] = t_text + ", " + profile_text
        plot_title = ""
        if age_type in AGE_TYPE_TITLES:
            plot_title = AGE_TYPE_TITLES.get(age_type)
        # noinspection PyTypeChecker
        fig_list[age_type] = generate_base_figure(n_rows, n_cols, plot_title, subplot_titles)
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
def generate_base_figure(n_rows, n_cols, plot_title, subplot_titles):
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
                     title_text=Y_AXIS_TITLE
                     )
    for i_col in range(0, n_cols):
        fig.update_xaxes(title_text=X_AXIS_TITLE, row=n_rows, col=(i_col + 1))

    fig.update_layout(title={'text': plot_title,
                             'font': dict(color='black'),
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


# estimate cell anode potential from terminal voltage (input and output as DataFrame)
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


if __name__ == "__main__":
    run()
