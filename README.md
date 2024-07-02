# Optimization scripts for generating a battery degradation model from battery aging data
Python optimization scripts to derive a battery degradation model from the battery aging data published in:  
- **Dataset:** "Comprehensive battery aging dataset: capacity and impedance fade measurements of a lithium-ion NMC/C-SiO cell [dataset]"  
  https://publikationen.bibliothek.kit.edu/1000168959  
  DOI: [10.35097/1947](https://doi.org/10.35097/1947)
- **Description:** "Comprehensive battery aging dataset: capacity and impedance fade measurements of a lithium-ion NMC/C-SiO cell"  
  [Paper in review]

*model_f056* is described in Chapter 7.2 of the dissertation:  
Matthias Luh: *"Robust electricity grids through intelligent,  highly efficient bidirectional charging systems  for electric vehicles"*


## File Overview and explanation

- **config_main.py:** IMPORTANT! Before starting, adjust file paths, e.g., define where the CFG and LOG_AGE .csv files of the dataset mentioned above are stored (the data is used to test how well the modeled and the measured capacity fade match).
- **optimize_models_automatic.py:** Automatic optimization using SciPy.
- **optimize_models_manual.py:** Manual optimization using fixed, user-defined values for variable, list-wise stepping, or matrix-wise "brute-force" testing of variable combinations.
- **optimize_models_manual_stepping_list.py:** List of variables to test for the "list-wise stepping" in optimize_models_manual.py.
- **config_labels.py:** Definition of data column labels. Not all columns of the published records are defined here, you can add them if you need them.
- **config_logging.py:** Used to log (debug) information, warnings, and errors to the console and a log text file.
- **requirements.txt**: Required libraries (and version with which they were successfully tested).

## Required input data
CFG and LOG_AGE .csv files of the dataset mentioned above.