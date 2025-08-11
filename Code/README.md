# MDPI 2023 journal paper - Use of Domain Labels during Pre-Training for Domain-Independent WiFi-CSI Gesture Recognition
Author: Bram van Berlo - b.r.d.v.berlo@tue.nl

### Requirements
In the requirements.txt file the needed python modules for this project are specified.

### Preprocessing steps
The pre-processed datasets in '../Data' are created using the pre-processing scripts in 'pre-processing/'.
Links to the original datasets have been added to this subdirectory as well.
The pre-processing scripts are not part of the audit procedure and can therefore not be run automatically.

Steps required in order to use the pre-processing scripts:

1) Download entire original dataset to a directory.
2) Download '../Code' subdirectory to a directory.
3) Using python 3.7 < ver. < 3.9, install imported packages listed in the pre-processing script.

Depending on pre-processing script, follow either one of the following set of steps.

### Widar_CSI_create_npy_files_domain_leave_out.py

4) Extract all CSI_*.zip archives into a specific subdirectory.
5) In the pre-processing script, update the top directory to the subdirectory of the previous step over which the os.walk() function should list files recursively.
6) In the pre-processing script, in-between lock.acquire() and lock.release(), update the directory location in which the .hdf5 dataset should be saved.
7) list_item_root.split(os.sep)[5] and current_root.split(os.sep)[5] calls depend on specific number of directories inside the path string. These calls should be checked on if list index 5 returns date substrings. If not, list index should be updated.
8) Execute pre-processing script.


### Signfi_CSI_create_raw_npy_files_domain_leave_out.py

4) Only keep dataset_lab_276_dl.mat, dataset_home_276.mat, and dataset_lab_150.mat in a specific subdirectory.
5) Inside the subdirectory, create subfolders 'raw' and 'processed'.
6) Place the abovementioned .mat files in the 'raw' folder.
7) Execute pre-processing script in subdirectory with script option -t. Possible values: env_home, env_lab, and user_lab.

### Reproducing results

The run.sh file includes all bash commands which should be run to acquire the results used in Figures 4-9, Figure 10a-b, and Figures 11-13 of the journal paper.
Prior to running the bash commands, make sure that all packages listed in the requirements.txt file are installed.
Prior to running the bash commands, copy the datasets inside the '../Data' directory to the 'Datasets/' directory.
The results in .csv format are placed in subdirectories per benchmark pipeline inside the results/ directory.

Bash command for acquiring results presented in Figure 10c is not provided. Reason being that the bash command has to be run on a polling schedule in a concurrent fashion w.r.t. bash commands in run.sh. In order to acquire Figure 10c results, while running one of the python bash commands in run.sh, poll the linux /proc/meminfo file for MemTotal and MemFree at an interval comparable to the one used in Figure 10c. Subsequently, plot MemTotal - MemFree.
Bash command for acquiring TIME results presented in Table 2 is not provided. Reason being that it is easier to extract timing information manually from STDOUT information written into a .txt file. While running one of the python bash commands in run.sh, write STDOUT to .txt file. Subsequently, extract time in seconds information for epoch 2 (epoch 1's time is offset due to Just in Time (JiT) compilation).
Other results presented in Table 2 is static hardware information for the Amazon EC2 g5.4xlarge and g5.24xlarge instances.

### Plots

Figure plots inside the journal paper were created by processing the .csv formatted results with MS Excel into a chart structure.

In every .csv file, the last 5 columns denote the summary statistics achieved on a specifically held out test dataset.
The data inside these columns for different cross validation splits should be grouped per deep learning technique and domain factor held out type.
On the grouped data, AVERAGE and VAR.S functions should be called per metric.
The function outputs should be structured according to a bar chart structure comparable to the structure used inside the journal paper for Figures 4, 6, 8, 11, 13. The WiGRUNT results in Figure 13 were extracted directly from the respective research paper.

In every .csv file, there are also columns called 'val_categorical_accuracy' and 'A'. Figures 5, 7, 9, and 12 can be created by, for the respective benchmark pipeline, dataset, and domain-leave-out CV split, overlaying a square with max. height at value of 'A' over 'val_categorical_accuracy' plot. 

Figure 10a-b can be created by, for the respective benchmark pipeline, dataset, and domain-leave-out CV split, plotting the first left-hand side columns called 'loss' and 'val_loss'.  

