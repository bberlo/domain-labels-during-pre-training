# MDPI Sensors journal paper - Use of Domain Labels during Pre-Training for Domain-Independent WiFi-CSI Gesture Recognition
Author: Bram van Berlo - b.r.d.v.berlo@tue.nl

### Use of the docker container
Note: steps listed below only work on a host machine running with a Linux OS.

1) Download the entire repository to a machine.

Call the following docker commands inside the repository:

2) docker build -t bberlo/mdpi-audit:2023.11 .
3) docker run --name bberlo_mdpi_audit -v $PWD/Data/:/project/Code/Datasets/ bberlo/mdpi-audit:2023.11 /bin/bash /project/Code/run.sh
4) (inside a new terminal) docker cp bberlo_mdpi_audit:/project/Code/results/ ./Code/results

Note: due to required storage capacity, the $PWD/Data/ directory is unavailable at GitHub. Please contact TU/e IRIS (https://iris.win.tue.nl/) for access.
Note: wait for a docker command to be finished running before executing the next docker command.
Note: In order to check if the docker run command is finished, inspect the terminal information of the run.sh command (docker run is not executed in detached mode).

### Plots

Please check the readme file inside the 'Code' directory on how the figure and table plots inside the journal paper should be created using the data inside the 'Code/results' directory.
