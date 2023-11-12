# Gaze contingency

## About
Tracking the position of the eye:
![Output sample](gifs/tracking_eye_pos.gif)

Rendering parts of the image based on the current eye position:
![Output sample](gifs/image_following_gaze.gif)

## How to start

Make a new venv and install the following packages:

    pip install opencv-python
    pip install pygame
    pip install PIL
    pip install --index-url=https://pypi.sr-support.com sr-research-pylink

Set the parameters to chosen values inside `parameters.py`, then start the script `gazeContig.py` with your workstation connected to the host PC with EyeLink configured.

### Output of the script
The results will be created inside the `results\{result_fn}_{dt}\` directory where `{result_fn}` is a command-line input given by the experimenter (name of the run) and `{dt}` is the current datetime. Inside this directory, two files will be populated:
* `.EDF` file: this is the output from EyeLink
* `.txt` file: this file contains the parameter values chosen for this run, as retrieved from the `parameters.py` file 