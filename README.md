# Gaze contingency

## About
This project aims to develop a framework for assessing Simulated Prosthetic Vision (SPV) experiments on gaze contingency. This script is written in Python 3.10 and works with the EyeLink1000 provided by [SR Research](https://www.sr-research.com/).

This script has been adapted from the example script provided by SR Research in `C:\Program Files (x86)\SR Research\EyeLink\SampleExperiments\Python\examples\Pygame_examples\fixationWindow_fastSamples.py`. For more information, visit [SR Research SUPPORT](https://www.sr-research.com/support/thread-7525.html).

### Demo

Tracking the position of the eye:
![Output sample](gifs/tracking_eye_pos.gif)

Rendering parts of the image based on the current eye position:
![Output sample](gifs/image_following_gaze.gif)

Rendering a gaze-contingent edge-detected version of the image:
![Output sample](gifs/gaze_cont_canny.gif)

## How to start

Make a new venv and install the following packages:

    pip install opencv-python
    pip install pygame
    pip install PIL
    pip install --index-url=https://pypi.sr-support.com sr-research-pylink

- [ ] _Update the above with a pip requirements file_.

Set the parameters to chosen values inside `parameters.py`.

**Optional**: copy in any images you want to use inside the `\images` directory. Alternatively, you can also run the experiments with the 2 images already provided in this directory.

Start the script `main.py` with your workstation connected to the host PC with EyeLink configured.

### Output of the script
The results will be created inside the `results\{result_fn}_{dt}\` directory where `{result_fn}` is a command-line input given by the experimenter (name of the run) and `{dt}` is the current datetime. Inside this directory, the following two files will be populated:
* `{result_fn}_{dt}.EDF` file: this is the output from EyeLink
* `{result_fn}_{dt}.txt` file: this file contains the parameter values chosen for this run, as retrieved from the `parameters.py` file 

Additionally, the following folder will be generated:
* `rendered_experiment/`: this folder will contain a subdirectory for each of the number of input images you have provided. E.g. with the 2 example images, the subdirectories created are `stimulus_1/` and  `stimulus_2/`. Each of these subdirectories contains images of each of the screens the participant sees in the experiment. You can use the `make_gif.py` script to create a gif out of these images to display the experiment how the participant has experienced it.