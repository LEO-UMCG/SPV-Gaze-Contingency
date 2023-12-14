#####################
# What is displayed #
#####################

# This determines what sort of experiment to run (string), and therefore what is displayed on the screen.
# Usage example: experiment_type = 'display_roi'
# Options are:
#       <follow_gaze_position> If you want to test gaze position is being followed with a cross
#       <display_roi> Render a portion of the image based on eye position
experiment_type = 'display_roi'

##########
# Images #
##########

# The file extension of the images provided (string).
# Usage example: image_extension = 'jpg'
image_extension = 'jpg'

##########
# Trials #
##########

# How many trials i.e. repititions of the images to show (int).
# Usage example: num_trials = 1
num_trials = 1

###########
#  Order  #
###########

# Whether to show the trials in the order the images are read, or whether to randomize them (bool).
# Usage example: to_randomize_trials = False
to_randomize_trials = False

##############
# Peripheral #
##############

# Whether to show the stimulus that would be visible to the periphery of the participant when gaze
# position exceeds the bounds of the monitor.
# Usage example: use_periphery = False
use_periphery = True

#################
# Edge-dilation #
#################

# Whether to dilate the edges when using an edge detection method (bool).
# Usage example: dilate_edges = True
dilate_edges = True

##################
# Image duration #
##################

# This determines how long (in ms) to present the participant with the image before timeout (int).
# Usage example: max_presentation_duration_img = 10000
max_presentation_duration_img = 10000

####################
# Trigger duration #
####################

# This determines how long (in ms) the participant has to fixate on the trigger cross before timeout (int).
# Usage example: trigger_timeout_duration = 10000
trigger_timeout_duration = 10000

#########################
# Visual representation #
#########################

# This is the method of visual representation to use (string).
# The prefixes ed_ represent edge detectors and the prefixes dl_ represent deep learning models.
# Usage example: vis_representation = 'ed_sobel'
# Options are:
#       <ed_canny> Canny edge detection
#       <ed_sobel> Sobel edge detection
#       <dl_jaap> Jaap's DL model
#       <dl_ash> Ashkan's DL model
vis_representation = 'dl_ash'

################
# Window shape #
################

# This is the shape of the window to crop (string).
# Usage example: shape_to_crop = 'circle_opt2'
# Options are:
#       <circle_opt1> Create a blur on the image while keeping the central circle sharp
#       <circle_opt2> Visual representations created in square, which is then masked out to leave a circle window
#       <square> Visual representations created in square window
shape_to_crop = 'circle_opt2'

##############
# Patch size #
##############

# This is the height (int) of the patch in which the viewer sees the gaze contingent image.
# Usage example: patch_size = 128
patch_size = 128

###########
# EDF2ASC #
###########

# This is the path to the edf2asc tool that is installed on your PC as part of EyeLink Developers Kit.
# Usage example: path_to_edf2asc = '/usr/bin/edf2asc'
path_to_edf2asc = '/usr/bin/edf2asc'                                                  # Linux (UMCG PC)
# path_to_edf2asc = 'C:\\Program Files (x86)\\SR Research\\EyeLink\\bin\\edf2asc.exe'     # Windows (local laptop)