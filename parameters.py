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
vis_representation = 'ed_canny'

################
# Window shape #
################

# This is the shape of the window to crop (string).
# Usage example: shape_to_crop = 'circle_opt2'
# Options are:
#       <circle_opt1> Edge detection on just a circle (-) generates border artifacts
#       <circle_opt2> Edge detection on a square, which is then cropped into a circle [*Recommended]
#       <square> Edge detection on a square
shape_to_crop = 'circle_opt2'

##############
# Patch size #
##############

# This is the height (int) of the patch in which the viewer sees the gaze contingent image.
# Usage example: patch_size = 128
patch_size = 128
