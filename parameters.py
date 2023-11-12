#####################
# What is displayed #
#####################

# This determines what sort of experiment to run (string), and therefore what is displayed on the screen.
# Usage example: experiment_type = 'display_roi'
# Options are:
#       <follow_gaze_position> If you want to test gaze position is being followed with a cross
#       <display_roi> Render a portion of the image based on eye position
experiment_type = 'display_roi'

#################
# Edge detector #
#################

# This is the method of edge detection to use (string).
# Usage example: edge_detector = 'sobel'
# Options are:
#       <canny> Canny edge detection
#       <sobel> Sobel edge detection
edge_detector = 'sobel'

################
# Window shape #
################

# This is the shape of the window to crop (string).
# Usage example: shape_to_crop = 'circle_opt2'
# Options are:
#       <circle_opt1> Edge detection on just a circle (-) generates border artifacts
#       <circle_opt2> Edge detection on a square, which is then cropped into a circle
#       <square> Edge detection on a square
shape_to_crop = 'circle_opt2'

##############
# Patch size #
##############

# This is the height (int) of the patch in which the viewer sees the gaze contingent image.
# Usage example: patch_size = 100
patch_size = 100
