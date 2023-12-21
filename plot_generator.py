import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


def combine_images(directory_path):
    # This function will combine all frame images so that we get to see each gaze-contingent window
    # that was seen by the participant in 1 image.

    # Get a list of all image files:
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.jpg')]

    # Read in the first image, which will act as the underlying image to which we consecutively paste more images onto:
    underlying_image_path = os.path.join(directory_path, image_files[0])
    underlying_image = cv2.imread(underlying_image_path)

    # Loop through the remaining images and blend them onto the base image
    for image_file in image_files[1:]:
        curr_image_path = os.path.join(directory_path, image_file)
        curr_image = cv2.imread(curr_image_path)
        # Combine the underlying images with the current:
        underlying_image = cv2.bitwise_or(underlying_image, curr_image)

    return underlying_image


def plot_gaze_coor(im, obj_gaze_data):
    # This function will plot the movement of the participant's gaze with arrows against a backdrop of the stimuli.

    # Plot the combined image in the background:
    plt.imshow(im)
    idx = 0
    # Use the prism color space for plotting gaze contingent arrows:
    colors = iter(cm.prism(np.linspace(0, 1, 100)))

    for eye_data_list in obj_gaze_data:
        # Ignore eye blink events:
        if len(eye_data_list) > 1:
            gaze_x = eye_data_list[1]
            gaze_y = eye_data_list[2]

            # Capture the movement every 500ms:
            if idx != 0 and idx % 500 == 0:
                # Plot an arrow from the previous gaze position to the current:
                plt.arrow(prev_gaze_x, prev_gaze_y, gaze_x - prev_gaze_x, gaze_y - prev_gaze_y, width=5,
                          color=next(colors), shape='full', head_width=30)
                prev_gaze_x = gaze_x
                prev_gaze_y = gaze_y

            if idx == 0:
                # Store gaze coordinates at t=0:
                prev_gaze_x = gaze_x
                prev_gaze_y = gaze_y
                # Plot starting gaze coordinates:
                plt.plot(prev_gaze_x, prev_gaze_y, 'x', color='r')

            idx += 1

    plt.ylabel("Gaze y Pixel Coordinates")
    plt.xlabel("Gaze x Pixel Coordinates")
    plt.show()


def plot_gaze_by_time(obj_gaze_data):
    # Plot the pixel positions of the eye data over time.

    # The eye_data_gaze is referenced at the index of the trial we want to plot:
    timestamp = [sublist[0] for sublist in obj_gaze_data]
    gaze_x = []
    gaze_y = []
    for eye_data_list in obj_gaze_data:
        # Eye blink events:
        if len(eye_data_list) == 1:
            gaze_x.append(0)
            gaze_y.append(0)
        else:
            gaze_x.append(eye_data_list[1])
            gaze_y.append(eye_data_list[2])
    plt.plot(range(len(timestamp)), gaze_x, color='r', label='Gaze x')
    plt.plot(range(len(timestamp)), gaze_y, color='b', label='Gaze y')
    plt.xlabel("Milliseconds")
    plt.ylabel("Pixel Coordinate")
    plt.legend()
    plt.show()


# Path to images of only the stimuli (no trigger images):
test_session_identifier = "test_2023_12_18_17_19"
# Object to plot options: mug, fork, guitar
object = "guitar"
images_path = f"results/{test_session_identifier}/{object}"

# Path to experiment df:
df = pd.read_pickle(f'results/{test_session_identifier}/{test_session_identifier}')
guitar_gaze_data = df.eye_data_gaze[1]
mug_gaze_data = df.eye_data_gaze[2]
fork_gaze_data = df.eye_data_gaze[3]
obj_gaze_data = guitar_gaze_data  # Select the data of a particular stimulus for the plotting

# First combine the images:
combined_image = combine_images(images_path)
# Then plot gaze information on this combined image:
plot_gaze_coor(combined_image, obj_gaze_data)

# Next make a gaze/time line plot:
plot_gaze_by_time(obj_gaze_data)
