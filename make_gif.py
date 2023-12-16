import re

from PIL import Image
import glob

# This file will take the images in the rendered_experiment folder and make a GIF out of them.
# Code from: https://stackoverflow.com/questions/72945567/how-do-i-save-an-array-of-pillow-images-as-a-gif

image_list = []
TO_GIF = "results/test_2023_12_16_17_08/rendered_experiment/stimulus_1"
GIF_NAME = "border_conditions_capped"

# Save consecutive images in an image list:
for filename in sorted(glob.glob(f'{TO_GIF}/*.jpg'), key=lambda name: int(''.join(filter(str.isdigit, name)))):
    im = Image.open(filename)
    image_list.append(im)

# Make gif from the images:
image_list[0].save(f"{TO_GIF}/{GIF_NAME}.gif", save_all=True, append_images=image_list[1:],
                   optimize=False, duration=50, disposal=2, loop=0)