import numpy as np
import cv2
from edge_detection import get_sobel_edges, get_canny_edges, generate_phosphenes_for_ed
from spvPlayer.models.BS_model import *
from spvPlayer.models.E2E_model import *
import pygame
from parameters import *


def initialisation_step():
    enc, sim = None, None

    # Initialisation for DL models:
    if vis_representation == 'dl_jaap':
        enc = prepJaapEncoder()  # load jaap's e2e model
        sim = prepSimulator()

    if vis_representation == 'dl_ash':
        enc = prepAshEncoder()
        sim = prepRegSimulator()

    # Initialisation for ED methods (using Ashkan's simulator):
    if "ed_" in vis_representation:
        enc = None  # No DL model used
        sim = prepRegSimulator()

    return enc, sim


def show_image(img, img_name, to_debug):
    # Show images only if we are in debug mode:
    if to_debug:
        cv2.imshow(img_name, img)
        cv2.waitKey(0)


def cap_gaze_coords_periphery(gaze_x, gaze_y, image):
    # Make sure we don't exceed the bounds of the image.
    # This acts like a periphery vision (i.e. looking right outside the monitor shows you the right-most part
    # of the image).

    # If we are outside the right side of the image, cap it at the right-hand borders:
    if gaze_x + patch_size >= image.shape[1]:
        gaze_x = image.shape[1] - patch_size

    # If we are outside the left side of the image, cap it at the left-hand borders:
    if gaze_x - patch_size <= 0:
        gaze_x = patch_size

    # If we are outside the bottom of the image, cap it at the bottom borders:
    if gaze_y - patch_size <= 0:
        gaze_y = patch_size

    # If we are outside the top of the image, cap it at the top borders:
    if gaze_y + patch_size >= image.shape[0]:
        gaze_y = image.shape[0] - patch_size

    return gaze_x, gaze_y


def truncate_gaze_coords(gaze_x, gaze_y, image):
    # Make sure we don't exceed the bounds of the image.
    # If we look outside the bounds of the monitor, show nothing.

    # If we are outside the right side or left side respectively:
    if gaze_x >= image.shape[1] or gaze_x <= 0:
        # we want to return a black screen:
        gaze_x = -32768.0

    # If we are outside the bottom or top bounds respectively:
    if gaze_y <= 0 or gaze_y >= image.shape[0]:
        # we want to return a black screen:
        gaze_y = -32768.0

    # If we aren't outside the bounds of the image, do nothing
    return gaze_x, gaze_y


def launch_according_vis_rep_method(crop_img, enc, sim, to_debug):
    # Get the visual representation according to the method desired:

    if "ed_" in vis_representation:
        if vis_representation == 'ed_sobel':
            cropped_edge_img = get_sobel_edges(crop_img)

        if vis_representation == 'ed_canny':
            cropped_edge_img = get_canny_edges(crop_img)

        show_image(cropped_edge_img, 'ED on patch', to_debug)

        if dilate_edges:
            kernel = np.ones((3, 3), np.uint8)
            cropped_edge_img = cv2.dilate(cropped_edge_img, kernel, iterations=1)
            show_image(cropped_edge_img, 'Dilated edges', to_debug)

        # Generate phosphene representation for ED methods:
        cropped_edge_img = generate_phosphenes_for_ed(cropped_edge_img, sim, toggle=False)

    if "dl_" in vis_representation:
        if vis_representation == 'dl_jaap':
            cropped_edge_img = jaapPredict(crop_img, enc, sim)
        if vis_representation == 'dl_ash':
            cropped_edge_img = ashPredict(crop_img, enc, sim, toggle=False)

    # Convert back to 3 channels to prevent mismatch of shape size:
    cropped_edge_img = cv2.cvtColor(cropped_edge_img, cv2.COLOR_GRAY2RGB)

    return cropped_edge_img


def get_cropped_patch_region(image, gaze_y, gaze_x, to_debug):

    # Start out by assuming we are within the bounds of the image.
    # Crop out a square patch of the image about the gaze x, y coordinates given:
    exceeded_bounds = False
    start_y = gaze_y - patch_size
    end_y = gaze_y + patch_size
    start_x = gaze_x - patch_size
    end_x = gaze_x + patch_size

    # The following 4 if-statements are for when we exceed the image bounds. Update the relevant gaze coordinates.
    # If we are outside the right side of the image:
    if gaze_x + patch_size >= image.shape[1]:
        remaining_space = image.shape[1] - gaze_x
        end_x = gaze_x + remaining_space
        exceeded_bounds = True

    # If we are outside the left side of the image:
    if gaze_x - patch_size <= 0:
        remaining_space = gaze_x
        start_x = gaze_x - remaining_space
        exceeded_bounds = True

    # If we are outside the bottom of the image:
    if gaze_y - patch_size <= 0:
        remaining_space = gaze_y
        start_y = gaze_y - remaining_space
        exceeded_bounds = True

    # If we are outside the top of the image:
    if gaze_y + patch_size >= image.shape[0]:
        remaining_space = image.shape[0] - gaze_y
        end_y = gaze_y + remaining_space
        exceeded_bounds = True

    # The following logic is only relevant when we exceed the bounds of the image:
    if exceeded_bounds:
        # Adjust the gaze coordinates to the edges of the image. i.e. if they are too close to the edges, adjust the
        # coordinates.
        clean_gaze_x, clean_gaze_y = cap_gaze_coords_periphery(gaze_x, gaze_y, image)
        clean_start_y = clean_gaze_y - patch_size
        clean_end_y = clean_gaze_y + patch_size
        clean_start_x = clean_gaze_x - patch_size
        clean_end_x = clean_gaze_x + patch_size

        # Crop out a truncated version first:
        crop_img = image[int(start_y): int(end_y), int(start_x): int(end_x)].copy()
        show_image(crop_img, 'Cropped image based on true gaze loc', to_debug)

        # Create a backdrop:
        backdrop_img = image[int(clean_start_y): int(clean_end_y), int(clean_start_x): int(clean_end_x)].copy()
        backdrop_img = np.zeros(backdrop_img.shape, dtype=backdrop_img.dtype)
        show_image(backdrop_img, 'Backdrop image', to_debug)

        # Paste the cropped image into the backdrop. This is necessary because the visual representation methods
        # can only deal with images of size patch_size*patch_size:
        new_width = clean_end_x - clean_start_x
        new_height = clean_end_y - clean_start_y
        backdrop_img[int(start_y - clean_start_y): int(new_height - (clean_end_y - end_y)),
        int(start_x - clean_start_x): int(new_width - (clean_end_x - end_x))] = crop_img
        crop_img = backdrop_img
        show_image(crop_img, 'Cropped image pasted onto backdrop', to_debug)
    else:
        crop_img = image[int(start_y): int(end_y), int(start_x): int(end_x)].copy()
        show_image(crop_img, 'Cropped image based on true gaze loc', to_debug)
        clean_start_y, clean_end_y, clean_start_x, clean_end_x = start_y, end_y, start_x, end_x

    return crop_img, start_y, end_y, start_x, end_x, clean_start_y, clean_end_y, clean_start_x, clean_end_x


def convert_float32_to_uint8(img):
    # Convert a given img of dtype float32 to an uint8 image:
    img_normalized = np.clip(img, 0, 1)
    converted_img = (img_normalized * 255).astype(np.uint8)

    return converted_img


def get_gaze_contig_img(image, gaze_x, gaze_y, enc, sim, to_debug):
    cropped_edge_img = None

    # For debugging:
    # print(f"Current gaze location: {gaze_x}, {gaze_y}")

    # Closed eye condition - return black screen:
    if gaze_x == -32768.0 and gaze_y == -32768.0:
        return np.zeros(image.shape, dtype=np.uint8)

    if use_periphery:
        gaze_x, gaze_y = cap_gaze_coords_periphery(gaze_x, gaze_y, image)
    else:
        gaze_x, gaze_y = truncate_gaze_coords(gaze_x, gaze_y, image)

        # Exceeding monitor bounds condition - return black screen:
        if gaze_x == -32768.0 or gaze_y == -32768.0:
            return np.zeros(image.shape, dtype=np.uint8)

    # Crop the image at the gaze coordinates:
    # (use a copy of the image so you don't change the original image)
    # (convert to ints because gaze coords are floats)

    if "circle" in shape_to_crop:
        # Localise circle on image:
        mask = np.zeros_like(image)
        mask = cv2.circle(mask, (int(gaze_x), int(gaze_y)), patch_size, (255, 255, 255), -1)
        show_image(mask, 'Localise circle', to_debug)

        # Crop down the circular region to only the area of interest:
        crop_mask_img, _, _, _, _, _, _, _, _ = get_cropped_patch_region(mask, gaze_y, gaze_x, to_debug)
        show_image(crop_mask_img, 'Tight circle region', to_debug)

        if shape_to_crop == 'circle_opt1':
            # Crop out a square:
            crop_img, start_y, end_y, start_x, end_x, clean_start_y, clean_end_y, clean_start_x, clean_end_x = get_cropped_patch_region(image, gaze_y, gaze_x, to_debug)

            # Blurred version of image:
            blurred = cv2.GaussianBlur(crop_img, (21, 21), 0)
            show_image(blurred, 'Blurred image', to_debug)

            # Inverted version of mask:
            inv_mask = cv2.bitwise_not(crop_mask_img)
            show_image(inv_mask, 'Inverted mask', to_debug)

            # Fill in white areas with actual image values:
            crop_img = cv2.bitwise_and(crop_img, crop_mask_img)
            show_image(crop_img, 'Filled circle region', to_debug)

            crop_img_borders = cv2.bitwise_and(blurred, inv_mask)
            show_image(crop_img_borders, 'Filled border region', to_debug)

            # Combine blurred outer regions with sharp inner circle:
            combined_result = cv2.add(crop_img, crop_img_borders)
            show_image(combined_result, 'Final result', to_debug)

            # Get the edges on the cropped portion of the image:
            crop_vis_rep_img = launch_according_vis_rep_method(combined_result, enc, sim, to_debug)
            show_image(crop_vis_rep_img, 'Edges', to_debug)

        if shape_to_crop == 'circle_opt2':
            # Crop out a square:
            crop_img, start_y, end_y, start_x, end_x, clean_start_y, clean_end_y, clean_start_x, clean_end_x = get_cropped_patch_region(image, gaze_y, gaze_x, to_debug)
            cropped_edge_img = launch_according_vis_rep_method(crop_img, enc, sim, to_debug)
            show_image(cropped_edge_img, 'Edge detection in square', to_debug)

            # Convert float32 to uint8 because PyGame requires it:
            converted_img = convert_float32_to_uint8(cropped_edge_img)
            # Fill in white with actual image values:

            crop_vis_rep_img = cv2.bitwise_and(converted_img, crop_mask_img, to_debug)
            show_image(crop_vis_rep_img, 'Filled circle region', to_debug)

    if shape_to_crop == 'square':
        # First get a cropped out patch about the gaze coordinates:
        crop_img, start_y, end_y, start_x, end_x, clean_start_y, clean_end_y, clean_start_x, clean_end_x = get_cropped_patch_region(image, gaze_y, gaze_x, to_debug)
        # Next use the according visual representation method on this result:
        crop_vis_rep_img = launch_according_vis_rep_method(crop_img, enc, sim, to_debug)
        show_image(crop_vis_rep_img, 'Edges', to_debug)

    # Crop the patch_size*patch_size image back to desired size:
    new_width = clean_end_x - clean_start_x
    new_height = clean_end_y - clean_start_y
    crop_img_back = crop_vis_rep_img[int(start_y - clean_start_y): int(new_height - (clean_end_y - end_y)),
        int(start_x - clean_start_x): int(new_width - (clean_end_x - end_x))].copy()

    # Create a black canvas of size of the image:
    black_canvas = np.zeros(image.shape, dtype=crop_vis_rep_img.dtype)

    # black_canvas[int(start_y): int(end_y), int(start_x): int(end_x)] = crop_vis_rep_img
    black_canvas[int(start_y): int(end_y), int(start_x): int(end_x)] = crop_img_back
    updated_img_to_show = black_canvas
    show_image(crop_img, 'Cropped image pasted onto black canvas', to_debug)

    if crop_vis_rep_img.dtype != 'uint8':
        # Convert float32 to uint8 because PyGame requires it:
        updated_img_to_show = convert_float32_to_uint8(updated_img_to_show)

    # For testing this function without the eyetracker:
    show_image(updated_img_to_show, 'Resulting image', to_debug)

    return updated_img_to_show


def troubleshoot_gaze_contig():
    # This function is for testing the methods which obtain a gaze-contingent image without the eyetracker.
    # You can call it on its own.

    encoder, simulator = initialisation_step()

    image = cv2.imread('images/img_51.jpg')

    x = get_gaze_contig_img(image, 20, 60, encoder, simulator, True)
    size = x.shape[1::-1]
    image_surface = pygame.image.frombuffer(x.flatten(), size, 'RGB')

    # Display in PyGame:
    pygame.init()
    screen = pygame.display.set_mode(size)
    screen.blit(image_surface, (0, 0))
    pygame.display.flip()

    while True:
        running = True
    pygame.quit()


# For testing:
# troubleshoot_gaze_contig()
