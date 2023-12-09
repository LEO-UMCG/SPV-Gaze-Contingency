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


def cap_gaze_coords_periphery(gaze_x, gaze_y, patch_size, image):
    # Make sure we don't exceed the bounds of the image.
    # This acts like a periphery vision (i.e. looking right outside the monitor shows you the right-most part
    # of the image).

    # If we are outside the right side of the image, cap it at the right-hand borders:
    if gaze_x + patch_size >= image.shape[1]:
        gaze_x = image.shape[1] - patch_size - 1

    # If we are outside the left side of the image, cap it at the left-hand borders:
    if gaze_x - patch_size <= 0:
        gaze_x = patch_size + 1

    # If we are outside the bottom of the image, cap it at the bottom borders:
    if gaze_y - patch_size <= 0:
        gaze_y = patch_size + 1

    # If we are outside the top of the image, cap it at the top borders:
    if gaze_y + patch_size >= image.shape[0]:
        gaze_y = image.shape[0] - patch_size - 1

    return gaze_x, gaze_y


def truncate_gaze_coords(gaze_x, gaze_y, patch_size, image):
    # Make sure we don't exceed the bounds of the image.
    # If we look outside the bounds of the monitor, show nothing.

    # If we are outside the right side or left side respectively:
    if gaze_x + patch_size >= image.shape[1] or gaze_x - patch_size <= 0:
        # we want to return a black screen:
        gaze_x = -32768.0

    # If we are outside the bottom or top bounds respectively:
    if gaze_y - patch_size <= 0 or gaze_y + patch_size >= image.shape[0]:
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


def get_cropped_patch_region(image, gaze_y, gaze_x):
    # Crop out a square patch of the image about the gaze x, y coordinates given:
    crop_img = image[int(gaze_y - patch_size): int(gaze_y + patch_size),
               int(gaze_x - patch_size): int(gaze_x + patch_size)].copy()
    return crop_img


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
        gaze_x, gaze_y = cap_gaze_coords_periphery(gaze_x, gaze_y, patch_size, image)
    else:
        gaze_x, gaze_y = truncate_gaze_coords(gaze_x, gaze_y, patch_size, image)

        # Exceeding monitor bounds condition - return black screen:
        if gaze_x == -32768.0 and gaze_y == -32768.0:
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
        crop_mask_img = get_cropped_patch_region(mask, gaze_y, gaze_x)
        show_image(crop_mask_img, 'Tight circle region', to_debug)

        if shape_to_crop == 'circle_opt1':
            # Crop out a square:
            crop_img = get_cropped_patch_region(image, gaze_y, gaze_x)

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
            crop_img = get_cropped_patch_region(image, gaze_y, gaze_x)
            cropped_edge_img = launch_according_vis_rep_method(crop_img, enc, sim, to_debug)
            show_image(cropped_edge_img, 'Edge detection in square', to_debug)

            # Convert float32 to uint8 because PyGame requires it:
            converted_img = convert_float32_to_uint8(cropped_edge_img)
            # Fill in white with actual image values:

            crop_vis_rep_img = cv2.bitwise_and(converted_img, crop_mask_img, to_debug)
            show_image(crop_vis_rep_img, 'Filled circle region', to_debug)

    if shape_to_crop == 'square':
        # First get a cropped out patch about the gaze coordinates:
        crop_img = get_cropped_patch_region(image, gaze_y, gaze_x)
        # Next use the according visual representation method on this result:
        crop_vis_rep_img = launch_according_vis_rep_method(crop_img, enc, sim)
        show_image(crop_vis_rep_img, 'Edges', to_debug)

    # Create a black canvas of size of the image:
    black_canvas = np.zeros(image.shape, dtype=crop_vis_rep_img.dtype)
    # Paste the cropped out region onto the black canvas:
    black_canvas[int(gaze_y - patch_size): int(gaze_y + patch_size),
               int(gaze_x - patch_size): int(gaze_x + patch_size)] = crop_vis_rep_img
    updated_img_to_show = black_canvas

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

    img = cv2.imread('images/img_3.jpg')
    x = get_gaze_contig_img(img, 1282, 652, encoder, simulator, True)
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
