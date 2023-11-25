import numpy as np
import cv2
from edgeDetection import getSobelEdges, getCannyEdges
from spvPlayer.models.BS_model import *
from spvPlayer.models.E2E_model import *
import pygame
from parameters import *


def initialisation_dl():
    enc, sim = None, None

    # Initialisation for DL models:
    if vis_representation == 'dl_jaap':
        enc = prepJaapEncoder()  # load jaap's e2e model
        sim = prepSimulator()

    if vis_representation == 'dl_ash':
        enc = prepAshEncoder()
        sim = prepRegSimulator()

    return enc, sim


def showImage(img, img_name, to_debug):
    # Show images only if we are in debug mode:
    if to_debug:
        cv2.imshow(img_name, img)
        cv2.waitKey(0)


def capGazeCoords(gaze_x, gaze_y, patch_size, image):
    # Make sure we don't exceed the bounds of the image.

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


def getGazeContigImg(image, gaze_x, gaze_y, vis_representation, shape_to_crop, patch_size, enc, sim, to_debug):
    cropped_edge_img = None

    # For debugging:
    # print(f"Current gaze location: {gaze_x}, {gaze_y}")

    # Closed eye condition - return black screen:
    if gaze_x == -32768.0 and gaze_y == -32768.0:
        return np.zeros(image.shape, dtype=np.uint8)

    gaze_x, gaze_y = capGazeCoords(gaze_x, gaze_y, patch_size, image)

    # Crop the image at the gaze coordinates:
    # (use a copy of the image so you don't change the original image)
    # (convert to ints because gaze coords are floats)

    dl = True
    if dl:
        crop_img = image[int(gaze_y - patch_size): int(gaze_y + patch_size),
                   int(gaze_x - patch_size): int(gaze_x + patch_size)].copy()

        if vis_representation == 'dl_jaap':
            cropped_edge_img = jaapPredict(crop_img, enc, sim)
        if vis_representation == 'dl_ash':
            cropped_edge_img = ashPredict(crop_img, enc, sim, toggle=False)

        # Convert back to 3 channels to prevent mismatch of shape size:
        cropped_edge_img = cv2.cvtColor(cropped_edge_img, cv2.COLOR_GRAY2RGB)
        # Make sure shape of array will be eventually compatible with the black canvas:
        # cropped_edge_img = (cropped_edge_img * 255).astype(np.uint8)

        showImage(cropped_edge_img, 'Resulting image', to_debug)
    else:
        if "circle" in shape_to_crop:
            # Localise circle on image:
            mask = np.zeros_like(image)
            mask = cv2.circle(mask, (int(gaze_x), int(gaze_y)), patch_size, (255, 255, 255), -1)
            showImage(mask, 'Localise circle', to_debug)

            # Crop down the circular region to only the area of interest:
            crop_mask_img = mask[int(gaze_y - patch_size): int(gaze_y + patch_size),
                       int(gaze_x - patch_size): int(gaze_x + patch_size)].copy()
            showImage(crop_mask_img, 'Tight circle region', to_debug)

            if shape_to_crop == 'circle_opt1':
                # Fill in white with actual image values:
                image_cropped = image[int(gaze_y - patch_size): int(gaze_y + patch_size),
                           int(gaze_x - patch_size): int(gaze_x + patch_size)].copy()
                crop_img = cv2.bitwise_and(image_cropped, crop_mask_img)
                showImage(crop_img, 'Filled circle region', to_debug)

                # Get the edges on the cropped portion of the image:
                if vis_representation == 'sobel':
                    cropped_edge_img = getSobelEdges(crop_img)
                if vis_representation == 'canny':
                    cropped_edge_img = getCannyEdges(crop_img)
                showImage(cropped_edge_img, 'Edges', to_debug)

            if shape_to_crop == 'circle_opt2':
                # Crop out a square:
                crop_img = image[int(gaze_y - patch_size): int(gaze_y + patch_size),
                           int(gaze_x - patch_size): int(gaze_x + patch_size)].copy()

                if vis_representation == 'sobel':
                    cropped_edge_img = getSobelEdges(crop_img)
                if vis_representation == 'canny':
                    cropped_edge_img = getCannyEdges(crop_img)
                showImage(cropped_edge_img, 'Edge detection in square', to_debug)

                # Fill in white with actual image values:
                cropped_edge_img = cv2.bitwise_and(cropped_edge_img, crop_mask_img)
                showImage(cropped_edge_img, 'Filled circle region', to_debug)

        if shape_to_crop == 'square':
            crop_img = image[int(gaze_y - patch_size): int(gaze_y + patch_size),
                       int(gaze_x - patch_size): int(gaze_x + patch_size)].copy()

            # Get the edges on the cropped portion of the image:
            if vis_representation == 'sobel':
                cropped_edge_img = getSobelEdges(crop_img)
            if vis_representation == 'canny':
                cropped_edge_img = getCannyEdges(crop_img)
            showImage(cropped_edge_img, 'Edges', to_debug)

    # Create a black canvas of size of the image:
    black_canvas = np.zeros(image.shape, dtype=cropped_edge_img.dtype)
    # Paste the cropped out region onto the black canvas:
    black_canvas[int(gaze_y - patch_size): int(gaze_y + patch_size),
               int(gaze_x - patch_size): int(gaze_x + patch_size)] = cropped_edge_img
    updated_img_to_show = black_canvas

    if cropped_edge_img.dtype != 'uint8':
        # Convert float32 to uint8 because PyGame requires it:
        img_normalized = np.clip(updated_img_to_show, 0, 1)
        updated_img_to_show = (img_normalized * 255).astype(np.uint8)

    # For testing this function without the eyetracker:
    showImage(updated_img_to_show, 'Resulting image', to_debug)

    return updated_img_to_show


def troubleshootGazeContig():
    # This function is for testing the methods which obtain a gaze-contingent image without the eyetracker.
    # You can call it on its own.

    img = cv2.imread('images/img_1.jpg')
    x = getGazeContigImg(img, 400, 600, 'sobel', 'circle_opt2', 128, None, None, True)
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
# troubleshootGazeContig()