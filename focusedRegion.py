import numpy as np
import cv2
from edgeDetection import getSobelEdges, getCannyEdges, generatePhosphenesForED
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


def launchAccordingVisRepMethod(crop_img, enc, sim, to_debug):
    # Get the visual representation according to the method desired:

    if "ed_" in vis_representation:
        if vis_representation == 'ed_sobel':
            cropped_edge_img = getSobelEdges(crop_img)

        if vis_representation == 'ed_canny':
            cropped_edge_img = getCannyEdges(crop_img)

        showImage(cropped_edge_img, 'ED on patch', to_debug)
        # Generate phosphene representation for ED methods:
        cropped_edge_img = generatePhosphenesForED(cropped_edge_img, sim, toggle=False)

    if "dl_" in vis_representation:
        if vis_representation == 'dl_jaap':
            cropped_edge_img = jaapPredict(crop_img, enc, sim)
        if vis_representation == 'dl_ash':
            cropped_edge_img = ashPredict(crop_img, enc, sim, toggle=False)

    # Convert back to 3 channels to prevent mismatch of shape size:
    cropped_edge_img = cv2.cvtColor(cropped_edge_img, cv2.COLOR_GRAY2RGB)

    return cropped_edge_img


def getCroppedPatchRegion(image, gaze_y, gaze_x):
    # Crop out a square patch of the image about the gaze x, y coordinates given:
    crop_img = image[int(gaze_y - patch_size): int(gaze_y + patch_size),
               int(gaze_x - patch_size): int(gaze_x + patch_size)].copy()
    return crop_img


def convertFloat32ToUint8(img):
    # Convert a given img of dtype float32 to an uint8 image:
    img_normalized = np.clip(img, 0, 1)
    converted_img = (img_normalized * 255).astype(np.uint8)

    return converted_img

def getGazeContigImg(image, gaze_x, gaze_y, enc, sim, to_debug):
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

    if "circle" in shape_to_crop:
        # Localise circle on image:
        mask = np.zeros_like(image)
        mask = cv2.circle(mask, (int(gaze_x), int(gaze_y)), patch_size, (255, 255, 255), -1)
        showImage(mask, 'Localise circle', to_debug)

        # Crop down the circular region to only the area of interest:
        crop_mask_img = getCroppedPatchRegion(mask, gaze_y, gaze_x)
        showImage(crop_mask_img, 'Tight circle region', to_debug)

        if shape_to_crop == 'circle_opt1':
            # Crop out a square:
            crop_img = getCroppedPatchRegion(image, gaze_y, gaze_x)
            # Fill in white with actual image values:
            crop_img = cv2.bitwise_and(crop_img, crop_mask_img)
            showImage(crop_img, 'Filled circle region', to_debug)

            # Get the edges on the cropped portion of the image:
            crop_vis_rep_img = launchAccordingVisRepMethod(crop_img, enc, sim, to_debug)
            showImage(crop_vis_rep_img, 'Edges', to_debug)

        if shape_to_crop == 'circle_opt2':
            # Crop out a square:
            crop_img = getCroppedPatchRegion(image, gaze_y, gaze_x)
            cropped_edge_img = launchAccordingVisRepMethod(crop_img, enc, sim, to_debug)
            showImage(cropped_edge_img, 'Edge detection in square', to_debug)

            # Convert float32 to uint8 because PyGame requires it:
            converted_img = convertFloat32ToUint8(cropped_edge_img)
            # Fill in white with actual image values:
            crop_vis_rep_img = cv2.bitwise_and(converted_img, crop_mask_img, to_debug)
            showImage(crop_vis_rep_img, 'Filled circle region', to_debug)

    if shape_to_crop == 'square':
        # First get a cropped out patch about the gaze coordinates:
        crop_img = getCroppedPatchRegion(image, gaze_y, gaze_x)
        # Next use the according visual representation method on this result:
        crop_vis_rep_img = launchAccordingVisRepMethod(crop_img, enc, sim)
        showImage(crop_vis_rep_img, 'Edges', to_debug)

    # Create a black canvas of size of the image:
    black_canvas = np.zeros(image.shape, dtype=crop_vis_rep_img.dtype)
    # Paste the cropped out region onto the black canvas:
    black_canvas[int(gaze_y - patch_size): int(gaze_y + patch_size),
               int(gaze_x - patch_size): int(gaze_x + patch_size)] = crop_vis_rep_img
    updated_img_to_show = black_canvas

    if crop_vis_rep_img.dtype != 'uint8':
        # Convert float32 to uint8 because PyGame requires it:
        updated_img_to_show = convertFloat32ToUint8(updated_img_to_show)

    # For testing this function without the eyetracker:
    showImage(updated_img_to_show, 'Resulting image', to_debug)

    return updated_img_to_show


def troubleshootGazeContig():
    # This function is for testing the methods which obtain a gaze-contingent image without the eyetracker.
    # You can call it on its own.

    encoder, simulator = initialisation_step()

    img = cv2.imread('images/img_1.jpg')
    x = getGazeContigImg(img, 1282, 652, encoder, simulator, True)
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
