import numpy as np
import cv2
from edgeDetection import getSobelEdges, getCannyEdges


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


def getGazeContigImg(image, gaze_x, gaze_y, edge_detector, shape_to_crop, patch_size):
    print(f"Current gaze location: {gaze_x}, {gaze_y}")

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
        # cv2.imshow('Localise circle', mask)
        # cv2.waitKey(0)

        # Crop down the circular region to only the area of interest:
        crop_mask_img = mask[int(gaze_y - patch_size): int(gaze_y + patch_size),
                   int(gaze_x - patch_size): int(gaze_x + patch_size)].copy()
        # cv2.imshow('Tight circle region', crop_mask_img)
        # cv2.waitKey(0)

        if shape_to_crop == 'circle_opt1':
            # Fill in white with actual image values:
            image_cropped = image[int(gaze_y - patch_size): int(gaze_y + patch_size),
                       int(gaze_x - patch_size): int(gaze_x + patch_size)].copy()
            crop_img = cv2.bitwise_and(image_cropped, crop_mask_img)
            # cv2.imshow('Tight circle region', crop_img)
            # cv2.waitKey(0)

            # Get the edges on the cropped portion of the image:
            cropped_edge_img = getSobelEdges(crop_img) if edge_detector == 'sobel' else getCannyEdges(crop_img)
            # cv2.imshow('Edges', cropped_edge_img)
            # cv2.waitKey(0)

        if shape_to_crop == 'circle_opt2':
            # Crop out a square:
            crop_img = image[int(gaze_y - patch_size): int(gaze_y + patch_size),
                       int(gaze_x - patch_size): int(gaze_x + patch_size)].copy()

            cropped_edge_img = getSobelEdges(crop_img) if edge_detector == 'sobel' else getCannyEdges(crop_img)
            # cv2.imshow('Edge detection in square', cropped_edge_img)
            # cv2.waitKey(0)

            # Fill in white with actual image values:
            cropped_edge_img = cv2.bitwise_and(cropped_edge_img, crop_mask_img)
            # cv2.imshow('Tight circle region', cropped_edge_img)
            # cv2.waitKey(0)

    if shape_to_crop == 'square':
        crop_img = image[int(gaze_y - patch_size): int(gaze_y + patch_size),
                   int(gaze_x - patch_size): int(gaze_x + patch_size)].copy()

        # Get the edges on the cropped portion of the image:
        cropped_edge_img = getSobelEdges(crop_img) if edge_detector == 'sobel' else getCannyEdges(crop_img)
        # cv2.imshow('Edges', cropped_edge_img)
        # cv2.waitKey(0)

    # Create a black canvas of size of the image:
    black_canvas = np.zeros(image.shape, dtype=np.uint8)
    # Paste the cropped out region onto the black canvas:
    black_canvas[int(gaze_y - patch_size): int(gaze_y + patch_size),
               int(gaze_x - patch_size): int(gaze_x + patch_size)] = cropped_edge_img
    updated_img_to_show = black_canvas

    # For testing this function without the eyetracker:
    # cv2.imshow('Resulting image', updated_img_to_show)
    # cv2.waitKey(0)

    return updated_img_to_show


# For testing this function without the eyetracker:
# img = cv2.imread('images/img_1.jpg')
# x = getGazeContigImg(img, 400, 600, 'sobel')

