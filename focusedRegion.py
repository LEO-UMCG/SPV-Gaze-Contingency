import cv2
import numpy as np
import pygame


def test():
    # original image
    image = cv2.imread('images/img_1.jpg')
    eye_location = [400, 300]

    height = 50
    contours = np.array(
        [[eye_location[0] - height, eye_location[1] - height], [eye_location[0] - height, eye_location[1] + height],
         [eye_location[0] + height, eye_location[1] + height], [eye_location[0] + height, eye_location[1] - height]])

    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, pts=[contours], color=(255, 255, 255))

    # apply the mask
    masked_image = cv2.bitwise_and(image, mask)

    # Draw dot: image, coords, radius, color, filled
    cv2.circle(masked_image, (eye_location[0], eye_location[1]), 10, (0, 0, 255), -1)

    # show the result
    cv2.imshow('image_masked.png', masked_image)
    cv2.waitKey(0)


def getGazeContigImg(image, gaze_x, gaze_y):
    eye_location = [gaze_x, gaze_y]
    print(f"Current gaze location: {gaze_x}, {gaze_y}")

    height = 100
    contours = np.array(
        [[eye_location[0] - height, eye_location[1] - height],
         [eye_location[0] - height, eye_location[1] + height],
         [eye_location[0] + height, eye_location[1] + height],
         [eye_location[0] + height, eye_location[1] - height]])

    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, pts=np.int32([contours]), color=(255, 255, 255))

    # apply the mask
    updated_img_to_show = cv2.bitwise_and(image, mask)

    # size = updated_img_to_show.shape[1::-1]
    # surface = pygame.image.frombuffer(updated_img_to_show.flatten(), size, 'RGB')
    # return surface.convert()
    return updated_img_to_show

# img = pygame.image.load('./images/img_1.jpg')
# img = pygame.transform.scale(img, (1000, 1800))
img = cv2.imread('images/img_1.jpg')
x = getGazeContigImg(img, 400, 600)