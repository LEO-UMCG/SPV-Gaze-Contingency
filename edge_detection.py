import cv2
import torch
from torchvision import transforms
from spvPlayer.config import *

DEVICE = torch.device(DEVICE_TYPE)


def generate_phosphenes_for_ed(img, simulator, toggle=False):
    # This method generates phosphenes for the edge detector methods.
    # Based on ashPredict form BS_model.py:

    gray2color = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)
    transes = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Resize(size=PATCH_SIZE * 2),
        gray2color
    ])

    input_tensor = transes(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = input_batch.to(DEVICE)
        # Reshape from size [1,3,256,256] to [1,3,32,32]:
        reshaped_output = torch.nn.functional.interpolate(output, size=(32, 32), mode='bilinear', align_corners=False)
        # Scale down the number of channels from 3 to 1, so we go from [1,3,256,256] to [1,1,256,256] (i.e RGB -> Gray):
        rescaled_output = torch.mean(reshaped_output, dim=1, keepdim=True)

        if toggle:
            res = rescaled_output.clone()
            res[output == 0] = 1
            res[output == 1] = 0
            rescaled_output = res
        spv = simulator(rescaled_output)

        return spv.cpu().numpy()[0, 0]


def preprocess(img):
    # Grayscale:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur:
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # cv2.imshow('blur res', img_blur)
    # cv2.waitKey(0)
    return img_blur


def get_sobel_edges(img):
    # First convert to grayscale + blur:
    img = preprocess(img)

    # Calculate derivatives in x and y directions:
    #   ddepth=-1 means the output image has same depth as input
    sobelx = cv2.Sobel(src=img, ddepth=-1, dx=1, dy=0, ksize=3)
    sobely = cv2.Sobel(src=img, ddepth=-1, dx=0, dy=1, ksize=3)
    # Approximate the gradient by adding both the x and y direction derivatives
    edges = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    # Convert back to 3 channels to prevent mismatch of shape size:
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # cv2.imshow('sobel res', edges)
    # cv2.waitKey(0)
    return edges


def get_canny_edges(img):
    # First convert to grayscale + blur:
    img = preprocess(img)

    edges = cv2.Canny(image=img, threshold1=100, threshold2=200)
    # Convert back to 3 channels to prevent mismatch of shape size:
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # cv2.imshow('canny res', edges)
    # cv2.waitKey(0)
    return edges


# To test:
# img = cv2.imread('images/img_1.jpg')
# get_canny_edges(img)