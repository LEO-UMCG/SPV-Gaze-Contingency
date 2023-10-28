import cv2


def preprocess(img):
    # Grayscale:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur:
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # cv2.imshow('blur res', img_blur)
    # cv2.waitKey(0)
    return img_blur


def getSobelEdges(img):
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


def getCannyEdges(img):
    # First convert to grayscale + blur:
    img = preprocess(img)

    edges = cv2.Canny(image=img, threshold1=100, threshold2=200)
    # Convert back to 3 channels to prevent mismatch of shape size:
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    # cv2.imshow('canny res', edges)
    # cv2.waitKey(0)
    return edges


# To test:
img = cv2.imread('images/img_1.jpg')
getCannyEdges(img)