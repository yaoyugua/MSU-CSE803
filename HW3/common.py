import cv2

def save_img(path, img):
    cv2.imwrite(path, img)
    print(path, "is saved!")


def display_img(img):
    cv2.imshow('Result', img)
    cv2.waitKey()


def read_img(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # if need double type, uncomment the following
    out = image.astype(float)
    return image


def read_colorimg(path):
    image = cv2.imread(path)
    # if need double type, uncomment the following
    # out = image.astype(float)
    return image
