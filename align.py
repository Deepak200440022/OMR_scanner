import cv2
from omr_utils import show, find_longest_horizontal_line
from scipy import ndimage
import sys


def align_sheet(img):
    """
    Aligns a scanned OMR sheet by detecting and correcting its skew angle.

    Steps:
        1. Converts the input image to grayscale.
        2. Applies Otsu’s thresholding to create a binary image.
        3. Performs Canny edge detection.
        4. Detects the longest horizontal line within a tolerance angle.
        5. Rotates the image to align it properly based on the detected angle.

    Args:
        img (numpy.ndarray): Input BGR image of the scanned OMR sheet.

    Returns:
        numpy.ndarray: Aligned OMR sheet image with corrected skew.
    """
    t_lower = 50
    t_upper = 150
    aperture_size = 5

    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edge = cv2.Canny(im_bw, t_lower, t_upper, apertureSize=aperture_size)

    line, angle = find_longest_horizontal_line(img.copy(), edge, horizontal_tolerance_deg=20)
    if angle:
        img = ndimage.rotate(img, angle, reshape=False, order=1, mode='nearest')
    return img


if __name__ == "__main__":
    path = sys.argv[1]
    img = cv2.imread(path)
    show(align_sheet(img))
