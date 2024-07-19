from PIL import Image
import numpy as np
import doxapy
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import data
from skimage.color import rgb2hsv
import colorsys

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


""" https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html """
def watershed1(file):
    img = cv.imread(file)
    plt.imshow(img)
    plt.waitforbuttonpress()
    assert img is not None, "file could not be read, check with os.path.exists()"
    # gray = cv.cvtColor(img, cv.COLOR_RGB2HSV)[:, :, 1]
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    print(gray)
    plt.imshow(gray)
    plt.waitforbuttonpress()
    
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # plt.imshow(thresh)
    # plt.waitforbuttonpress()
    
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
     # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv.watershed(img,markers)
    plt.show()
    img[markers == -1] = [255,0,0]
    plt.imshow(img)
    plt.waitforbuttonpress()

""" https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html """
def watershed2(img):
    image = cv.imread(file)

    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)

    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray)
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[2].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()


""" https://pypi.org/project/Watershed/ """
def watershed3(image):
    return

""" https://github.com/manoharmukku/watershed-segmentation/blob/master/watershed.py """
def watershed3(image):
    return

""" https://pyimagesearch.com/2015/11/02/watershed-opencv/ """
def watershed4(image):
    return

image = "original-1-7.JPG"
# generate_facsimile(image)
watershed1(image)