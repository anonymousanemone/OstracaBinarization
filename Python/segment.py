from PIL import Image
import numpy as np
import doxapy
import pandas as pd
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
    cv.resize(img, None, fx = 0.75, fy = 0.75)
    assert img is not None, "file could not be read, check with os.path.exists()"
    # gray = cv.cvtColor(img, cv.COLOR_RGB2HSV)[:, :, 1]
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # print(gray)

    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    cv.imshow("threshold", thresh)
    cv.waitKey(0)
    
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
    plt.imshow(markers)
    # cv.waitKey(0)
    img[markers == -1] = [255,0,0]
    cv.imshow("final", img)
    cv.waitKey(0)

""" https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html """
def watershed2(file):
    image = cv.imread(file)
    cv.resize(image, None, fx = 0.75, fy = 0.75)
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
def watershed4(image):
    return

""" https://pyimagesearch.com/2015/11/02/watershed-opencv/ """
def watershed5(image):
    return

""" from the R original """
def watershed6(image):
    """
    m <- sample_n(d,1e4) %>% lm(value ~ x*y,data=.) 
    gray.c <- gray-predict(m,d)
    bg <- (!threshold(gray.c,"10%"))
    fg <- (threshold(gray.c,"90%"))
    imlist(fg,bg)
    seed <- bg+2*fg
    edges <- imgradient(gray,"xy") %>% enorm
    p <- 1/(1+edges)
    ws <- (watershed(seed,p)==1)
    ws <- bucketfill(ws,1,1,color=2) %>% {!( . == 2) } 
    ws <- fill(ws,30) # closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    ws <- clean(ws,20) %>% plot""" # opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
     
    img = cv.imread(file)
    cv.resize(img, None, fx = 0.75, fy = 0.75)
    assert img is not None, "file could not be read, check with os.path.exists()"
    # gray = cv.cvtColor(img, cv.COLOR_RGB2HSV)[:, :, 1]
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # print(gray)

    #d as dataframe of img
    d = pd.DataFrame(gray)
    m = d.sample(n=int(1e4))
    # Fit linear model: value ~ x * y
    X = m[['x', 'y']]
    X_interaction = X['x'] * X['y']  # Create interaction term
    X['x*y'] = X_interaction
    y = m['value']

    model = LinearRegression().fit(X, y)

    # Predict gray using the model
    gray_pred = model.predict(d[['x', 'y']].assign(**{'x*y': d['x'] * d['y']}))

    # Calculate gray.c
    gray_c = gray - gray_pred

    # Apply threshold
    bg = gray_c <= np.percentile(gray_c, 10)  # Background threshold (10%)
    fg = gray_c >= np.percentile(gray_c, 90)  # Foreground threshold (90%)
    
    # Marker labelling
    ret, markers = cv.connectedComponents(fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    markers = cv.watershed(img,markers)
    plt.imshow(markers)
    # cv.waitKey(0)
    img[markers == -1] = [255,0,0]
    cv.imshow("final", img)
    cv.waitKey(0)

     

""" https://www.geeksforgeeks.org/image-segmentation-using-pythons-scikit-image-module/# """
def segment5(image):
    return

def edgedetect(image): 
# Read the original image
    img = cv.imread(image) 
    # Display original image
    cv.imshow('Original', img)
    cv.waitKey(0)
    
    # Convert to graycsale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv.GaussianBlur(img_gray, (3,3), 0) 
    
    # Sobel Edge Detection
    sobelx = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    cv.imshow('Sobel X', sobelx)
    cv.waitKey(0)
    cv.imshow('Sobel Y', sobely)
    cv.waitKey(0)
    cv.imshow('Sobel X Y using Sobel() function', sobelxy)
    cv.waitKey(0)
    
    # Canny Edge Detection
    edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv.imshow('Canny Edge Detection', edges)
    cv.waitKey(0)
    
    cv.destroyAllWindows()

def grayenhance1(image):
    img = cv.imread(image)
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    cv.imshow("original",img)
    cv.waitKey(0)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(25,25))
    # Top Hat Transform
    topHat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    cv.imshow("tophat",topHat)
    cv.waitKey(0)
    # Black Hat Transform
    blackHat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
    cv.imshow("blackhat",blackHat)
    cv.waitKey(0)
    res = img + topHat - blackHat
    cv.imshow("res",res)
    cv.waitKey(0)
    # cv.destroyAllWindows()

image = "original-1-7.JPG"
grayenhance1(image)
watershed1(image)