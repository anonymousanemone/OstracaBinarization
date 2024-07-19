# https://pypi.org/project/doxapy/
from PIL import Image
import numpy as np
import doxapy
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import data
from skimage.color import rgb2hsv
import colorsys

def read_image(file):
    return np.array(Image.open(file).convert('L'))
    # return np.array(Image.open(file).convert('RGB'))


def watershed_segment(file):
   return

def borderize():
    return

def grayify():
    return

def binarize():
    return

def generate_facsimile(file):
    # Read our target image and setup an output image buffer
    grayscale_image = read_image(file)
    binary_image = np.empty(grayscale_image.shape, grayscale_image.dtype)

    # Pick an algorithm from the DoxaPy library and convert the image to binary
    facsimile = doxapy.Binarization(doxapy.Binarization.Algorithms.WOLF)
    facsimile.initialize(grayscale_image)
    facsimile.to_binary(binary_image, {"window": 75, "k": 0.2})

    # do watershed segmentation

    # Display our resulting image
    Image.fromarray(binary_image).show()

image = "original-1-7.JPG"
# generate_facsimile(image)
watershed_segment(image)



# #convert to grayscale - saturation extraction - imager
# sherd.hsl <- RGBtoHSL(img)
# gray <- channels(sherd.hsl)[[2]] #extracts channel 2 ("S") of HSL

# #watershed form background - imager
# # gray<- load.image("satextract-cropped.png")
# d <- as.data.frame(gray)
# m <- sample_n(d,1e4) %>% lm(value ~ x*y,data=.) 
# gray.c <- gray-predict(m,d)
# bg <- (!threshold(gray.c,"10%"))
# fg <- (threshold(gray.c,"90%"))
# imlist(fg,bg)
# seed <- bg+2*fg
# edges <- imgradient(gray,"xy") %>% enorm
# p <- 1/(1+edges)
# ws <- (watershed(seed,p)==1)
# ws <- bucketfill(ws,1,1,color=2) %>% {!( . == 2) }
# ws <- fill(ws,30)
# ws <- clean(ws,20) %>% plot

# #put this shit together - magick
# border <-cimg2magick(ws)
# # image_browse(border)
# border <- image_negate(border)
# border <- image_blur(border, 10, 15)
# border <- image_threshold(border, type = "black", threshold = "90%")
# border <- image_transparent(border, "white", fuzz = 0)
# image_browse(border)
# m_gray <- image_convert(m_img, format = "PGM", colorspace = "Gray")
# image_write(m_gray, path="gray.png", format = "png")
# combo <- image_composite(image_flop(border), m_gray, operator = "in")
# image_browse(combo)
# image_write(combo, path="cut_gray.png", format = "png")




