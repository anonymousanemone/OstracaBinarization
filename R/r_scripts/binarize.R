library(magick)
library(image.binarization)
library(imager)
library(imagerExtra)

#read in grayscale image

file_num <- "8"
# file <- paste("processed_images/grayscale_bulk/gray-",file_num,".png", sep="")
file <- paste("processed_images/grayscale_bulk/satextract-",file_num,".png", sep="")
img <- image_read(file)


#ThresholdTriclass - also pretty good
im <- load.image(file)
im <- ThresholdTriclass(im, repeatnum = 1)
imager::save.image(cimg(im), "processed_images/binarized/threshtri.png")

#otsu - best with saturation extraction
binary <- image_binarization(img, type = "otsu")
image_write(binary, path = "processed_images/binarized/otsu.png", format = "png")

#gatos - overall solid
binary <- image_binarization(img, type = "gatos")
image_write(binary, path = "processed_images/binarized/gatos.png", format = "png")

#isauvola, wolf, nick - overall mediocre
binary <- image_binarization(img, type = "isauvola")
image_write(binary, path = "processed_images/binarized/isauvola.png", format = "png")
binary <- image_binarization(img, type = "wolf")
image_write(binary, path = "processed_images/binarized/wolf.png", format = "png")
binary <- image_binarization(img, type = "nick")
image_write(binary, path = "processed_images/binarized/nick.png", format = "png")