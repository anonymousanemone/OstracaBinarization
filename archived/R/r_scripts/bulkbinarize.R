###### using grayscale image, generate binary images using 12 different algorithms and save

library(magick)
library(image.binarization)

#read in grayscale image
img <- image_read("croppedgray.png")

binary <- image_binarization(img, type = "otsu")
image_write(binary, path = "binarized/otsu.png", format = "png")

binary <- image_binarization(img, type = "bernsen")
image_write(binary, path = "binarized/bernsen.png", format = "png")

binary <- image_binarization(img, type = "niblack")
image_write(binary, path = "binarized/niblack.png", format = "png")

binary <- image_binarization(img, type = "sauvola")
image_write(binary, path = "binarized/sauvola.png", format = "png")

binary <- image_binarization(img, type = "wolf")
image_write(binary, path = "binarized/wolf.png", format = "png")

binary <- image_binarization(img, type = "nick")
image_write(binary, path = "binarized/nick.png", format = "png")

binary <- image_binarization(img, type = "gatos")
image_write(binary, path = "binarized/gatos.png", format = "png")

binary <- image_binarization(img, type = "su")
image_write(binary, path = "binarized/su.png", format = "png")

binary <- image_binarization(img, type = "trsingh")
image_write(binary, path = "binarized/trsingh.png", format = "png")

binary <- image_binarization(img, type = "bataineh")
image_write(binary, path = "binarized/bataineh.png", format = "png")

binary <- image_binarization(img, type = "wan")
image_write(binary, path = "binarized/wan.png", format = "png")

binary <- image_binarization(img, type = "isauvola")
image_write(binary, path = "binarized/isauvola.png", format = "png")
