library(magick)

#read
binarize <- image_read("otsu.png")
border <- image_read("border-satextract.png")

#modify border
border <- image_negate(border)
border <- image_blur(border, 10, 15)
border = image_threshold(border, type = "black", threshold = "90%")
border <- image_transparent(border, "white", fuzz = 0)
# image_browse(border)

# #composite images
img <- image_composite(border, binarize, operator = "in")
image_browse(img)

#write to file
image_write(img, path = "finished-satextract.png", format = "png")
