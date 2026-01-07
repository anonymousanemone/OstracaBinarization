library(magick)
library(image.binarization)
library(imager)
library(dplyr)

#read image - using imager
path <- "source_images/original-"
file_num <- "1-38"
file_type <- ".JPG"
f <- paste(path,file_num,file_type, sep="")
img <- load.image(f)
# plot(img)
m_img <- image_read(f)

#convert to grayscale - saturation extraction - imager
sherd.hsl <- RGBtoHSL(img)
gray <- channels(sherd.hsl)[[2]] #extracts channel 2 ("S") of HSL
# m_gray <- cimg2magick(gray)
imager::save.image(gray,"satextract.png")

#watershed form background - imager
# gray<- load.image("satextract-cropped.png")
d <- as.data.frame(gray)
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
ws <- fill(ws,30)
ws <- clean(ws,20) %>% plot

#put this shit together - magick
border <-cimg2magick(ws)
# image_browse(border)
border <- image_negate(border)
border <- image_blur(border, 10, 15)
border <- image_threshold(border, type = "black", threshold = "90%")
border <- image_transparent(border, "white", fuzz = 0)
image_browse(border)
m_gray <- image_convert(m_img, format = "PGM", colorspace = "Gray")
image_write(m_gray, path="gray.png", format = "png")
combo <- image_composite(image_flop(border), m_gray, operator = "in")
image_browse(combo)
image_write(combo, path="cut_gray.png", format = "png")

#binarize - magick
combo <- image_binarization(combo, type = "wolf")
image_browse(combo)

#write to file
path <- "facsimiles/facsimile-"
file_type <- ".png"
f <- paste(path,file_num,file_type, sep="")
image_write(combo, path=f, format = "png")
image_write(image_append(c(m_img, combo)), path=f, format = "png")



