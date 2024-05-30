library(magick)
library(image.binarization)
library(imager)
library(dplyr)

#read image - using imager
path <- "source_images/original-"
file_num <- ""
file_type <- ".JPG"
f <- paste(path,file_num,file_type, sep="")
img <- load.image(f)

#convert to grayscale - saturation extraction - imager
sherd.hsl <- RGBtoHSL(img)
gray <- channels(sherd.hsl)[[2]] #extracts channel 2 ("S") of HSL
m_gray <- cimg2magick(gray)

# plot(img)

#binarize - magick
binary <- image_binarization(m_gray, type = "otsu")

#watershed form background - imager
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
border <- image_negate(border)
border <- image_blur(border, 10, 15)
border <- image_threshold(border, type = "black", threshold = "90%")
border <- image_transparent(border, "white", fuzz = 0)
combo <- image_composite(border, binary, operator = "in")
combo <- image_flop(combo)
image_browse(combo)

#write to file
path <- "facsimiles/facsimile-"
file_type <- ".png"
f <- paste(path,file_num,file_type, sep="")
image_write(combo, path=f, format = "png")



