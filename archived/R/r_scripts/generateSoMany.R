library(magick)
library(image.binarization)
library(imager)
library(dplyr)

#read image - using imager
n<- 9
binarize_algorithm <- c("otsu", "bernsen", "niblack", "sauvola", 
                        "wolf", "gatos", "nick", "su", "trsingh", 
                        "bataineh", "isauvola", "wan")
#functions
file_path <- function(path, file_num, file_type){
  return(paste(path,file_num,file_type, sep=""))
}

grayify <- function(img){
  return(image_convert(img, format = "PGM", colorspace = "Gray"))
}
satextraction <- function(img){
  sherd.hsl <- RGBtoHSL(img)
  gray <- channels(sherd.hsl)[[2]] #extracts channel 2 ("S") of HSL
  return(gray)
}


for (file_num in 1:n) {
  f <- file_path("source_images/original-",file_num,".JPG")
  img <- load.image(f)
  
  
  for (j in 1:12){}
    binary <- image_binarization(img, type=binarize_algorithm[j])
    f <- paste("facsimiles/",binarize_algorithm,"-facsimile-",file_num,".png", sep="")
    image_write(combo, path=f, format = "png")
  }
}


#convert to grayscale - saturation extraction - imager
sherd.hsl <- RGBtoHSL(img)
gray <- channels(sherd.hsl)[[2]] #extracts channel 2 ("S") of HSL
m_gray <- cimg2magick(gray)
imager::save.image(gray,"satextract.png")

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

# #put this shit together - magick
# border <-cimg2magick(ws)
# border <- image_negate(border)
# border <- image_blur(border, 10, 15)
# border <- image_threshold(border, type = "black", threshold = "90%")
# border <- image_transparent(border, "white", fuzz = 0)
# # image_browse(border)
# m_img <- image_convert(image_read(f), format = "PGM", colorspace = "Gray")
# image_write(m_img, path="gray.png", format = "png")
# combo <- image_composite(image_flop(border), m_img, operator = "in")
# combo <- image_flop(combo)
# image_browse(combo)
# image_write(combo, path="cut_gray.png", format = "png")

#binarize - magick
combo <- image_binarization(combo, type = "wolf")
# image_browse(combo)

#write to file
path <- "facsimiles/facsimile-"
file_type <- ".png"
f <- paste(path,file_num,file_type, sep="")
image_write(combo, path=f, format = "png")



