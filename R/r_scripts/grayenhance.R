library(magick)
library(imager)
library(imagerExtra)

# file_num <- "9"
# file <- paste("source_images/original-",file_num,".JPG", sep="")
# # #magick grayscale converter
# img <- image_read(file)
# img <- image_convert(img, format = "PGM", colorspace = "Gray")
# # image_browse(img)
# write <- paste("processed_images/grayscale_bulk/gray-",file_num,".png", sep="")
# image_write(img, path=write, format = "png")
# 
# #saturation extraction
# im <- load.image(file)
# sherd.hsl <- RGBtoHSL(im)
# chan <- channels(sherd.hsl) #Extract the channels as a list of images
# names(chan) <- c("H","S","L")
# graysat <- chan[["S"]]
# # graysat <- channels(sherd.hsl)[[2]] #extracts channel 2 ("S") of HSL
# # plot(graysat)
# write <- paste("processed_images/grayscale_bulk/satextract-",file_num,".png", sep="")
# imager::save.image(graysat,write)

#simplest color balance
g <- load.image("satextract.png")
# g <- graysat
BalanceSimplest(g, 1, 1) %>% plot(main = "sleft = 1, sright = 1")
scb <-BalanceSimplest(g, 1, 1)
plot(scb)
imager::save.image(scb,"scb_satextract.png")

# #screened poisson equation - not for satextract
# papers <- load.image("gray.png")
# layout(matrix(1:2, 1, 2))
# plot(papers, main = "Original")
# SPE(papers, 0.1) %>% plot(main = "SPE (lamda = 0.1)")
# spe <- SPE(papers, 0.1)
# imager::save.image(spe,"spe_gray.png")

