library(magick)
library(imager)
library(imagerExtra)

#blurring then binarizing - it's a thought for sure
# img <- image_read("otsu.png")
# img <- image_blur(img, 10, 3)
# img <- image_threshold(img, type = "white", threshold = "80%")
# img <- image_threshold(img, type = "black", threshold = "90%")
# image_browse(img)

# # #denoiseDCT - does not work? 
# g <- load.image("scb_satextract.png")
# plot(g, main = "Original")
# DenoiseDCT(g, 0.1) %>% plot(., main = "Denoised (8x8 window)")

# #ThresholdTriclass
# tt <- load.image("binarizedphotos/satextract_binarized/gatos.png")
# tt <- ThresholdTriclass(tt, repeatnum = 1)
# plot(tt)
# imager::save.image(cimg(tt), "thresholdtriclass4.png")

# #threshhold adaptive - nfg
# papers <- load.image("grayscale/satextract.jpeg")
# layout(matrix(1:2,1,2))
# plot(papers, main = "Original")
# hello <- ThresholdAdaptive(papers, 0.1, windowsize = 17, range = c(0,1))
# plot(hello, main = "Binarizesd")
# imager::save.image(cimg(hello), "thresholdadaptive.png")

# #fuzzy thresholding - nfg
# g <- load.image("binarizedphotos/satextract_binarized/gatos.png")
# layout(matrix(1:2,1,2))
# plot(g, main = "Original")
# ThresholdFuzzy(g) %>% plot(main = "Fuzzy Thresholding")

#multilevel thresholding - nfg
# g <- load.image("grayscale/satextract.jpeg")
# layout(matrix(1:2,1,2))
# ThresholdML(g, k = 3) %>% plot(main = "Level of Thresholds: 3")
# ThresholdML(g, thr = c(0.2, 0.4, 0.6)) %>% plot(main = "Thresholds: 0.2, 0.4, and 0.6")


