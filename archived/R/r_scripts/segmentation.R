library(dplyr)
library(magick)
library(imager)
# library(spatstat)

# im <- load.image("binarizedphotos/satextract_binarized/gatos.png")
im <- load.image("satextract.png")
# im <- load.image("grayscale/spe_satextract.jpeg")

d <- as.data.frame(im)
print(d)
#Subsamble, fit a linear model
m <- sample_n(d,1e4) %>% lm(value ~ x*y,data=.) 
print(m)
#Correct by removing the trend
im.c <- im-predict(m,d)
# out <- threshold(im.c)
# plot(out)

#watershed
bg <- (!threshold(im.c,"10%"))
fg <- (threshold(im.c,"90%"))
imlist(fg,bg) #%>% plot(layout="row")

#Build a seed image where fg pixels have value 2, bg 1, and the rest are 0
seed <- bg+2*fg
# plot(seed)

#propagate bg and fg pixels to neighbors
edges <- imgradient(im,"xy") %>% enorm
p <- 1/(1+edges)
plot(p)

#watershed transform
ws <- (watershed(seed,p)==1)
plot(ws)

#fill holes
ws <- bucketfill(ws,1,1,color=2) %>% {!( . == 2) }
plot(ws)

#remove spurious
cleaned <- fill(ws,30) #%>% plot
cleaned <- clean(cleaned,20) %>% plot
# border <- highlight(cleaned)
imager::save.image(cleaned, "border-satextract.png")

# layout(matrix(1:4,2,2))
# clean(ws,0) %>% plot(main="fill=0")
# clean(ws,10) %>% plot(main="fill=10")
# clean(ws,20) %>% plot(main="fill=20")
# clean(ws,50) %>% plot(main="fill=50")



