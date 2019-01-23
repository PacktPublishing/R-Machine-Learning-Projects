# loading the required libraries
library(mxnet)
library(imager)

# loading the inception_bn model to memory
model = mx.model.load("/home/sunil/Desktop/book/chapter 6/Inception/Inception_BN", iteration=39)
# loading the mean image
mean.img = as.array(mx.nd.load("/home/sunil/Desktop/book/chapter 6/Inception/mean_224.nd")[["mean_img"]])

# loading the image that need to be classified
im <- load.image("/home/sunil/Desktop/book/chapter 6/image1.jpeg")

# displaying the image
plot(im)

# function to pre-process the image so as to be consumed by predict function that is using inception_bn model
preproc.image <- function(im, mean.image) {
  # crop the image
  shape <- dim(im)
  short.edge <- min(shape[1:2])
  xx <- floor((shape[1] - short.edge) / 2)
  yy <- floor((shape[2] - short.edge) / 2)
  cropped <- crop.borders(im, xx, yy)
  # resize to 224 x 224, needed by input of the model.
  resized <- resize(cropped, 224, 224)
  # convert to array (x, y, channel)
  arr <- as.array(resized) * 255
  dim(arr) <- c(224, 224, 3)
  # subtract the mean
  normed <- arr - mean.img
  # Reshape to format needed by mxnet (width, height, channel, num)
  dim(normed) <- c(224, 224, 3, 1)
  return(normed)
}

# calling the image pre-processing function on the image to be classified
normed <- preproc.image(im, mean.img)

# predicting the probabilties of labels for the image using the pre-trained model
prob <- predict(model, X=normed)

# sorting and filtering the top three labels with highest probabilities
max.idx <- order(prob[,1], decreasing = TRUE)[1:3]

# printing the ids with highest probabilities
print(max.idx)

# loading the pre-trained labels from inception_bn model 
synsets <- readLines("/home/sunil/Desktop/book/chapter 6/Inception/synset.txt")

# printing the english labels corresponding to the top 3 ids with highest probabilities
print(paste0("Predicted Top-classes: ", synsets[max.idx]))

# getting the labels for second image
im2 <- load.image("/home/sunil/Desktop/book/chapter 6/image2.jpeg")
plot(im2)
normed <- preproc.image(im2, mean.img)
prob <- predict(model, X=normed)
dim(prob)
max.idx <- order(prob[,1], decreasing = TRUE)[1:3]
max.idx
print(paste0("Predicted Top-classes: ", synsets[max.idx]))

# getting the labels for third image
im3 <- load.image("/home/sunil/Desktop/book/chapter 6/image3.jpeg")
plot(im3)
normed <- preproc.image(im3, mean.img)
prob <- predict(model, X=normed)
dim(prob)
max.idx <- order(prob[,1], decreasing = TRUE)[1:3]
max.idx
print(paste0("Predicted Top-classes: ", synsets[max.idx]))