## setting the working directory
setwd('/home/sunil/Desktop/book/chapter 6/MNIST')

# function to load image files
load_image_file = function(filename) {
  ret = list()
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  data.frame(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

# function to load label files
load_label_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
}

# load images
train = load_image_file("train-images-idx3-ubyte")
test  = load_image_file("t10k-images-idx3-ubyte")

# converting the train and test data into a format as required by LeNet
train.x <- t(data.matrix(train))
test <- t(data.matrix(test))

# loading the labels
train.y = load_label_file("train-labels-idx1-ubyte")
test.y  = load_label_file("t10k-labels-idx1-ubyte")

# linearly transforming the grey scale image i.e. between 0 and 255 to 0 and 1
train.x <- train.x/255
test <- test/255

# including the required mxnet library 
library(mxnet)

# input
data <- mx.symbol.Variable('data')

# first convolution layer
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)

# applying the tanh activation function
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")

# applying max pooling 
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)

# applying the tanh activation function again
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")

#performing max pooling again
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# flattening the data
flatten <- mx.symbol.Flatten(data=pool2)

# first fullconnected later
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)

# applying the tanh activation function
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")

# second fullconnected layer
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)

# defining the output layer with softmax activation function to obtain class probabilities 
lenet <- mx.symbol.SoftmaxOutput(data=fc2)

# transforming the train and test dataset into a format required by MxNet functions
train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))
test.array <- test
dim(test.array) <- c(28, 28, 1, ncol(test))

# setting the seed for the experiment so as to ensure that the results are reproducible
mx.set.seed(0)

# defining that the experiment should run on cpu
devices <- mx.cpu()

# building the model with the network architecture defined above
model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=devices, num.round=3, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))
# making predictions on the test dataset
preds <- predict(model, test.array)
# verifying the predicted output
dim(preds)

# getting the label for each observation in test dataset; the predicted class is the one with highest probability
pred.label <- max.col(t(preds)) - 1

# including the rfUtilities library so as to use accuracy function
library(rfUtilities)

# obtaining the performance of the model
print(accuracy(pred.label,test.y))

# printing the network architecture
graph.viz(model$symbol)