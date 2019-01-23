# setting the working directory
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

# function to load the label files
load_label_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
}

# loading the image files
train = load_image_file("train-images-idx3-ubyte")
test  = load_image_file("t10k-images-idx3-ubyte")

# loading the labels
train.y = load_label_file("train-labels-idx1-ubyte")
test.y  = load_label_file("t10k-labels-idx1-ubyte")

# lineaerly transforming the grey scale image i.e. between 0 and 255 to 0 and 1
train.x <- data.matrix(train/255)
test <- data.matrix(test/255)

# verifying the distribution of the digit labels in train dataset
print(table(train.y))
# verifying the distribution of the digit labels in test dataset
print(table(test.y))

# including the required mxnet library 
library(mxnet)

# defining the input layer in the network architecture
data <- mx.symbol.Variable("data")

# 1st convolutional layer
conv_1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)

# tanh activation
tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")

#max pooling 
pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

# 2nd convolutional layer
conv_2 <- mx.symbol.Convolution(data = pool_1, kernel = c(5, 5), num_filter = 50)

# 2nd tanh activation
tanh_2 <- mx.symbol.Activation(data = conv_2, act_type = "tanh")

# max pooling
pool_2 <- mx.symbol.Pooling(data=tanh_2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))

#flattening the data
flatten <- mx.symbol.Flatten(data = pool_2)

# 1st fully connected layer
fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)

# 3rd tanh activation
tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "tanh")

# 2nd fully connected layer
fc_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 40)

# Softmax activation to output so as to get probabilities.
softmax <- mx.symbol.SoftmaxOutput(data = fc_2)

# defining that the experiment should run on cpu
devices <- mx.cpu(4)

# setting the seed for the experiment so as to ensure that the results are reproducible
mx.set.seed(0)

# building the model with the network architecture defined above
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                     ctx=devices, num.round=50, array.batch.size=100,array.layout = "rowmajor",
                                     learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))

# making predictions on the test dataset
preds <- predict(model, test)
# verifying the predicted output
print(dim(preds))
# getting the label for each observation in test dataset; the predicted class is the one with highest probability
pred.label <- max.col(t(preds)) - 1

# observing the distribution of predicted labels in the test dataset
print(table(pred.label))

# including the rfUtilities library so as to use accuracy function
library(rfUtilities)

# obtaining the performance of the model
print(accuracy(pred.label,test.y))

# printing the network architecture
graph.viz(model$symbol)