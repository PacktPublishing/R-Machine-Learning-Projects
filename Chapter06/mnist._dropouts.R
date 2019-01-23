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

# defining the first hidden layer with 128 neurons and also naming the layer as fc1
# passing the input data layer as input to the fc1 layer
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)

# defining the relu activation function on the fc1 output and also naming the layer as relu1
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")

# defining a 50% dropout of weights learnt
dropout1 <- mx.symbol.Dropout(data = act1, p = 0.5)

# defining the second hidden layer with 64 neurons and also naming the layer as fc2
# passing the previous dropout output as input to the fc2 layer
fc2 <- mx.symbol.FullyConnected(dropout1, name="fc2", num_hidden=64)

# defining the relu activation function on the fc2 output and also naming the layer as relu2
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")

# defining a dropout with 30% weight drop
dropout2 <- mx.symbol.Dropout(data = act2, p = 0.3)

# defining the third and final hidden layer in our network with 10 neurons and also naming the layer as fc3
# passing the previous dropout output as input to the fc3 layer
fc3 <- mx.symbol.FullyConnected(dropout2, name="fc3", num_hidden=10)

# defining the output layer with softmax activation function to obtain class probabilities 
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

# defining that the experiment should run on cpu
devices <- mx.cpu()

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