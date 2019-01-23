#including the required libraries
library("readr")
library("stringr")
library("stringi")
library("mxnet")
library("languageR")

# Using the languageR's ALICE'S ADVENTURES IN WONDERLAND book text and loading it into memory  
data(alice)

# Next we transform the test into feature vectors that is fed into the RNN model. 
# The make_data function reads the dataset, cleans it of any non-alphanumeric characters, 
# splits it into individual characters and groups it into sequences of length seq.len. 
# In this case, seq.len is set to 100

make_data <- function(txt, seq.len = 32, dic=NULL) {
  
  text_vec <- as.character(txt)
  text_vec <- stri_enc_toascii(str = text_vec)
  text_vec <- str_replace_all(string = text_vec, pattern = "[^[:print:]]", replacement = "")
  text_vec <- strsplit(text_vec, '') %>% unlist
  
  if (is.null(dic)) {
    char_keep <- sort(unique(text_vec))
  } else char_keep <- names(dic)[!dic == 0]
  
# Remove terms not part of dictionary
  text_vec <- text_vec[text_vec %in% char_keep]
  
# Building a dictionary
  dic <- 1:length(char_keep)
  names(dic) <- char_keep
  
# reversing the dictionary
  rev_dic <- names(dic)
  names(rev_dic) <- dic
  
# Adjust by -1 to have a 1-lag for labels
  num.seq <- (length(text_vec) - 1) %/% seq.len
  
  features <- dic[text_vec[1:(seq.len * num.seq)]] 
  labels <- dic[text_vec[1:(seq.len*num.seq) + 1]]
  
  features_array <- array(features, dim = c(seq.len, num.seq))
  labels_array <- array(labels, dim = c(seq.len, num.seq))
  
  return (list(features_array = features_array, labels_array = labels_array, dic = dic, rev_dic = rev_dic))
}

# setting the sequence length as 100
seq.len <- 100

# building the long sequence of text from individual words in alice data character vector
alice_in_wonderland<-paste(alice,collapse=" ") 

# calling the make_data function on the alice_in_wonderland text 
# observe that seq.len and an empty dictionary is passed as input
# seq.len dictates the context that is the number of characters that the RNN need to look back
# inorder to generate the next character
# during training seq.len is utilize to get the right weights
data_prep <- make_data(alice_in_wonderland, seq.len = seq.len, dic=NULL)

#examining the prepared data
print(str(data_prep))

# Viewing the features array
View(data_prep$features_array)

# Viewing the labels array
View(data_prep$labels_array)

# printing the dictionary - the unique characters
print(data_prep$dic)

# printing the indexes of the characters
print(data_prep$rev_dic)

# Fetch the features and labels for training the model
# split the data into training and evaluation in 90:10 ratio
X <- data_prep$features_array
Y <- data_prep$labels_array
dic <- data_prep$dic
rev_dic <- data_prep$rev_dic
vocab <- length(dic)

samples <- tail(dim(X), 1)
train.val.fraction <- 0.9

X.train.data <- X[, 1:as.integer(samples * train.val.fraction)]
X.val.data <- X[, -(1:as.integer(samples * train.val.fraction))]

X.train.label <- Y[, 1:as.integer(samples * train.val.fraction)]
X.val.label <- Y[, -(1:as.integer(samples * train.val.fraction))]

train_buckets <- list("100" = list(data = X.train.data, label = X.train.label))
eval_buckets <- list("100" = list(data = X.val.data, label = X.val.label))

train_buckets <- list(buckets = train_buckets, dic = dic, rev_dic = rev_dic)
eval_buckets <- list(buckets = eval_buckets, dic = dic, rev_dic = rev_dic)

# Create iterators for training and evaluation datasets
vocab <- length(eval_buckets$dic)

batch.size <- 32

train.data <- mx.io.bucket.iter(buckets = train_buckets$buckets, batch.size = batch.size, 
                                data.mask.element = 0, shuffle = TRUE)

eval.data <- mx.io.bucket.iter(buckets = eval_buckets$buckets, batch.size = batch.size,
                               data.mask.element = 0, shuffle = FALSE)

# Creating the RNN model, it is a multi-layer RNN for sampling from character-level language models. 
# It has a one-to-one model configuration since for each character, we want to predict the next one. 
# For a sequence of length 100, there are also 100 labels, corresponding the same sequence of characters 
# but offset by a position of +1. 
# The parameters output_last_state is set to TRUE 
# this is to access the state of the RNN cells when performing inference.
# we can see lstm cells are used
rnn_graph_one_one <- rnn.graph(num_rnn_layer = 3, 
                               num_hidden = 96,
                               input_size = vocab,
                               num_embed = 64, 
                               num_decode = vocab,
                               dropout = 0.2, 
                               ignore_label = 0,
                               cell_type = "lstm",
                               masking = F,
                               output_last_state = T,
                               loss_output = "softmax",
                               config = "one-to-one")

# visualizing the RNN model
graph.viz(rnn_graph_one_one, type = "graph",
          graph.height.px = 650, shape=c(500, 500))

# setting that confirms that the device on which the code to be executed is CPU
devices <- mx.cpu()

# initializing the weights of the network through Xavier initializer
initializer <- mx.init.Xavier(rnd_type = "gaussian", factor_type = "avg", magnitude = 3)

# using the adadelta optimizer to update the weights in the network thru the learning process
optimizer <- mx.opt.create("adadelta", rho = 0.9, eps = 1e-5, wd = 1e-8,
                           clip_gradient = 5, rescale.grad = 1/batch.size)

# setting up logging of metric
logger <- mx.metric.logger()
epoch.end.callback <- mx.callback.log.train.metric(period = 1, logger = logger)
batch.end.callback <- mx.callback.log.train.metric(period = 50)

# defining a custom measurement function
mx.metric.custom_nd <- function(name, feval) {
  init <- function() {
    c(0, 0)
  }
  update <- function(label, pred, state) {
    m <- feval(label, pred)
    state <- c(state[[1]] + 1, state[[2]] + m)
    return(state)
  }
  get <- function(state) {
    list(name=name, value = (state[[2]] / state[[1]]))
  }
  ret <- (list(init = init, update = update, get = get))
  class(ret) <- "mx.metric"
  return(ret)
}

# Perplexity is a measure of how variable a prediction model is. 
# It perplexity is a measure of prediction error
# Defining a function to compute the error
mx.metric.Perplexity <- mx.metric.custom_nd("Perplexity", function(label, pred) {
  label <- mx.nd.reshape(label, shape = -1)
  label_probs <- as.array(mx.nd.choose.element.0index(pred, label))
  batch <- length(label_probs)
  NLL <- -sum(log(pmax(1e-15, as.array(label_probs)))) / batch
  Perplexity <- exp(NLL)
  return(Perplexity)
})

# executing the model creation
# observe that in this project we are runing it for 20 iterations
model <- mx.model.buckets(symbol = rnn_graph_one_one,
                          train.data = train.data, eval.data = eval.data, 
                          num.round = 20, ctx = devices, verbose = TRUE,
                          metric = mx.metric.Perplexity, 
                          initializer = initializer, optimizer = optimizer, 
                          batch.end.callback = NULL, 
                          epoch.end.callback = epoch.end.callback)

# saving the model for later use
mx.model.save(model, prefix = "one_to_one_seq_model", iteration = 20)

# loading the model from the disk
# using the loaded modelto do inference and sample text character by character 
# the generated text is expected to be similar to the training data
set.seed(0)
model <- mx.model.load(prefix = "one_to_one_seq_model", iteration = 20)

internals <- model$symbol$get.internals()
sym_state <- internals$get.output(which(internals$outputs %in% "RNN_state"))
sym_state_cell <- internals$get.output(which(internals$outputs %in% "RNN_state_cell"))
sym_output <- internals$get.output(which(internals$outputs %in% "loss_output"))
symbol <- mx.symbol.Group(sym_output, sym_state, sym_state_cell)

# providing the seed character to start the text with
infer_raw <- c("e")
infer_split <- dic[strsplit(infer_raw, '') %>% unlist]
infer_length <- length(infer_split)

infer.data <- mx.io.arrayiter(data = matrix(infer_split), label = matrix(infer_split),  
                              batch.size = 1, shuffle = FALSE)

infer <- mx.infer.rnn.one(infer.data = infer.data, 
                          symbol = symbol,
                          arg.params = model$arg.params,
                          aux.params = model$aux.params,
                          input.params = NULL, 
                          ctx = devices)

pred_prob <- as.numeric(as.array(mx.nd.slice.axis(
  infer$loss_output, axis = 0, begin = infer_length-1, end = infer_length)))
pred <- sample(length(pred_prob), prob = pred_prob, size = 1) - 1
predict <- c(predict, pred)

for (i in 1:200) {
  
  infer.data <- mx.io.arrayiter(data = as.matrix(pred), label = as.matrix(pred),  
                                batch.size = 1, shuffle = FALSE)
  
  infer <- mx.infer.rnn.one(infer.data = infer.data, 
                            symbol = symbol,
                            arg.params = model$arg.params,
                            aux.params = model$aux.params,
                            input.params = list(rnn.state = infer[[2]], 
                                                rnn.state.cell = infer[[3]]), 
                            ctx = devices)
  
  pred_prob <- as.numeric(as.array(infer$loss_output))
  pred <- sample(length(pred_prob), prob = pred_prob, size = 1, replace = T) - 1
  predict <- c(predict, pred)
}

# post processing the predicted characters and merging them together into one sentence
predict_txt <- paste0(rev_dic[as.character(predict)], collapse = "")
predict_txt_tot <- paste0(infer_raw, predict_txt, collapse = "")

# printing the predicted text
print(predict_txt_tot)