library(tidyverse)
library(h2o)
library(rio)
library(doParallel)
library(viridis)
library(RColorBrewer)
library(tidyverse)
library(ggthemes)
library(knitr)
library(tidyverse)
library(caret)
library(caretEnsemble)
library(plotly)
library(lime)
library(plotROC)
library(pROC)

# initializing the H2O cluster in localhost under the port 54321 
# nthreads define the number of thread pools to be used, this is close to number of cpus to be used
# in our case we are saying use all CPUs
# we are also specifying the maximum memory to use by H2O cluster is 8G
localH2O = h2o.init(ip = 'localhost', port = 54321, nthreads = -1,max_mem_size = "8G")

# Detecting the available number of cores
no_cores <- detectCores() - 1
# utilizing all available cores
cl<-makeCluster(no_cores)
registerDoParallel(cl)

# setting the working directory where the data file is location 
setwd("/home/sunil/Desktop/book/chapter 7")
# loading the Rdata file and reading it into the dataframe called cc_fraud
cc_fraud<-get(load("creditcard.Rdata"))
# performing basic EDA on the dataset
# Viewing the dataframe to confirm successful load of the dataset
View(cc_fraud)
# printing the dataframe structure
print(str(cc_fraud))
# printing the class distribution
print(table(cc_fraud$Class))
# Printing the Histograms for Multivariate analysis
theme_set(theme_economist_white())
# visualization showing the relationship between variable V1 and the class
ggplot(cc_fraud, aes(x ="",y=V1, fill=Class))+ geom_boxplot()+labs(x="V1",y="")
# visualization showing the distribution of transaction amount with 
# respect to the class, it may be observed that the amount are discretized
# into 50 bins for plotting purposes
ggplot(cc_fraud,aes(x = Amount)) +
  geom_histogram(color = "#D53E4F", fill = "#D53E4F", bins = 50) +
  facet_wrap( ~ Class, scales = "free", ncol = 2)
# visualization showing the distribution of transaction time with 
# respect to the class; the transaction time is binned into 30 buckets
ggplot(cc_fraud, aes(x =Time,fill = Class))+ geom_histogram(bins = 30)+
  facet_wrap( ~ Class, scales = "free", ncol = 2)
# visualization showing the V2 variable with 
# respect to the class; the varible is binned into 30 buckets for plotting purposes
ggplot(cc_fraud, aes(x =V2, fill=Class))+ geom_histogram(bins = 30)+
  facet_wrap( ~ Class, scales = "free", ncol = 2)
# visualization showing the V3 variable with 
# respect to the class; the varible is binned into 30 buckets for plotting purposes
ggplot(cc_fraud, aes(x =V3, fill=Class))+ geom_histogram(bins = 30)+
  facet_wrap( ~ Class, scales = "free", ncol = 2)
# visualization showing the V4 variable with 
# respect to the class; the varible is binned into 30 buckets for plotting purposes
ggplot(cc_fraud, aes(x =V4,fill=Class))+ geom_histogram(bins = 30)+
  facet_wrap( ~ Class, scales = "free", ncol = 2)
# visualization showing the V6 variable with 
# respect to the class
ggplot(cc_fraud, aes(x=V6, fill=Class)) + geom_density(alpha=1/3) + scale_fill_hue()
# visualization showing the V7 variable with 
# respect to the class
ggplot(cc_fraud, aes(x=V7, fill=Class)) + geom_density(alpha=1/3) + scale_fill_hue()
# visualization showing the V8 variable with 
# respect to the class
ggplot(cc_fraud, aes(x=V8, fill=Class)) + geom_density(alpha=1/3) + scale_fill_hue()
# visualization showing the V9 variable with 
# respect to the class
ggplot(cc_fraud, aes(x=V9, fill=Class)) + geom_density(alpha=1/3) + scale_fill_hue()
# visualization showing the V10 variable with 
# respect to the class, observe we are plotting the data quantiles
ggplot(cc_fraud, aes(x ="",y=V10, fill=Class))+ geom_violin(adjust = .5,draw_quantiles = c(0.25, 0.5, 0.75))+labs(x="V10",y="")
#plotting the distribution of classes in the data
cc_fraud %>%
  ggplot(aes(x = Class)) +
  geom_bar(color = "chocolate", fill = "chocolate", width = 0.2) +
  theme_bw()
#featuring engineering with time variable in data
#printing the time variable in its current form in the dataset
print(summary(cc_fraud$Time))
# creating a new variable called day based on the seconds represented in Time variable
cc_fraud=cc_fraud %>%  
  mutate(Day = case_when(.$Time > 3600 * 24 ~ "day2",.$Time < 3600 * 24 ~ "day1"))
#visualizing the dataset post creating the new variable
View(cc_fraud%>%head())
View(cc_fraud%>%tail())
# Printing the distribution of transactions by days in which the transactions falls
print(table(cc_fraud[,"Day"]))
# creating a new variable called Time_day based on the seconds represented in Time variable
cc_fraud$Time_day <- if_else(cc_fraud$Day == "day2", cc_fraud$Time - 86400, cc_fraud$Time)
# summarizing the Time_day variable with respect to Day to understand it better
print(tapply(cc_fraud$Time_day,cc_fraud$Day,summary,simplify = FALSE))
# converting all character variables in the dataset as factors
cc_fraud<-cc_fraud%>%mutate_if(is.character,as.factor)
# splitting the time of the day into buckets and setting up a new Time_Group variable
cc_fraud=cc_fraud %>%  
  mutate(Time_Group = case_when(.$Time_day <= 38138~ "morning" ,
                                .$Time_day <= 52327~  "afternoon",
                                .$Time_day <= 69580~"evening",
                                .$Time_day > 69580~"night"))
#Visualizing the data post creating the new variable
View(head(cc_fraud))
View(tail(cc_fraud))

#visualizing the transaction count by day
cc_fraud %>%drop_na()%>%
  ggplot(aes(x = Day)) +
  geom_bar(fill = "chocolate",width = 0.3,color="chocolate") +
  theme_economist_white()

# converting the class variable as a factor
cc_fraud$Class <- factor(cc_fraud$Class)

# visualizing the data by Time_Group variable split by class
cc_fraud %>%drop_na()%>%
  ggplot(aes(x = Time_Group)) +
  geom_bar(color = "#238B45", fill = "#238B45") +
  theme_bw() +
  facet_wrap( ~ Class, scales = "free", ncol = 2)

# getting the summary of amount with respect to the class
print(tapply(cc_fraud$Amount  ,cc_fraud$Class,summary))

# converting R dataframe to H2O dataframe
cc_fraud_h2o <- as.h2o(cc_fraud)

#splitting the data into 60%, 20%, 20% chunks to use them as training, 
#vaidation and test datasets
splits <- h2o.splitFrame(cc_fraud_h2o, 
                         ratios = c(0.6, 0.2), 
                         seed = 148)   
# creating new train, validation and test h2o dataframes
train <- splits[[1]]
validation <- splits[[2]]
test <- splits[[3]]
# getting the target and features name in vectors
target <- "Class"
features <- setdiff(colnames(train), target)

# building autoencoder model
model_one = h2o.deeplearning(x = features, training_frame = train,
                             autoencoder = TRUE,
                             reproducible = TRUE,
                             seed = 148,
                             hidden = c(10,10,10), epochs = 100,activation = "Tanh",
                             validation_frame = test)

# saving the model so we do not have to retrain it again and agin
h2o.saveModel(model_one, path="model_one", force = TRUE)
# loading the model that is persisted on the disk 
model_one <- h2o.loadModel("/home/sunil/model_one/DeepLearning_model_R_1544970545051_1")
# printing the model to verify the autoencoder learning
print(model_one)
#making predictions on test dataset using the autoencoder model that is built
test_autoencoder <- h2o.predict(model_one, test)
train_features <- h2o.deepfeatures(model_one, train, layer = 2) %>%
  as.data.frame() %>%
  mutate(Class = as.vector(train[, 31]))
# printing the reduced data represented in layer2
print(train_features%>%head(3))
# plotting to verify if encoder has detected the fraud transactions
ggplot(train_features, aes(x = DF.L2.C1, y = DF.L2.C2, color = Class)) +
  geom_point(alpha = 0.1,size=1.5)+theme_bw()+
  scale_fill_brewer(palette = "Accent") 
ggplot(train_features, aes(x = DF.L2.C3, y = DF.L2.C4, color = Class)) +
  geom_point(alpha = 0.1,size=1.5)+theme_bw()+
  scale_fill_brewer(palette = "Accent")
# let's consider the third hidden layer. This is again a random choice
# in fact we could have taken any layer among the 10 inner layers
train_features <- h2o.deepfeatures(model_one, validation, layer = 3) %>%
  as.data.frame() %>%
  mutate(Class = as.factor(as.vector(validation[, 31]))) %>%
  as.h2o()
# getting the feature names from the sliced encoder layer
features_two <- setdiff(colnames(train_features), target)
model_two <- h2o.deeplearning(y = target,
                              x = features_two,
                              training_frame = train_features,
                              reproducible = TRUE, 
                              balance_classes = TRUE,
                              ignore_const_cols = FALSE,
                              seed = 148,
                              hidden = c(10, 5, 10), 
                              epochs = 100,
                              activation = "Tanh")
# saving the model to avoid retraining again
h2o.saveModel(model_two, path="model_two", force = TRUE)
# retrieving the model and printing it
model_two <- h2o.loadModel("/home/sunil/model_two/DeepLearning_model_R_1544970545051_2")
print(model_two)
test_3 <- h2o.deepfeatures(model_one, test, layer = 3)
# making predictions on the test data set with model_two
test_pred=h2o.predict(model_two, test_3,type="response")%>%
  as.data.frame() %>%
  mutate(actual = as.vector(test[, 31]))
# visualizing the predictions
test_pred%>%head()
# summarizing the predictions
print(h2o.predict(model_two, test_3) %>%
  as.data.frame() %>%
  dplyr::mutate(actual = as.vector(test[, 31])) %>%
  group_by(actual, predict) %>%
  dplyr::summarise(n = n()) %>%
  mutate(freq = n / sum(n)))
# shutting down the h2o cluster
h2o.shutdown()