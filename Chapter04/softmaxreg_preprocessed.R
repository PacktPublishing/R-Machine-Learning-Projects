library(softmaxreg)
data(word2vec)
View(word2vec)
docVectors = function(x)
{
  wordEmbed(x, word2vec, meanVec = TRUE)
}
setwd('/home/sunil/Desktop/sentiment_analysis/')
text = read.csv(file='Sentiment Analysis Dataset.csv', header = TRUE)
text = text[,c("Sentiment", "SentimentText")]
temp=t(sapply(text$SentimentText, docVectors))

temp_train=temp[1:800,]
temp_test=temp[801:1000,]
labels_train=as.factor(as.character(text[1:800,]$Sentiment))
labels_test=as.factor(as.character(text[801:1000,]$Sentiment))

library(randomForest)
rf_senti_classifier=randomForest(temp_train, labels_train,ntree=20)

rf_predicts<-predict(rf_senti_classifier, temp_test)
library(rminer)
print(mmetric(rf_predicts, labels_test, c("ACC")))