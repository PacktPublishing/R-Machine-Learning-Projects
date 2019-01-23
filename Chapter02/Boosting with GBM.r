
library(doMC)
registerDoMC(cores=4)

setwd("~/Desktop/chapter 2")
mydata <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
mydata$EmployeeNumber=mydata$Over18=mydata$EmployeeCount=mydata$StandardHours = NULL

library(mlbench)
library(gbm)

mydata$Attrition = as.numeric(mydata$Attrition)
head(mydata)

mydata = transform(mydata, Attrition=Attrition-1)
head(mydata)

gbm.model = gbm(Attrition~., data=mydata, shrinkage=0.01, distribution = 'bernoulli', cv.folds=10, n.trees=3000, verbose=F)

best.iter = gbm.perf(gbm.model, method="cv")

best.iter

summary(gbm.model)

library(caret)

set.seed(123)
mydata1=mydata
mydata1$Attrition=as.factor(mydata1$Attrition)
fitControl = trainControl(method="repeatedcv", number=10,repeats=10)
caretmodel = train(Attrition~., data=mydata1, method="gbm",distribution="bernoulli", trControl=fitControl, verbose=F, tuneGrid=data.frame(.n.trees=best.iter, .shrinkage=0.01, .interaction.depth=1, .n.minobsinnode=1))
caretmodel




