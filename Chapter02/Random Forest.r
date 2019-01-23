
library(doMC)
registerDoMC(cores=4)

setwd("~/Desktop/chapter 2")
mydata <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
mydata$EmployeeNumber=mydata$Over18=mydata$EmployeeCount=mydata$StandardHours = NULL

library(caret)

set.seed(10000)
fitControl = trainControl(method="repeatedcv", number=10,repeats=10)
caretmodel = train(Attrition~., data=mydata, method="rf", trControl=fitControl, verbose=F)
caretmodel






