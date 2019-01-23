
library(doMC)
registerDoMC(cores=4)

setwd("~/Desktop/chapter 2")
mydata <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
mydata$EmployeeNumber=mydata$Over18=mydata$EmployeeCount=mydata$StandardHours = NULL

library(caret)
library(caretEnsemble)

control <- trainControl(method="repeatedcv", number=10, repeats=10, savePredictions=TRUE, classProbs=TRUE,index=createMultiFolds(mydata$Attrition, k=10, times=10))
algorithmList <- c('C5.0', 'nb', 'glm', 'knn', 'svmRadial')
set.seed(10000)
models <- caretList(Attrition~., data=mydata, trControl=control, methodList=algorithmList)
results <- resamples(models)
results

summary(results)

# correlation between results
modelCor(results)

stackControl <- trainControl(method="repeatedcv", number=10, repeats=10, savePredictions=TRUE, classProbs=TRUE)
# stack using glm
stack.glm <- caretStack(models, method="glm", trControl=stackControl)
print(stack.glm)

stack.rf <- caretStack(models, method="rf", trControl=stackControl)
print(stack.rf)


