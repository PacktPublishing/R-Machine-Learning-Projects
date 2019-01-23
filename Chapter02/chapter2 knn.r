
library(doMC)
registerDoMC(cores=4)
library(caret)

setwd("~/Desktop/chapter 2")
mydata <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
mydata$EmployeeNumber=mydata$Over18=mydata$EmployeeCount=mydata$StandardHours = NULL



set.seed(10000)
fitControl = trainControl(method="repeatedcv", number=10,repeats=10)
caretmodel = train(Attrition~., data=mydata, trControl=fitControl, method = "knn", tuneLength = 20)
caretmodel

set.seed(1000)
fitControl = trainControl(method="repeatedcv", number=10,repeats=10)
caretmodel1 = train(Attrition~., data=mydata, trControl=fitControl, method = "knn", tuneLength = 20)

caretmodel1

caretmodel1$finalModel

modelLookup(model = "knn")

getModelInfo(model = "knn")

library(kknn)

# save the model to disk
saveRDS(caretmodel1, "production_model.rds")

setwd("~/Desktop/chapter 2")



