
library(dplyr)
library(ggplot2)
library(ggthemes)

setwd("~/Desktop/chapter 2")

mydata <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

head(mydata)

str(mydata)

dim(mydata)

numeric_mydata <- mydata[,c(1,4,6,7,10,11,13,14,15,17,19,20,21,24,25,26,28:35)]
numeric_Attrition = as.numeric(mydata$Attrition)- 1
numeric_mydata = cbind(numeric_mydata, numeric_Attrition)
library(corrplot)
M <- cor(numeric_mydata)
corrplot(M, method="circle")

#Finding how many correlations are bigger than 0.70
k = 0
for(i in 1:25){
for(r in 1:25){
  if(M[i,r]> 0.70 & i != r){
    print(rownames(M)[i])
    print(colnames(M)[r])
    print(M[i,r])
    k= k + 1
  }
}  }
print(k/2)

options( warn = -1 )
### Overtime vs Attiriton
l <- ggplot(mydata, aes(OverTime,fill = Attrition))
l <- l + geom_histogram(stat="count")
print(l)
tapply(as.numeric(mydata$Attrition) - 1 ,mydata$OverTime,mean)

### MaritalStatus vs Attiriton
l <- ggplot(mydata, aes(MaritalStatus,fill = Attrition))
l <- l + geom_histogram(stat="count")
print(l)
tapply(as.numeric(mydata$Attrition) - 1 ,mydata$MaritalStatus,mean)

###JobRole vs Attrition
l <- ggplot(mydata, aes(JobRole,fill = Attrition))
l <- l + geom_histogram(stat="count")
print(l)
tapply(as.numeric(mydata$Attrition) - 1 ,mydata$JobRole,mean)
mean(as.numeric(mydata$Attrition) - 1)

###Gender vs Attrition
l <- ggplot(mydata, aes(Gender,fill = Attrition))
l <- l + geom_histogram(stat="count")
print(l)
tapply(as.numeric(mydata$Attrition) - 1 ,mydata$Gender,mean)

###EducationField vs Attrition
l <- ggplot(mydata, aes(EducationField,fill = Attrition))
l <- l + geom_histogram(stat="count")
print(l)
tapply(as.numeric(mydata$Attrition) - 1 ,mydata$EducationField,mean)

###Department vs Attrition
l <- ggplot(mydata, aes(Department,fill = Attrition))
l <- l + geom_histogram(stat="count")
print(l)
tapply(as.numeric(mydata$Attrition) - 1 ,mydata$Department,mean)

###BusinessTravel vs Attrition
l <- ggplot(mydata, aes(BusinessTravel,fill = Attrition))
l <- l + geom_histogram(stat="count")
print(l)
tapply(as.numeric(mydata$Attrition) - 1 ,mydata$BusinessTravel,mean)

### x=Overtime, y= Age, z = MaritalStatus , t = Attrition
ggplot(mydata, aes(OverTime, Age)) +  
  facet_grid(.~MaritalStatus) +
  geom_jitter(aes(color = Attrition),alpha = 0.4) +  
  ggtitle("x=Overtime, y= Age, z = MaritalStatus , t = Attrition") +  
  theme_light()

### MonthlyIncome vs. Age, by  color = Attrition
ggplot(mydata, aes(MonthlyIncome, Age, color = Attrition)) + 
  geom_jitter() +
  ggtitle("MonthlyIncome vs. Age, by  color = Attrition ") +
  theme_light()

if (!require("corrplot"))          install.packages("corrplot")
if (!require("ggplot2"))           install.packages("ggplot2")
if (!require("caret"))             install.packages("caret")
if (!require("rpart"))             install.packages("rpart")
if (!require("rpart.plot"))        install.packages("rpart.plot")
if (!require("e1071"))             install.packages("e1071")
if (!require("caTools"))           install.packages("caTools")
if (!require("rattle"))            install.packages("rattle")
if (!require("gridExtra"))         install.packages("gridExtra")
if (!require("ROCR"))              install.packages("ROCR")
if (!require("randomForest"))      install.packages("randomForest")
if (!require("randomForestSRC"))   install.packages("randomForestSRC")
if (!require("reshape2"))          install.packages("reshape2")
if (!require("RColorBrewer"))      install.packages("RColorBrewer")

sum(is.na(mydata))

library(caret)
#Setting the random seed for replication
set.seed(1234)

#setting up cross-validation
cvcontrol <- trainControl(method="repeatedcv", repeats=10, number = 10, allowParallel=TRUE)

mydata$EmployeeNumber=mydata$Over18=mydata$EmployeeCount=mydata$StandardHours = NULL

train.bagg <- train(Attrition ~ ., 
                   data=mydata,
                   method="treebag",B=10,
                   trControl=cvcontrol,
                   importance=TRUE)

train.bagg

library(party)

bagctrl <- bagControl(fit = ctreeBag$fit,
                      predict = ctreeBag$pred ,
                      aggregate = ctreeBag$aggregate)
ctrainbagg <- train(Attrition ~ ., 
                   data=mydata,
                   method="bag",
                   trControl=cvcontrol,bagControl = bagctrl,
                   allowParallel=TRUE)

ctrainbagg

ctrainbagg$finalModel

svm.predict <- function (object, x)
{
 if (is.character(lev(object))) {
    out <- predict(object, as.matrix(x), type = "probabilities")
    colnames(out) <- lev(object)
    rownames(out) <- NULL
  }
  else out <- predict(object, as.matrix(x))[, 1]
  out
}


bagctrl <- bagControl(fit = svmBag$fit,
                      predict = svm.predict ,
                      aggregate = svmBag$aggregate)

# fit the bagged svm model
set.seed(300)
svmbag <- train(Attrition ~ ., data = mydata, method="bag",
                trControl = cvcontrol, bagControl = bagctrl,allowParallel = TRUE)

svmbag

svm.predict <- function (object, x)
{
 if (is.character(lev(object))) {
    out <- predict(object, as.matrix(x), type = "probabilities")
    colnames(out) <- lev(object)
    rownames(out) <- NULL
  }
  else out <- predict(object, as.matrix(x))[, 1]
  out
}
    
ctrl <- trainControl(method = "cv", number = 10,
                     selectionFunction = "oneSE")


bagctrl <- bagControl(fit = svmBag$fit,
                      predict = svm.predict ,
                      aggregate = svmBag$aggregate)

# fit the bagged svm model
set.seed(300)
svmbag <- train(Attrition ~ ., data = mydata, method="bag",
                trControl = cvcontrol, bagControl = bagctrl)

svmbag

library(kernlab)

str(nbBag)

nb.predict <- function (object, x)
{
 if (is.character(lev(object))) {
    out <- predict(object, as.matrix(x), type = "probabilities")
    colnames(out) <- lev(object)
    rownames(out) <- NULL
  }
  else out <- predict(object, as.matrix(x))[, 1]
  out
}

bagctrl <- bagControl(fit = nbBag$fit,
                      predict = nbBag$pred ,
                      aggregate = nbBag$aggregate)

# fit the bagged nb model
set.seed(300)
nbbag <- train(Attrition ~ ., data = mydata, method="bag",
                trControl = cvcontrol, bagControl = bagctrl)

nbbag

library(klaR)

library(doMC) 
registerDoMC(cores = 4) 

str(nnetBag)

bagctrl <- bagControl(fit = nbBag$fit,
                      predict = nbBag$pred ,
                      aggregate = nbBag$aggregate)


# fit the bagged nb model
set.seed(300)
nbbag1 <- train(Attrition ~ ., data = mydata, method="bag",B=100,
                trControl = cvcontrol, bagControl = bagctrl)

nbbag1




