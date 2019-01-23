library(recommenderlab)
data(Jester5k)
# getting to know the data struture
str(Jester5k)
## number of ratings
print(nratings(Jester5k))
## number of ratings per user
print(summary(rowCounts(Jester5k)))
## rating distribution
hist(getRatings(Jester5k), main="Distribution of ratings")
## 'best' joke with highest average rating
best <- which.max(colMeans(Jester5k))
cat(JesterJokes[best])
head(JesterJokes,2)
head(getRatingMatrix(Jester5k),5)
# split the data into the training and the test set
Jester5k_es <- evaluationScheme(Jester5k, method="split", train=0.8, given=20, goodRating=0)
print(Jester5k_es)
type = "IBCF"
#train ITCF cosine similarity models

# non-normalized
ITCF_N_C <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = NULL, method="Cosine"))

# centered
ITCF_C_C <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = "center",method="Cosine"))

# Z-score normalization
ITCF_Z_C <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = "Z-score",method="Cosine"))

#train ITCF Euclidean Distance models

# non-normalized
ITCF_N_E <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = NULL, method="Euclidean"))

# centered
ITCF_C_E <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = "center",method="Euclidean"))

# Z-score normalization
ITCF_Z_E <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = "Z-score",method="Euclidean"))

#train ITCF pearson correlation models

# non-normalized
ITCF_N_P <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = NULL, method="pearson"))

# centered
ITCF_C_P <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = "center",method="pearson"))

# Z-score normalization
ITCF_Z_P <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = "Z-score",method="pearson"))
# compute predicted ratings from each of the 9 models on the test dataset
pred1 <- predict(ITCF_N_C, getData(Jester5k_es, "known"), type="ratings")

pred2 <- predict(ITCF_C_C, getData(Jester5k_es, "known"), type="ratings")

pred3 <- predict(ITCF_Z_C, getData(Jester5k_es, "known"), type="ratings")

pred4 <- predict(ITCF_N_E, getData(Jester5k_es, "known"), type="ratings")

pred5 <- predict(ITCF_C_E, getData(Jester5k_es, "known"), type="ratings")

pred6 <- predict(ITCF_Z_E, getData(Jester5k_es, "known"), type="ratings")

pred7 <- predict(ITCF_N_P, getData(Jester5k_es, "known"), type="ratings")

pred8 <- predict(ITCF_C_P, getData(Jester5k_es, "known"), type="ratings")

pred9 <- predict(ITCF_Z_P, getData(Jester5k_es, "known"), type="ratings")

# set all predictions that fall outside the valid range to the boundary values
pred1@data@x[pred1@data@x[] < -10] <- -10
pred1@data@x[pred1@data@x[] > 10] <- 10

pred2@data@x[pred2@data@x[] < -10] <- -10
pred2@data@x[pred2@data@x[] > 10] <- 10

pred3@data@x[pred3@data@x[] < -10] <- -10
pred3@data@x[pred3@data@x[] > 10] <- 10

pred4@data@x[pred4@data@x[] < -10] <- -10
pred4@data@x[pred4@data@x[] > 10] <- 10

pred5@data@x[pred5@data@x[] < -10] <- -10
pred5@data@x[pred5@data@x[] > 10] <- 10

pred6@data@x[pred6@data@x[] < -10] <- -10
pred6@data@x[pred6@data@x[] > 10] <- 10

pred7@data@x[pred7@data@x[] < -10] <- -10
pred7@data@x[pred7@data@x[] > 10] <- 10

pred8@data@x[pred8@data@x[] < -10] <- -10
pred8@data@x[pred8@data@x[] > 10] <- 10

pred9@data@x[pred9@data@x[] < -10] <- -10
pred9@data@x[pred9@data@x[] > 10] <- 10

# aggregate the performance statistics
error_ITCF <- rbind(
  ITCF_N_C = calcPredictionAccuracy(pred1, getData(Jester5k_es, "unknown")),
  ITCF_C_C = calcPredictionAccuracy(pred2, getData(Jester5k_es, "unknown")),
  ITCF_Z_C = calcPredictionAccuracy(pred3, getData(Jester5k_es, "unknown")),
  ITCF_N_E = calcPredictionAccuracy(pred4, getData(Jester5k_es, "unknown")),
  ITCF_C_E = calcPredictionAccuracy(pred5, getData(Jester5k_es, "unknown")),
  ITCF_Z_E = calcPredictionAccuracy(pred6, getData(Jester5k_es, "unknown")),
  ITCF_N_P = calcPredictionAccuracy(pred7, getData(Jester5k_es, "unknown")),
  ITCF_C_P = calcPredictionAccuracy(pred8, getData(Jester5k_es, "unknown")),
  ITCF_Z_P = calcPredictionAccuracy(pred9, getData(Jester5k_es, "unknown"))
)
# printing the performance measurements
library(knitr)
print(kable(error_ITCF))
