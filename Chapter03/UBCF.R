library(recommenderlab)
data(Jester5k)
# split the data into the training and the test set
Jester5k_es <- evaluationScheme(Jester5k, method="split", train=0.8, given=20, goodRating=0)
print(Jester5k_es)
type = "UBCF"
#train UBCF cosine similarity models

# non-normalized
UBCF_N_C <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = NULL, method="Cosine"))

# centered
UBCF_C_C <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = "center",method="Cosine"))

# Z-score normalization
UBCF_Z_C <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = "Z-score",method="Cosine"))

#train UBCF Euclidean Distance models

# non-normalized
UBCF_N_E <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = NULL, method="Euclidean"))

# centered
UBCF_C_E <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = "center",method="Euclidean"))

# Z-score normalization
UBCF_Z_E <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = "Z-score",method="Euclidean"))

#train UBCF pearson correlation models

# non-normalized
UBCF_N_P <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = NULL, method="pearson"))

# centered
UBCF_C_P <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = "center",method="pearson"))

# Z-score normalization
UBCF_Z_P <- Recommender(getData(Jester5k_es, "train"), type, 
                        param=list(normalize = "Z-score",method="pearson"))
# compute predicted ratings from each of the 9 models on the test dataset
pred1 <- predict(UBCF_N_C, getData(Jester5k_es, "known"), type="ratings")

pred2 <- predict(UBCF_C_C, getData(Jester5k_es, "known"), type="ratings")

pred3 <- predict(UBCF_Z_C, getData(Jester5k_es, "known"), type="ratings")

pred4 <- predict(UBCF_N_E, getData(Jester5k_es, "known"), type="ratings")

pred5 <- predict(UBCF_C_E, getData(Jester5k_es, "known"), type="ratings")

pred6 <- predict(UBCF_Z_E, getData(Jester5k_es, "known"), type="ratings")

pred7 <- predict(UBCF_N_P, getData(Jester5k_es, "known"), type="ratings")

pred8 <- predict(UBCF_C_P, getData(Jester5k_es, "known"), type="ratings")

pred9 <- predict(UBCF_Z_P, getData(Jester5k_es, "known"), type="ratings")

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
error_UBCF <- rbind(
  UBCF_N_C = calcPredictionAccuracy(pred1, getData(Jester5k_es, "unknown")),
  UBCF_C_C = calcPredictionAccuracy(pred2, getData(Jester5k_es, "unknown")),
  UBCF_Z_C = calcPredictionAccuracy(pred3, getData(Jester5k_es, "unknown")),
  UBCF_N_E = calcPredictionAccuracy(pred4, getData(Jester5k_es, "unknown")),
  UBCF_C_E = calcPredictionAccuracy(pred5, getData(Jester5k_es, "unknown")),
  UBCF_Z_E = calcPredictionAccuracy(pred6, getData(Jester5k_es, "unknown")),
  UBCF_N_P = calcPredictionAccuracy(pred7, getData(Jester5k_es, "unknown")),
  UBCF_C_P = calcPredictionAccuracy(pred8, getData(Jester5k_es, "unknown")),
  UBCF_Z_P = calcPredictionAccuracy(pred9, getData(Jester5k_es, "unknown"))
)
# printing the performance measurements
library(knitr)
print(kable(error_UBCF))