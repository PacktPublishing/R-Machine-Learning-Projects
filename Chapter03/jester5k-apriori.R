library(recommenderlab)
data(Jester5k)
# binarizing the Jester ratings
Jester5k_bin <- binarize(Jester5k, minRating=1)
class(Jester5k_bin)
View(Jester5k_bin)
Jester5k_bin_mat <- as(Jester5k_bin,"matrix")
View(Jester5k_bin_mat)
Jester5k_bin_mat_num <- 1*Jester5k_bin_mat
View(Jester5k_bin_mat_num)
colnames(Jester5k_bin_mat_num)[colSums(is.na(Jester5k_bin_mat_num)) > 0]
rules <- apriori(data = Jester5k_bin_mat_num, parameter = list(supp = 0.5, conf = 0.8))
inspect(rules)
rulesdf <- as(rules, "data.frame") 
rulesdf[order(-rulesdf$lift, -rulesdf$confidence), ]