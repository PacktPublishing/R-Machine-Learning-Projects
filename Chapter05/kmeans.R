# setting the working directory to a folder where dataset is located
setwd('/home/sunil/Desktop/chapter5/')
# reading the dataset to cust_data dataframe
cust_data = read.csv(file='Wholesale_customers_ data.csv', header = TRUE)
# removing the non-required columns
cust_data<-cust_data[,c(-1,-2)]
# including the NbClust library
library(NbClust)
# Computing the optimal number of clusters thru NbClust function with distance as euclidean and using kmeans 
NbClust(cust_data,distance="euclidean", method="kmeans")
# including the cluster library from where kmeans algorithm can be accessed
# computing the the intra cluster distance with Ks ranging from 2 to 10
library(purrr)
tot_withinss <- map_dbl(2:10, function(k){
  model <- kmeans(cust_data, centers = k, nstart = 50)
  model$tot.withinss
})
# converting the Ks and computed intra cluster distances to a dataframe
screeplot_df <- data.frame(k = 2:10,
                           tot_withinss = tot_withinss)
# plotting the elbow curve
library(ggplot2)
print( ggplot(screeplot_df, aes(x = k, y = tot_withinss)) + 
         geom_line() + 
         scale_x_continuous(breaks = 1:10) + 
         labs(x = "k", y = "Within Cluster Sum of Squares") + 
         ggtitle("Total Within Cluster Sum of Squares by # of Clusters (k)") +
         geom_point(data = screeplot_df[2,], aes(x = k, y = tot_withinss),
                    col = "red2", pch = 4, size = 7))
library(cluster)
# runing kmeans in cust_data dataset to obtain 3 clusters
kmout <- kmeans(cust_data, centers = 3, nstart = 50)
print(kmout)
library(factoextra)
print(fviz_cluster(kmout,data=cust_data))
# computing the silhouette index for the clusters
si <- silhouette(kmout$cluster, dist(cust_data, "euclidean"))
print(summary(si))