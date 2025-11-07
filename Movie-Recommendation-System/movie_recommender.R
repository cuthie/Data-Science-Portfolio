# Reading Data

movies <- read.csv("movies.csv", header = T, sep = ",")
ratings <- read.csv("ratings.csv", header = T, sep = ",")
links <- read.csv("links.csv", header = T, sep = ",")
links_stars <- read.csv("links_stars.csv", header = T, sep = ",")

# Packages

library(tidyr)
library(stringr)
library(RColorBrewer)
library(wordcloud)
library(tm)
library(data.table)
library(reshape2)



### Making movieIDs in dataframes movies and ratings have the same length.

a <- sort(unique(ratings$movieId))
b <- sort(unique(movies$movieId))
if (length(a) > length(b)){
  ind.diff <- setdiff(a,b)
} else {
  ind.diff <- setdiff(b,a)
  ind.diff.original <- which(movies$movieId %in% ind.diff)
  movies <- movies[-ind.diff.original,]
}



#ratings = ratings[1:200,] # A portion of rating

ratings_spread = as.matrix(spread(ratings, key = movieId, value = rating)) 
ratings_spread[,-1] = ratings_spread[,-1] - rowMeans(ratings_spread[,-1], na.rm = TRUE) # create user-item matrix (and deduct average ratings of users)
ratings_spread[is.na(ratings_spread)] = 0 # Set values corresponding to those movies that a user has not yet seen (rated) to zero
users = ratings_spread[,1]
ratings_spread = ratings_spread[,2:ncol(ratings_spread)] # remove user_id column


## Extracting geners --  Item (Genre) profile (movie~genre)

genres <- as.data.frame(movies$genres, stringsAsFactors=FALSE)
genres[genres == "(no genres listed)"] = NA

genres_separate <- as.data.frame(tstrsplit(genres[,1], '[|]', type.convert=TRUE), stringsAsFactors=FALSE)
colnames(genres_separate) <- c(1:10)

decade_list <- c("1900s","1910s","1920s","1930s","1940s","1950s","1960s","1970s","1980s","1990s","2000s","2010s")
genre_list <- c("Action", "Adventure", "Animation", "Children", "Comedy", "Crime","Documentary", "Drama", "Fantasy","Film-Noir", "Horror", "Musical", "Mystery","Romance","Sci-Fi", "Thriller", "War", "Western","IMAX")
genre_decade_list <- c(genre_list, decade_list)


## Extracting decades based on year of production

toMatch = as.character(1900:2016)
toMatch.year <- grep(paste(toMatch,collapse="|"), movies$title, value=TRUE)

year <- str_sub(movies$title,-5,-2)
year[which(grepl(paste(toMatch,collapse="|"), year) == F)] = NA


# cut years into decades

decade = cut(as.numeric(year), breaks = seq(1900,2020,10), labels = c("1900s","1910s","1920s","1930s","1940s","1950s","1960s","1970s","1980s","1990s","2000s","2010s"))

genre_decade_matrix <- matrix(0,nrow(genres_separate),length(genre_decade_list)) 
for (r in 1:nrow(genre_decade_matrix)){
  ind.genre <- which(genre_list %in% genres_separate[r,])
  ind.decade <- which(genre_decade_list %in% decade[r])
  genre_decade_matrix[r,c(ind.genre,ind.decade)] <- 1
}
colnames(genre_decade_matrix) <- genre_decade_list #set column names to genre list
genre_decade_matrix <- as.data.frame(genre_decade_matrix, stringsAsFactors=FALSE)


## Word-cloud analysis 

genre_matrix = genre_decade_matrix[,1:19]
decade_matrix = genre_decade_matrix[,20:31]

freq_genre = colSums(genre_matrix)/sum(genre_matrix)
freq_decade = colSums(decade_matrix)/sum(decade_matrix)

wordcloud(genre_list, freq_genre*1000, colors = brewer.pal(5, "Dark2") ,  max.words = 1000, random.order = FALSE)
wordcloud(decade_list, freq_decade*100000, colors = brewer.pal(6, "Dark2") ,  max.words = 100000, random.order = T)



no_stars = 4; # Number of stars 
links_stars = read.csv("links_stars.csv", header = T, sep = ",")
links_stars = links_stars[movies$movieId,] # Trim links_stars s.t. it corresponds to the moviesID in the movies data frame

links_stars[,8:11] = NULL 

# Stack all stars into stars list 

stars = NULL 
for (i in 4: (4 + no_stars-1)){
  stars = c(stars, as.character(links_stars[,i]))
}
stars = unique(stars)


star_matrix <- matrix(0,nrow(links_stars), length(stars)) 
tmp = as.matrix(links_stars[,4:7])
for (r in 1:nrow(star_matrix)){
  ind.stars <- which(stars %in% tmp[r,])
  star_matrix[r,ind.stars] <- 1
}

star_matrix <- as.data.frame(star_matrix, stringsAsFactors=FALSE)
names(star_matrix) = stars



# Limit the star list to the first top ones 
stars_list = sort(colSums(star_matrix), decreasing = T)[1:51] 
stars_list = stars_list[-which(is.na(names(stars_list)))] # remove NAs in the star list (This is due to the fact that IMDb )

# Use the data only for the top 50 actors 
star_matrix = star_matrix[,names(stars_list)]

# Wordcloud analysis of stars
freq_stars = colSums(star_matrix)/sum(star_matrix)
wordcloud(names(stars_list), freq_stars*10000, colors = brewer.pal(5, "Dark2") ,  max.words = 50, random.order = T)

genre_decade_star_matrix = cbind(genre_decade_matrix, star_matrix)


# Define a function that determines cosine similarity (in cosine or correlation sense) between two arrays 

cosine.sim <- function(x, y) {
  sim = sum(x*y)/(sqrt(sum(x^2)*sum(y^2)))
  return(sim)
}

# Item-based recommender system (Collaborative filtering)

# Determine item similarity matrix

sim_matrix <- matrix(0,ncol(ratings_spread),ncol(ratings_spread)) 
for (i in 1:ncol(ratings_spread)){
  for (j in i:ncol(ratings_spread)) {
     sim_matrix[i,j] = cosine.sim(ratings_spread[,i],ratings_spread[,j])
  }
}
# Copy the upper-triangular part of the matrix to its lower-triangular counterpart

sim_matrix = sim_matrix + t(sim_matrix)
diag(sim_matrix) = 1

# Recommandations for user 1 

user = 1
predicted_ratings = ratings_spread
ind_rated_movies = which(ratings_spread[user,] != 0)
ind_non_rated_movies = which(ratings_spread[user,] == 0)
for (i in 1:ncol(ratings_spread)){
  if (ratings_spread[user,i] == 0){
    predicted_ratings[user,i] = sum(ratings_spread[user,ind_rated_movies]*sim_matrix[i,ind_rated_movies])/sum(sim_matrix[i,ind_rated_movies])
  }
}
ind_rec1 = sort(predicted_ratings[user,ind_non_rated_movies],decreasing = T)[1:10]
rec1 = movies[movies$movieId %in% as.integer(names(ind_rec1)),]



# User-based recommender system (Collaborative filtering)

# Determine user similarity matrix
sim_matrix_user <- matrix(0,nrow(ratings_spread),nrow(ratings_spread))
for (i in 1:nrow(ratings_spread)){
  for (j in i:nrow(ratings_spread)) {
    sim_matrix_user[i,j] = cosine.sim(ratings_spread[i,],ratings_spread[j,])
  }
}
# Copy the upper-triangular part of the matrix to its lower-triangular counterpart

sim_matrix_user = sim_matrix_user + t(sim_matrix_user)
diag(sim_matrix_user) = 1

# Recommandations for user 1 

user = 1
predicted_ratings_user = ratings_spread
ind_rated_movies = which(ratings_spread[user,] != 0)
ind_non_rated_movies = which(ratings_spread[user,] == 0)
for (i in 1:ncol(ratings_spread)){
  if (ratings_spread[user,i] == 0){
    predicted_ratings_user[user,i] = sim_matrix_user[user,-user] %*% ratings_spread[-user,i]/sum(sim_matrix_user[user,-user])
  }
}

ind_rec2 = sort(predicted_ratings_user[user,ind_non_rated_movies],decreasing = T)[1:10]
rec2 = movies[movies$movieId %in% as.integer(names(ind_rec2)),]



