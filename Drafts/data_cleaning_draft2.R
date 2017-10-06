

##Initial cleaning
library(dplyr)
library(readr)
library(tidyr)
library(magrittr)
library(stringr)
library(tidyr)

old_data <- read.csv("./Drafts/movie_metadata.csv", stringsAsFactors = FALSE)
filter_data <- filter(old_data, country=="USA", language=="English")
select_data <- select(filter_data, movie_title, title_year, director_name, duration, gross, genres, num_voted_users, num_user_for_reviews, budget, imdb_score, movie_facebook_likes, actor_1_name, actor_2_name, actor_3_name)
select_data_filter <- filter(select_data, title_year >= 2011, gross !=0, gross != "NA", gross != "", imdb_score != 0, imdb_score != "NA", imdb_score != "", budget != 0, budget != "NA", budget != "", movie_facebook_likes != 0, movie_facebook_likes != "NA", movie_facebook_likes != "")

head(select_data_filter)

##Genre conversion

typeof(select_data_filter$genres)
toString(select_data_filter$genres)
str(select_data_filter)

genre_names <- c("Action","Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "History", "Horror", "Musical", "Mystery", "News", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western")
genre_types <- c("genre1", "genre2","genre3", "genre4", "genre5", "genre6", "genre7")
genre_filter <- select_data_filter %>% separate(genres, into=genre_types,sep = "\\|")

##Establishing a Gross Revenue / Budget variable and cleaning up movie title names
RB_data <- mutate(.data = genre_filter, return = gross / budget)
RB_data$return <- round(RB_data$return, 2)
#head(RB_data)

##Clean RB_data of duplicates and the additional character in name
RB_data$movie_title = substr(RB_data$movie_title,1,nchar(RB_data$movie_title)-1)
RB_data <- RB_data[-c(as.vector(which(duplicated(RB_data) == TRUE))),]

#head(RB_data)

##Introducing flags for genres

RB_data$actiondummy <- as.numeric( RB_data$genre1 == "Action" | RB_data$genre2 == "Action" | RB_data$genre3 == "Action" | RB_data$genre4 == "Action" | RB_data$genre5 == "Action" | RB_data$genre6 == "Action" | RB_data$genre7 == "Action")
RB_data$actiondummy[is.na(RB_data$actiondummy)] <- 0

RB_data$adventuredummy <- as.numeric( RB_data$genre1 == "Adventure" | RB_data$genre2 == "Adventure" | RB_data$genre3 == "Adventure" | RB_data$genre4 == "Adventure" | RB_data$genre5 == "Adventure" | RB_data$genre6 == "Adventure" | RB_data$genre7 == "Adventure")
RB_data$adventuredummy[is.na(RB_data$adventuredummy)] <- 0

RB_data$animationdummy <- as.numeric( RB_data$genre1 == "Animation" | RB_data$genre2 == "Animation" | RB_data$genre3 == "Animation" | RB_data$genre4 == "Animation" | RB_data$genre5 == "Animation" | RB_data$genre6 == "Animation" | RB_data$genre7 == "Animation")
RB_data$animationdummy[is.na(RB_data$animationdummy)] <- 0

RB_data$biographydummy <- as.numeric( RB_data$genre1 == "Biography" | RB_data$genre2 == "Biography" | RB_data$genre3 == "Biography" | RB_data$genre4 == "Biography" | RB_data$genre5 == "Biography" | RB_data$genre6 == "Biography" | RB_data$genre7 == "Biography")
RB_data$biographydummy[is.na(RB_data$biographydummy)] <- 0

RB_data$comedydummy <- as.numeric( RB_data$genre1 == "Comedy" | RB_data$genre2 == "Comedy" | RB_data$genre3 == "Comedy" | RB_data$genre4 == "Comedy" | RB_data$genre5 == "Comedy" | RB_data$genre6 == "Comedy" | RB_data$genre7 == "Comedy")
RB_data$comedydummy[is.na(RB_data$comedydummy)] <- 0

RB_data$crimedummy <- as.numeric( RB_data$genre1 == "Crime" | RB_data$genre2 == "Crime" | RB_data$genre3 == "Crime" | RB_data$genre4 == "Crime" | RB_data$genre5 == "Crime" | RB_data$genre6 == "Crime" | RB_data$genre7 == "Crime")
RB_data$crimedummy[is.na(RB_data$crimedummy)] <- 0

RB_data$documentarydummy <- as.numeric( RB_data$genre1 == "Documentary" | RB_data$genre2 == "Documentary" | RB_data$genre3 == "Documentary" | RB_data$genre4 == "Documentary" | RB_data$genre5 == "Documentary" | RB_data$genre6 == "Documentary" | RB_data$genre7 == "Documentary")
RB_data$documentarydummy[is.na(RB_data$documentarydummy)] <- 0

RB_data$dramadummy <- as.numeric( RB_data$genre1 == "Drama" | RB_data$genre2 == "Drama" | RB_data$genre3 == "Drama" | RB_data$genre4 == "Drama" | RB_data$genre5 == "Drama" | RB_data$genre6 == "Drama" | RB_data$genre7 == "Drama")
RB_data$dramadummy[is.na(RB_data$dramadummy)] <- 0

RB_data$familydummy <- as.numeric( RB_data$genre1 == "Family" | RB_data$genre2 == "Family" | RB_data$genre3 == "Family" | RB_data$genre4 == "Family" | RB_data$genre5 == "Family" | RB_data$genre6 == "Family" | RB_data$genre7 == "Family")
RB_data$familydummy[is.na(RB_data$familydummy)] <- 0

RB_data$fantasydummy <- as.numeric( RB_data$genre1 == "Fantasy" | RB_data$genre2 == "Fantasy" | RB_data$genre3 == "Fantasy" | RB_data$genre4 == "Fantasy" | RB_data$genre5 == "Fantasy" | RB_data$genre6 == "Fantasy" | RB_data$genre7 == "Fantasy")
RB_data$fantasydummy[is.na(RB_data$fantasydummy)] <- 0

RB_data$historydummy <- as.numeric( RB_data$genre1 == "History" | RB_data$genre2 == "History" | RB_data$genre3 == "History" | RB_data$genre4 == "History" | RB_data$genre5 == "History" | RB_data$genre6 == "History" | RB_data$genre7 == "History")
RB_data$historydummy[is.na(RB_data$historydummy)] <- 0

RB_data$horrordummy <- as.numeric( RB_data$genre1 == "Horror" | RB_data$genre2 == "Horror" | RB_data$genre3 == "Horror" | RB_data$genre4 == "Horror" | RB_data$genre5 == "Horror" | RB_data$genre6 == "Horror" | RB_data$genre7 == "Horror")
RB_data$horrordummy[is.na(RB_data$horrordummy)] <- 0

RB_data$musicaldummy <- as.numeric( RB_data$genre1 == "Musical" | RB_data$genre2 == "Musical" | RB_data$genre3 == "Musical" | RB_data$genre4 == "Musical" | RB_data$genre5 == "Musical" | RB_data$genre6 == "Musical" | RB_data$genre7 == "Musical")
RB_data$musicaldummy[is.na(RB_data$musicaldummy)] <- 0

RB_data$mysterydummy <- as.numeric( RB_data$genre1 == "Mystery" | RB_data$genre2 == "Mystery" | RB_data$genre3 == "Mystery" | RB_data$genre4 == "Mystery" | RB_data$genre5 == "Mystery" | RB_data$genre6 == "Mystery" | RB_data$genre7 == "Mystery")
RB_data$mysterydummy[is.na(RB_data$mysterydummy)] <- 0

RB_data$newsdummy <- as.numeric( RB_data$genre1 == "News" | RB_data$genre2 == "News" | RB_data$genre3 == "News" | RB_data$genre4 == "News" | RB_data$genre5 == "News" | RB_data$genre6 == "News" | RB_data$genre7 == "News")
RB_data$newsdummy[is.na(RB_data$newsdummy)] <- 0

RB_data$romancedummy <- as.numeric( RB_data$genre1 == "Romance" | RB_data$genre2 == "Romance" | RB_data$genre3 == "Romance" | RB_data$genre4 == "Romance" | RB_data$genre5 == "Romance" | RB_data$genre6 == "Romance" | RB_data$genre7 == "Romance")
RB_data$romancedummy[is.na(RB_data$romancedummy)] <- 0

RB_data$scifidummy <- as.numeric( RB_data$genre1 == "Sci-Fi" | RB_data$genre2 == "Sci-Fi" | RB_data$genre3 == "Sci-Fi" | RB_data$genre4 == "Sci-Fi" | RB_data$genre5 == "Sci-Fi" | RB_data$genre6 == "Sci-Fi" | RB_data$genre7 == "Sci-Fi")
RB_data$scifidummy[is.na(RB_data$scifidummy)] <- 0

RB_data$sportdummy <- as.numeric( RB_data$genre1 == "Sport" | RB_data$genre2 == "Sport" | RB_data$genre3 == "Sport" | RB_data$genre4 == "Sport" | RB_data$genre5 == "Sport" | RB_data$genre6 == "Sport" | RB_data$genre7 == "Sport")
RB_data$sportdummy[is.na(RB_data$sportdummy)] <- 0

RB_data$thrillerdummy <- as.numeric( RB_data$genre1 == "Thriller" | RB_data$genre2 == "Thriller" | RB_data$genre3 == "Thriller" | RB_data$genre4 == "Thriller" | RB_data$genre5 == "Thriller" | RB_data$genre6 == "Thriller" | RB_data$genre7 == "Thriller")
RB_data$thrillerdummy[is.na(RB_data$thrillerdummy)] <- 0

RB_data$wardummy <- as.numeric( RB_data$genre1 == "War" | RB_data$genre2 == "War" | RB_data$genre3 == "War" | RB_data$genre4 == "War" | RB_data$genre5 == "War" | RB_data$genre6 == "War" | RB_data$genre7 == "War")
RB_data$wardummy[is.na(RB_data$wardummy)] <- 0

RB_data$westerndummy <- as.numeric( RB_data$genre1 == "Western" | RB_data$genre2 == "Western" | RB_data$genre3 == "Western" | RB_data$genre4 == "Western" | RB_data$genre5 == "Western" | RB_data$genre6 == "Western" | RB_data$genre7 == "Western")
RB_data$westerndummy[is.na(RB_data$westerndummy)] <- 0

#sum(RB_data$actiondummy)

##To search for items within the Movie Title column
row <- grep(RB_data$movie_title,"The Dark Knight ")
as.character(RB_data$movie_title[1]) == as.character("The Dark Knight Rises ")
head(RB_data)
typeof(RB_data$genre1)
typeof(RB_data$movie_title)
RB_data$movie_title[1] == as.character("The Dark Knight Rises")


#to search for a string in any column
str_detect(RB_data$movie_title,"John Carter")

##cleaning awards data 
awards_data <- read.csv("./Drafts/awards_metadata.csv", stringsAsFactors = FALSE, encoding = "UTF-8")
awards_data <- filter(awards_data, Winner!="NA")
awards_data <- select(awards_data, Year, Award, Name, Film)
movie_wins <- filter(awards_data, Award=="Best Picture" | Award=="Best Motion Picture")
movie_wins <- count(movie_wins, Name)
actor_wins <- filter(awards_data, Award=="Actor" | Award=="Actress" | Award=="Actor in a Leading Role" | Award=="Actress in a Leading Role" | Award=="Actor in a Supporting Role" | Award=="Actress in a Supporting Role")
actor_wins <- count(actor_wins, Name)
director_wins <- filter(awards_data, Award=="Directing")
director_wins$Name = director_wins$Film
director_wins <- count(director_wins, Name)
award_wins <- rbind(actor_wins,director_wins,movie_wins)
head(data11_15)


##Getting the occurences of directors/actors in the database
for (i in (1:length(RB_data))){
  RB_data$director_occurence[i] = length(which((str_detect(RB_data$director_name, RB_data$director_name[i])) == TRUE))
  RB_data$actor1_occurence[i] = length(which((str_detect(cbind(RB_data$actor_1_name, RB_data$actor_2_name, RB_data$actor_3_name), RB_data$actor_1_name[i])) == TRUE))
  RB_data$actor2_occurence[i] = length(which((str_detect(cbind(RB_data$actor_1_name, RB_data$actor_2_name, RB_data$actor_3_name), RB_data$actor_2_name[i])) == TRUE))
  RB_data$actor3_occurence[i] = length(which((str_detect(cbind(RB_data$actor_1_name, RB_data$actor_2_name, RB_data$actor_3_name), RB_data$actor_3_name[i])) == TRUE))
}


#FOR GETTING AWARDS DATA
occurences = table(unlist(award_wins$Name))
occurences = as.data.frame(occurences)
RB_data = left_join(x=RB_data, y=occurences, by=c(director_name = "Var1"))
colnames(RB_data)[47] <- "dir_oscars"
RB_data = left_join(x=RB_data, y=occurences, by=c(actor_1_name = "Var1"))
colnames(RB_data)[48] <- "act1_oscars"
RB_data = left_join(x=RB_data, y=occurences, by=c(actor_2_name = "Var1"))
colnames(RB_data)[49] <- "act2_oscars"
RB_data = left_join(x=RB_data, y=occurences, by=c(actor_3_name = "Var1"))
colnames(RB_data)[50] <- "act3_oscars"
RB_data = left_join(x=RB_data, y=occurences, by=c(director_name = "Var1"))
colnames(RB_data)[51] <- "movie_oscars"
RB_data[c("dir_oscars","act1_oscars","act2_oscars","act3_oscars","movie_oscars")][is.na(RB_data[c("dir_oscars","act1_oscars","act2_oscars","act3_oscars","movie_oscars")])] <- 0
RB_data$actor_oscars = RB_data$act1_oscars + RB_data$act2_oscars + RB_data$act3_oscars
RB_data = RB_data[ , !(names(RB_data) %in% c("act1_oscars","act2_oscars","act3_oscars","genre1","genre2","genre3","genre4","genre5","genre6","genre7"))]


##Establishing separate datasets for time periods
data11_15 <- filter(.data = RB_data, title_year < 2016)
#head(data11_15)
data16 <- filter(.data = RB_data,title_year == 2016)
#head(data16)
#print(data11_15$movie_facebook_likes)
#print(data11_15$movie_facebook_likes)
