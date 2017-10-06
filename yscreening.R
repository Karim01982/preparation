#Graphs

library(ggplot2)
library(dplyr)

##Density plots

ggplot(data=data11_15)+geom_density(aes(x=imdb_score))

ggplot(data=data11_15)+geom_density(aes(x=movie_facebook_likes))

ggplot(data=data11_15)+geom_density(aes(x=gross))

ggplot(data=data11_15)+geom_density(aes(x=return))

ggplot(data=data11_15)+geom_density(aes(x=budget))

##Boxplots

ggplot(data=data11_15)+geom_boxplot(aes(x=1, y=imdb_score))

ggplot(data=data11_15)+geom_boxplot(aes(x=1, y=movie_facebook_likes)) + ylim(0,100000)

ggplot(data=data11_15)+geom_boxplot(aes(x=1, y=gross))

ggplot(data=data11_15)+geom_boxplot(aes(x=1, y=return)) + ylim(0,5)

ggplot(data=data11_15)+geom_boxplot(aes(x=1, y=budget))

##Correlation table

cordata <- select(data11_15, imdb_score, movie_facebook_likes, gross, return, budget)
cor_table <- round(cor(cordata),2)
print(cor_table)

##Scatterplot data

ggplot(data=data11_15) + geom_jitter(aes(x=return <1, y=movie_facebook_likes)) + xlab("How do facebook likes compare for films that have made <1.0x money?")
returns_higher1 <- data11_15[data11_15[,"return"] >1, "return"]
returns_fb1 <- data11_15[data11_15[,"return"] >1, "movie_facebook_likes"]
returns_imdb1 <- data11_15[data11_15[,"return"] >1, "imdb_score"]
return_data <- data.frame(returns_higher1, returns_fb1, returns_imdb1)

ggplot(data=return_data) + geom_jitter(aes(x=returns_higher1, y=returns_imdb1)) + xlab("How do facebook likes compare for films that have made <1.0x money?") +xlim(1,10)

cor.test(returns_higher1, returns_imdb1)
