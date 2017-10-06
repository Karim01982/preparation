##Regression analysis - Number of user votes

head(data11_15)

usereg1 <- lm(data = data11_15, num_voted_users~gross)
summary(usereg1)

usereg2 <- lm(data = data11_15, num_voted_users ~ gross +actor1_occurence+ actor2_occurence+ actor3_occurence)
summary(usereg2)

usereg3 <- lm(data = data11_15, num_voted_users ~ gross +actor1_occurence+ director_occurence)
summary(usereg3)

usereg4 <- lm(data = data11_15, num_voted_users ~ gross +actor1_occurence + movie_oscars + actor_oscars)
summary(usereg4)

usereg5 <- lm(data = data11_15, num_voted_users ~ gross +actor1_occurence + duration + actor_oscars)
summary(usereg5)

usereg6 <- lm(data = data11_15, num_voted_users ~ gross +actor1_occurence + duration + actor_oscars + actiondummy + adventuredummy + comedydummy + familydummy + fantasydummy+horrordummy+romancedummy + scifidummy+thrillerdummy)
summary(usereg6)

usereg7 <- lm(data = data11_15, num_voted_users ~ gross +actor1_occurence + duration + actor_oscars + actiondummy + familydummy +romancedummy + scifidummy + thrillerdummy)
summary(usereg7)

usereg8 <- lm(data = data11_15, num_voted_users ~ gross +actor1_occurence + duration + actor_oscars + familydummy + scifidummy )
summary(usereg8)

usereg9 <- lm(data = data11_15, num_voted_users ~ gross + duration + actor_oscars + familydummy + scifidummy)
summary(usereg9)

install.packages("car")

library(car)
residualPlot(usereg9)

##Regression analysis - Facebook likes

fbreg1 <- lm(data = data11_15, movie_facebook_likes~gross)
summary(fbreg1)

fbreg2 <- lm(data = data11_15, movie_facebook_likes ~ gross +actor1_occurence+ actor2_occurence+ actor3_occurence)
summary(fbreg2)

fbreg3 <- lm(data = data11_15, movie_facebook_likes ~ gross +actor1_occurence+ director_occurence)
summary(fbreg3)

fbreg4 <- lm(data = data11_15, movie_facebook_likes ~ gross +actor1_occurence + movie_oscars + actor_oscars)
summary(fbreg4)

fbreg5 <- lm(data = data11_15, movie_facebook_likes ~ gross + movie_oscars + duration + actor_oscars)
summary(fbreg5)

fbreg6 <- lm(data = data11_15, movie_facebook_likes ~ gross + movie_oscars + duration + actor_oscars + actiondummy + adventuredummy + comedydummy + familydummy + fantasydummy+horrordummy+romancedummy + scifidummy+thrillerdummy)
summary(fbreg6)

fbreg7 <- lm(data = data11_15, movie_facebook_likes ~ gross + movie_oscars + duration + actor_oscars + actiondummy + comedydummy + familydummy +horrordummy + romancedummy)
summary(fbreg7)

fbreg8 <- lm(data = data11_15, movie_facebook_likes ~ gross + movie_oscars + duration + comedydummy + familydummy +horrordummy + romancedummy )
summary(fbreg8)

fbreg9 <- lm(data = data11_15, movie_facebook_likes ~ gross + duration + actor_oscars + comedydummy + familydummy +horrordummy + romancedummy)
summary(fbreg9)

residualPlot(fbreg9)