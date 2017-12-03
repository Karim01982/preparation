#regressions
model1 <- lm(data=data11_15, gross ~ (num_voted_users))
summary(model1)

model2 <- lm(data=data11_15, gross ~ (num_voted_users + budget))
summary(model2)

model3 <- lm(data=data11_15, gross ~ (num_voted_users + budget + actiondummy + scifidummy + thrillerdummy))
summary(model3)

model4 <- lm(data=data11_15, gross ~ (num_voted_users + budget + actiondummy + title_year + actor_oscars))
summary(model4)

library(car)
residualPlot(model4)
