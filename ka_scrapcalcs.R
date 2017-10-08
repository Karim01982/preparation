install.packages("Ecdat")
library("Ecdat")
data("Cigarette")
tail(Cigarette)

library(dplyr)
library(magrittr)

cig_by_state <- Cigarette %>%
  group_by(state) %>%
  summarise(state_mean = mean(avgprs),
  n=n()) 
  
cig_by_year <- Cigarette %>%
  group_by(year) %>%
  summarise(annual_mean = mean(avgprs))

tail(cig_by_state)