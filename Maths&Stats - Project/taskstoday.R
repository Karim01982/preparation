1) Coming up with a hypothesis --> Done
2) Back-testing with 2016 data
3) Looking to skewness and implications on our findings --> Done
4) Incorporating residual plots
5) Functionality for tables and results (e.g. stargazer; summary of descriptive statistics, etc)
6) Data-cleaning (e.g. changing from facebook likes to number of users; being clear on any NAs / blank cells)
7) Conditional descriptive statistics

Hypothesis testing

US vs Non-US - 
Between genres
Backtesting

Things outstanding

1) write-up of introduction
2) write up of conclusion
3) Completion of section 5
4) Scrub-down of dataset 




--------------

```{r results="asis"}
stargazer(cars, type="html", title="Table with Stargazer")
```