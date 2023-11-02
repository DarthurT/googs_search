library(data.table)
library(funModeling)
library(plyr)
library(dplyr)
library(ggplot2)
library(reshape2)
library(imputeTS)
library(tsoutliers)
library(lubridate)
library(zoo)
library(xts)
library(forecast)
library(timeSeries)
library(stringi)
library(tidyverse)
library(tibbletime)
library(anomalize)
library(tcltk)
library(readxl)
library(scales)
library(grid)
library(tidyverse)
library(gganimate)
library(gifski)
library(ggthemes)
library(ggrepel)
library(hrbrthemes)
library(gghighlight)
        

data <- multiTimeline


data$date <- as.Date(paste(data$V1,"-01",sep=""))
data$month <- month(data$date)
data$year <- year(data$date)
data$rollmean <- rollmean(data$V2, 3, na.pad = TRUE)



data$month.text <- with(data, month.abb[month])

data2 <- data


ggplot(data = data,aes(x = month, y = V2)) + 
  geom_line(data = data[data$year == 2020,],aes(x = month, y = V2, colour = as.factor(year)),stat="smooth",method = "loess", alpha = 0.8) +
  geom_line(data = data[data$year != 2020,],aes(x = month, y = V2, colour = as.factor(year)),stat="smooth",method = "loess", alpha = 0.25) +
  gghighlight(year == 2020) +
  geom_boxplot(data = data,aes(x = as.factor(month), y = V2), fill =  "#d3d3d3", colour = "#F8F8F8", alpha = 0.5) +
  labs(x = "",
       y = "Interest",
       title = "Year-on-year google search trends for 'jobs'",
       colour = "Year",
       caption = "Data from @Google Trends (UK only)") +
  scale_x_discrete(breaks=c(1, 2, 3, 4 , 5, 6, 7, 8, 9, 10, 11, 12),
                   labels=c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")) +
  ylim(40, 100) +
  theme_ft_rc() 

h <- h + gghighlight(data = data2, year == 2020) 
h