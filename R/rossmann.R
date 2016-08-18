require(readr)
require(forecast)
require(stats)
require(dplyr)
require(ggplot2)
require(scales)
require(Metrics)

setwd("C:/Users/jennhaur/Desktop/Working/ML/rossmann")
getwd()
df <- read_csv("train.csv",col_names=TRUE,col_types=cols(Store="i",DayOfWeek="i",Date=col_date("%Y-%m-%d"),Sales="i",Customers=col_skip(),Open="i",Promo="i",StateHoliday=col_factor(c(0,"a","b","c")),SchoolHoliday="i"))
df <- read_csv("train.csv",col_names=TRUE,col_types=cols(Store="i",DayOfWeek="i",Date=col_date("%Y-%m-%d"),Sales="i",Customers=col_skip(),Open="i",Promo="i",StateHoliday="c",SchoolHoliday="i"))
df$StateHoliday = as.factor(df$StateHoliday)

#Analysis
store1 <- subset(df,subset=c(Store==1))
store1d <- filter(df,Store==1)

store1$year <- apply(store1[c("Date")],1,function(x) year(x[1]))
store1$day <- apply(store1[c("Date")],1,function(x) day(x[[1]]))
store1$week <- apply(store1[c("Date")],1,function(x) as.integer(week(x[[1]])))
store1$month <- apply(store1[c("Date")],1,function(x) as.integer(month(x[[1]])))

http://neondataskills.org/R/time-series-plot-ggplot/
qplot(x=store1$Date,y=store1$Sale,data=store1,xlab="Date",ylab="Sales")
ggplot(store1,aes(store1$Date,store1$Sale)) + 
geom_line() + 
ggtitle("Sales for Store1")+
xlab("Date")+
(scale_x_date(labels=date_format("%b %y")))+
ylab("Sales")+
stat_smooth(colour="green")
#gemo_line, geom_bar, geom_point

#Kaggle Competition 1

mdl <- store1 %>% 
group_by(Store,DayOfWeek,Promo) %>% 
summarise(PredSales=mean(Sales)) %>% 
ungroup()

test <- store1 %>% left_join(mdl,by=c("Store","DayOfWeek","Promo"))

ar_fit = function(x) {
  test <- ts(x$logSales)
  lambda <- BoxCox.lambda(Sales)
  tsclean(Sales, replace.missing = TRUE, lambda=lambda)
  xreg <- cbind(DayOfWeek = x$DayOfWeek , 
                Open = x$Open,
                Promo = x$Promo,
                StateHoliday = x$StateHoliday,
                SchoolHoliday = x$SchoolHoliday,
                mSales = x$mSales
                )
  fit <- auto.arima(Sales, xreg=xreg)
  return(fit)
}


td = seq(as.Date("2013/1/1"), as.Date("2015/7/31"), "days") 
# "months", "years"
tsz = zoo(x=store1$Sales,order.by=td)
tsz[as.Date(c("2012/1/1","2014/1/1"))]
window(tsz, start=as.Date("2013/3/1"), end=as.Date("2013/4/1")) 

write_csv(df,"output.csv")
write_csv(df[c("Queue","Date","Time","volume")],"output.csv")


https://www.kaggle.com/emehdad/rossmann-store-sales/time-series-linear-models-tslm
https://www.kaggle.com/pchitta/rossmann-store-sales/arima-try/comments
http://robjhyndman.com/hyndsight/




