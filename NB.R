
library(tidyverse)



setwd("/Users/stepthom/sandbox")
df <- read.csv("NaiveBayes.csv")
df

nrow(df)




test2 <- data.frame(Age=c("<=30"),Income=c("Medium"),Student=c("yes"),Credit_Rating=c("fair"))
test2

df$Age = as.factor(df$Age)
df$Income = as.factor(df$Income)
df$Credit_Rating = as.factor(df$Credit_Rating)
df$Student=as.factor(df$Student)
df$buys_computer=as.factor(df$buys_computer)
df

test = df[15,]
train = df[1:14,]

test$Age = as.factor(test$Age)
test$Income = as.factor(test$Income)
test$Credit_Rating = as.factor(test$Credit_Rating)
test$Student=as.factor(test$Student)
test

train <- df
train

# Naive Bayes
#Load the required packages.

library(e1071)

#Build the model

nb <- naiveBayes(buys_computer~ ., data=train)
nb

#predict whether or not the test row buys a computer
predict(nb, test)

prediction <- predict(nb, test ,type="raw")
prediction

 #check to see what the prediction changes to with a different test df

test1 <-data.frame(Age=c("<=30"),Income=c("Low"),Student=c("no"),credit_rating=c("poor"))
test1

predict(nb, test1)

prediction <- predict(nb, test1 ,type="raw")
prediction
