# Text Classification of the Amazon Reviews (Food) dataset.
#
# Using the tm package, let's do the entire classification process:
# - Preprocessing
# - Term weighting
# - Split data
# - Building models


library(tidytext)
library(tidyr)
library(dplyr)
library(ggplot2)
library(readr)
library(tm)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)


# Read in the data
df = read_csv("data/reviews_Grocery_and_Gourmet_Food_5_50000.csv")

# Take sample?
take_sample = TRUE
if (take_sample) {
  df = df %>%
    sample_frac(size = 0.01, replace=FALSE)
}

dim(df)
head(df)
str(df)

# A vector source interprets each element of the vector as a document
sourceData <- VectorSource(df$reviewText)
corpus <- Corpus(sourceData)

# Example document before pre-processing
corpus[[20]]$content

# preprocess/clean the training corpus
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords('english'))
corpus <- tm_map(corpus, stemDocument, language = "english") 
corpus <- tm_map(corpus, stripWhitespace)

# example document after pre-processing
corpus[[20]]$content

# create term document matrix (tdm)
tdm <- DocumentTermMatrix(corpus, 
                          control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))

# inspecting the tdm
dim(tdm)


# Split into training and testing.
N = nrow(tdm)

set.seed(1234)
idx_train = sample(seq_len(N), size=floor(0.80 * N))

df_tdm = as.data.frame(as.matrix(tdm))

df_tdm$is.positive = df$overall >= 5
df_tdm$is.positive = factor(df_tdm$is.positive, labels = c("yes", "no"))

df_train = df_tdm[idx_train, ]
df_test = df_tdm[-idx_train, ]

table(df_train$is.positive)
table(df_test$is.positive)


# set resampling scheme
ctrl <- trainControl(method="repeatedcv", number = 10, repeats = 3) #,classProbs=TRUE)

# fit a kNN model using the weighted (td-idf) term document matrix
# tuning parameter: K

idx = which(colnames(df_train)=="is.positive")
x = df_train[,-idx]
y = df_train[,idx]

dim(x)
str(x)
dim(y)
x[1:4, 1:10]
y
set.seed(100)
knn <- train(x= x, y=y, method = "knn", trControl = ctrl)
knn
knn$finalModel

set.seed(100)
dt <- train(x= x, y=y, method = "rpart", trControl = ctrl)
dt
dt$finalModel
rpart.plot(dt$finalModel, extra=2)


set.seed(100)
svm <- train(x= x, y=y, method = "svmRadial", trControl = ctrl) #, tuneLength = 20)
svm
svm$results
svm$finalModel

set.seed(100)
m <- train(x= x, y=y, method = "rf", trControl = ctrl) #, tuneLength = 20)
m
m$results

models = c("rf", "neuralnet", "nnet", "svmRadial", "svmLinear", "rpart", "knn", "xgbTree",
           "nb", "lm", "LogitBoos", "lasso", "J48", "glmboost", "C5.0Tree", "C5.0Rules", 
           "adaboost")

set.seed(100)
m <- train(x= x, y=y, method = "neuralnet", trControl = ctrl)
m
m$finalModel


names(getModelInfo())


# predict on test data
knn.tfidf.predict <- predict(knn.tfidf, newdata = tdm_test)



