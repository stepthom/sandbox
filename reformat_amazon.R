# This script reformats the Amazon Review data from its original
# JSON format into a more-usable CSV format, which is easier/better
# for subsequent scripts to use.

library(tidyverse)
library(jsonlite)
library(rjson)
library(data.table)

setwd("/Users/stepthom/sandbox")

# The file is, for some reason, in a strange format, where each line is its own JSON.
dat <- lapply(readLines('data/reviews_Grocery_and_Gourmet_Food_5_50000.json'), fromJSON)

# The following calls fail with parse errors, due to the above
#dat = read_json('reviews_Grocery_and_Gourmet_Food_5.json', simplifyVector = TRUE)
#dat = fromJSON(file='reviews_Grocery_and_Gourmet_Food_5.json')

# Create a dataframe out of the data.
# (Actually, use a data.table for speed. Also use the set() method for great speedup.)
# The following code is an ugly hack. Unfortunately, simple methods like as.data.frame were not working,
# and in the interest of time, I just brute forced the creation of a dataframe by looping through
# the list of json docs manually.
df = data.table(
  reviewID = rep(1L, n),
  reviewerID = rep("", n),
  asin = rep("", n),
  reviewerName = rep("", n),
  reviewText = rep("", n),
  overall = rep(-1L, n) ,
  summary = rep("", n),
  unixReviewTime = rep(-1L, n),
  stringsAsFactors = FALSE
  )

i = 1L
for (u in dat) {
  # Some reviews have a NULL/empty reviewer name.
  if (!exists("reviewerName", where = u)) {
    rn = ""
  } else {
    rn = u$reviewerName
  }
  set(df, i=i, j=1L, value=i)
  set(df, i=i, j=2L, value=u$reviewerID)
  set(df, i=i, j=3L, value=u$asin)
  set(df, i=i, j=4L, value=rn)
  set(df, i=i, j=5L, value=u$reviewText)
  set(df, i=i, j=6L, value=u$overall)
  set(df, i=i, j=7L, value=u$summary)
  set(df, i=i, j=8L, value=u$unixReviewTime)
  i = i+1L
}

write_csv(df, "data/reviews_Grocery_and_Gourmet_Food_5_50000.csv")

# Be kind to memory, and remove unneeded stuff!
rm(dat)
