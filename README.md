# Sandbox

This repository holds scripts and notebooks for Steve's musings, investigations, case studies,
animations, and slides.

Here's a high-level snapshot of each script.

## Non-text Analytics

| File  | Language  | Dataset  | Package   |Notes |
|---|---|---|---|---|
| `NB.R`  |  R |  `NaiveBayes.csv` | `e1071`   | Simple example of NB.  |
| `arules.Rmd`  | R  | `arules::Groceries` |  `arules`, `arulesViz` |   |
| `bigdata.Rmd`  |  R | N/A  |  `tidyverse` | Just some charts for the big data slides.  |
| `classifiers.R` | R | `laheart.csv` | `rpart`, `e1071`, `MLmetrics` | Compares NB and DT. |
| `intro.Rmd` | R | `gapminder` | `tidyr`, `dplyr`, `ggplot2` | An intro to R and the tidyverse.  |
| `recSys.R` | R | `recommenderlab::MovieLense` | `recommenderlab` | Recommendation system for Movie Lense data. Uses CF. |
| `slide_plots.Rmd` | R | `chirps.csv`, `Prestige.txt`, `clusters.csv` | `tidytext`, `tm`, `tidyverse` | Just a script to create some plots/charts I've used in slides. |
| `spark-sample.mdR` | R | `nycflights13`, `Lahman` | `sparklyr` | Simple of example of how to use `sparklyr`. |
| `sql.Rmd` | R | `customer.csv`, `transaction.csv` | `sqldf` | Shows how to use the `sqldf` package. Used for some of my slides on SQL. |
| `sqlChallenge.Rmd` | R | `Lahman` | `sqldf` | Used for creating the SQL challenge. |
| `titanic.Rmd` | R | `titanic` | `tidyverse`, `rpart`, `MLmetrics` | Titanic case study. Builds a DT to predict survival. |



## Text Analytics

| File  | Language  | Dataset  | Package   |Notes |
|---|---|---|---|---|
| `cluster_20.ipynb` | Python | `sklearn.datasets::20newsgroups` | `nltk`, `sklearn` | Clustering the 20 Newsgroup dataset. |
| `imdb.Rmd` | R | `all.imdb.pipe.csv` | `tidytext`, `cleanNLP`, `tm` | Classifying IMDB data. |
| `kiva.Rmd` | R | `kiva.csv` | `tidytext`, `topicmodels`, `rpart`, `MLmetrics` | Classifying KIVA loans. Used as a case study. |
| `nltk-cluster.py` | Python | `sklearn.datasets::20newsgroups` | `nltk`, `sklearn` | I'm not sure how this is different from `cluster_20.ipynb` |
| `sentiment-manning.Rmd` | R | `manning.csv`, `brady.csv` | `tidytext` | Sentiment analysis on tweets about Peyton Manning and Tom Brady. |
| `slides_sentiment.R` | R | N/A | `tidytext` | Just a script to do some simple tidy-based sentiment analysis on some made-up data. |
| `slides_text_amazon.Rmd` | R | `reviews_Grocery_and_Gourmet_Food_5_50000.csv` | `tidytext`, `tm`, `wordcloud` | Descriptive stats on Amazon Reviews (Food category). |
| `slides_text_amazon_classify.R` | R | `reviews_Grocery_and_Gourmet_Food_5_50000.csv` | `tidytext`, `tm`, `caret` | Classifying Amazon reviews. |
| `slides_text_reuters.Rmd` | R | `reutersCSV.csv` | `tidytext`, `tm`, `wordcloud`  | Descriptive stats on Reuters dataset. |

## Data

Note: the source isn't actually "Unknown" for most of the data files below. I just haven't done it yet.

| File  | Source
|---|---|
| `HR_comma_sep.csv` | Unknown |
| `Master.csv` | Unknown |
| `NaiveBayes.csv` | Unknown |
| `Prestige.txt` | Unknown |
| `Salaries.csv` | Unknown |
| `all.imdb.pipe.csv` | Unknown |
| `alltweets.csv` | Unknown |
| `beta.csv` | Unknown |
| `beta_12.csv` | Unknown |
| `chirps.csv` | Unknown |
| `clusters.csv` | Unknown |
| `customer.csv` | Unknown |
| `gamma.csv` | Unknown |
| `gamma_12.csv` | Unknown |
| `jackastors.csv` | Unknown |
| `kiva..csv` | Unknown |
| `laheart.csv` | Unknown |
| `laheart2.csv` | Unknown |
| `site.csv` | Unknown |
| `student.csv` | Unknown |
| `survey.csv` | Unknown |
| `topicnames_12.csv` | Unknown |
| `transaction.csv` | Unknown |
| `visited.csv` | Unknown |
| `groceries.csv` | Unknown |
| `loan_small.csv` | Unknown |
| `all.imdb.pipe.csv` | Unknown |
| `brady.csv` | Unknown |
| `manning.csv` | Unknown |
| `reutersCSV.csv` | Unknown |
| `reviews_Grocery_and_Gourmet_Food_5_50000.csv` | Unknown |