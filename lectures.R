library(arules)
data("Groceries")

# Build the rules
rules <- apriori(Groceries, parameter = list(supp = 0.01, conf = 0.3, target = "rules"))

# List the top 40 rules, ordered by lift
inspect(head(rules, n=40, by = "lift"))

# Calculate two more interestingness measures: hyperConfidence, and conviction
quality(rules) <- cbind(quality(rules),
                  hyperConfidence = interestMeasure(rules, measure = "hyperConfidence",
                                                          transactions = Groceries),
                  conviction = interestMeasure(rules, measure="conviction", transactions = Groceries))

# List the top 40 rules, ordered by lift
inspect(head(rules, n=40, by = "conviction"))