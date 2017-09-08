"""
Stephen W. Thomas

Clustering the 20NewsGroup dataset:

- Load data (from the scikit-learn package)
- Select subset of data
- Preprocess data
- Transform data into a term document matrix (TDM)
- Apply TF-IDF to TDM
- Run K-Means clustering on TDM
- Measure effectiveness
- Output results and documents

"""
from __future__ import print_function

import os
import re

import nltk.corpus
import nltk.stem
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


# The directory in which to output all output files
output_dir = "/Users/stepthom/Desktop/output"

####################################
# Load data
####################################

# For now, we'll just use these categories
categories = ['rec.sport.baseball', 'talk.religion.misc', 'comp.graphics','sci.space']
data = fetch_20newsgroups(subset='all', categories=categories, shuffle=False)

# Print some metrics about the data
print("Number of docs: %d:" % len(data.data))
print("Number of categories: %d" % len(data.target_names))
i = 0
for cat in range(len(data.target_names)):
    cat_name = data.target_names[cat]
    num_docs = len([d for d in data.target if d == i])
    print("Category %d (%s): %d docs" % (i, cat_name, num_docs))
    i += 1

labels = data.target
true_k = np.unique(labels).shape[0]

####################################
# Preprocessing
####################################

print("Preprocessing the data...")

# Define a new member, .pre, to hold the preprocessed version.
data.pre = []

# A list of custom stopwords to remove
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.update(
    ['subject', 're', 'don', 'jan',
     'feb', 'mar', 'apr', 'may', 'june',
     'july', 'aug', 'sep', 'oct', 'nov', 'dec'])

# Regex pattern for email addresses
email_pattern = re.compile(r'[^@]+@[^@]+\.[^@]+', re.IGNORECASE & re.UNICODE)

# Regex pattern for things found at the top "header" part of the message
header_pattern = re.compile(
    r'^summary:|^x-newsreader:.*|^date:.*'
    r'|^disclaimer:.*|^distribution:.*|^organization:.*'
    r'|^nntp-posting-host:.*|^keywords:.*|^to:.*'
    r'|^in-reply-to:.*|^x-news-reader:.*|^lines:.*',
    re.IGNORECASE & re.UNICODE)

# Regex pattern for general special characters
#special_char_pattern = re.compile(r'[\|=-\[\]\'\":;,\.\<\>\\\/\?_\(\)!$%^&*,]', re.IGNORECASE & re.UNICODE)
special_char_pattern = re.compile(r"[^\w']|_")

# Regex pattern for whole-word numbers
number_pattern = re.compile(r'\b\d+\b')

# Regex pattern to handle strings like the following:
# In article <me@me.com> me@me.com (Me) writes:
writes_pattern = re.compile(r'^.*writes:$')

# Loop through data do the preprocessing
for j in range(0, len(data.data)):
    lines = data.data[j].lower().split("\n")
    for i in range(0, len(lines)):

        # Use the regexes above to remove bad things
        lines[i] = header_pattern.sub(' ', lines[i])
        lines[i] = email_pattern.sub(' ', lines[i])
        lines[i] = number_pattern.sub(' ', lines[i])
        lines[i] = writes_pattern.sub(' ', lines[i])
        lines[i] = special_char_pattern.sub(' ', lines[i])

        # Remove short words
        lines[i] = ' '.join([w for w in lines[i].split() if len(w) > 2])

        # Remove stopwords
        lines[i] = ' '.join([w for w in lines[i].split() if w not in stopwords])

        # Stem the words
        lines[i] = ' '.join([nltk.stem.snowball.SnowballStemmer("english").stem(w) for w in lines[i].split()])

        # Remove extra spaces, just for beauty
        re.sub('\s\s+', " ", lines[i])

    pre = " ".join(lines)
    data.pre.append(pre)


####################################
# Build the TDM matrix
####################################

print("Building the TDM...")

vectorizer = TfidfVectorizer(max_df=0.6, max_features=10000,
                             min_df=2, stop_words='english',
                             use_idf=True)
tdm = vectorizer.fit_transform(data.pre)


####################################
# Run the clustering algorithm
####################################

print("Running the clustering algorithm...")

km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=10, verbose=False)
km.fit(tdm)


####################################
# Measure effectiveness
####################################

print("\nEvaluation metrics:")

# Rows are true, colums are pred
print("Confusion matrix:")
print(metrics.confusion_matrix(y_true=labels, y_pred=km.labels_))
mat = metrics.confusion_matrix(y_true=labels, y_pred=km.labels_)
mat.data
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(tdm, km.labels_, sample_size=1000))

print()


####################################
# Output results files
####################################

print("Writing results files...")

# Get the vocabulary (i.e., terms mapped to their indices)
vocab = vectorizer.vocabulary_

# Inverse the vocabulary (i.e., indices mapped to terms)
ivocab = {}
for term, index in vocab.iteritems():
    ivocab[index] = term

f = open(os.path.join(output_dir, "0000-terms.txt"), 'w')
for w in sorted(vocab, key=vocab.get, reverse=False):
    f.write("{}, {}\n".format(w.encode("utf-8"), vocab[w]))
f.close()

# The top terms for each cluster
cluster_terms = []

order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
f = open(os.path.join(output_dir, "0000-clusters.txt"), 'w')
for i in range(true_k):
    f.write("Cluster {}\n".format(i))
    f.write("Num docs: {}\n".format(len([x for x in p if x == i])))
    f.write("Top terms: ".format(i))
    cterms = []
    for ind in order_centroids[i, :10]:
        f.write(' {}'.format(terms[ind]))
        cterms.append(terms[ind])
    cluster_terms.append(' '.join(cterms))
    f.write("\n")

# for j in range(0, len(data.data)):
for j in range(0, 100):
    filename = os.path.basename(data.filenames[j])
    outfilename = "doc-{}-{}-{}.txt".format(km.labels_[j], data.target[j], filename)
    f = open(os.path.join(output_dir, outfilename), 'w')
    f.write("Filename = {}\n".format(data.filenames[j]))
    f.write("Target = {} ({})\n".format(data.target[j], data.target_names[data.target[j]]))
    f.write("Estimated = {} ({})\n".format(km.labels_[j], cluster_terms[km.labels_[j]]))
    f.write("\nRaw:\n")
    f.write("===============================================\n")
    f.write(data.data[j].encode("utf-8"))
    f.write("\n===============================================\n")
    f.write("\nRaw (repr):\n")
    f.write("===============================================\n")
    f.write(repr(data.data[j]))
    f.write("\n===============================================\n")
    f.write("\nPreprocessed: {}\n".format(data.pre[j].encode("utf-8")))
    f.write("\nFeature vector (sorted, non-zero values only): ]\n")

    Xa = tdm.toarray()[j]
    sorted_ind = sorted(range(len(Xa)), key=lambda k: Xa[k])
    for ind in reversed(sorted_ind):
        val = Xa[ind]
        if val > 0:
            f.write(("ind={}, term={}, value={}\n".format(ind, ivocab[ind], val)))