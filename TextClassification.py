########################################################################
# Text clasification to identify nature-related pandemic articles
########################################################################
# load python libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
########################################################################
# laod data
path = path_df + 'dfNLP2.csv'
df = pd.read_csv(path)
df.info()
########################################################################
# Scikt Learn

y = df['category']

# Create training sets
X_train, X_test, y_train, y_test = train_test_split(df['NLPtext'], 
                                                    y,
                                                    test_size=0.33,
                                                    random_state=53,
                                                    stratify=y)

########################################################################
# count vectorizer

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words='english', min_df = 20)

# Set the training data 
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

########################################################################
# tfidf vectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df = 20)

# Set the training data 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Naive Bayes with  scikit-learn | tfidf vectorizer
nb_classifier2 = MultinomialNB()
nb_classifier2.fit(tfidf_train, y_train)
pred2 = nb_classifier2.predict(tfidf_test)

print('Accuracy score:', metrics.accuracy_score(y_test, pred2), '\n')
print(metrics.confusion_matrix(y_test, pred2, labels=[0,1]))
########################################################################
# Prepare data for scikit-learn
X_new = tfidf_vectorizer.transform(df_all['NLPtext'])

# Check X_new
print('X_new: ', X_new.shape[0])

# Predict y_new using the established data
y_new = nb_tfidf_classifier1.predict(X_new)

# Print result
print('y_new:', y_new.shape[0])
print('Selected:', y_new.sum())
########################################################################

