import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("data.csv")

data.head()

#this is a classic supervised machine learning problem -> to detect it a text/email is spam
#how do I get the features to build a NLP model?
#I have to turn this into a bag of words
#count vectorizer is a class that turns a text/corpus into a bag of words model performing already the usual stemming etc.

# create the vectorizer object

count_vectorizer = CountVectorizer()

# usual train-test split for fitting and testing our machine learning model
X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Category'])

# the fit transform method looks at all the words in the dataset and creates the bag of words, but also then transforms the orgiinal dataset into bag of words vectors
X_train_bag = count_vectorizer.fit_transform(X_train)

# okay now that we have done the fitting of the bag of words, we need to transform the test data
X_test_bag = count_vectorizer.transform(X_test)

# build a quick classifier
# initializing the model object
model = MultinomialNB()

# fitting the model object
model.fit(X_train_bag, y_train)

# lets see how we did
# with the test data
model.score(X_test_bag, y_test)

#let's try to make a prediction on new data
input_text = "Free perfume"
model.predict(count_vectorizer.transform([input_text]))

"""
we learn that if I want to make new predictions I need two things:
the count_vectorizer
the model (model.predict)
and so I am going to need to save both these objects save these objects to a pickle file
"""

import pickle

# create two pickle file objects
#vectorizer
vectorizer = "vectorizer.pkl"

# opening the file
with open(vectorizer, 'wb') as file:
  pickle.dump(count_vectorizer, file)

# dumping the vectorizer object to the pickl

model_file = "spam_classifier.pkl"

with open(model_file, 'wb') as file:
  pickle.dump(model, file)

# dumping the model to the pickle
# quickly read a trained model, apply the pre processing techniques and make a prediction

with open("vectorizer.pkl", 'rb') as file:
    vectorizer = pickle.load(file)

# import the classifier pickle object
with open("spam_classifier.pkl", 'rb') as file:
    read_model = pickle.load(file)

read_model.predict(vectorizer.transform(["free ham"]))