import pickle

import preprocessing_helpers as hp
import train_test_feature_map as ttm


""" Until now, we have preprocessed the message in each documets, tokenized the messages, extracted the features as bow, 
we also splitted the messages in train and test and we have mapped the features for each document based on bow.
Now as each document in train set have the features associated with it with the information whether the document contains 
the specific feature or not ? let's build a classifier which can detect  the sms messages as spam or ham based on the 
presence of the certain features in documents. """

# Training the classifier with NaiveBayes algorithm
spamClassifier = hp.nltk.NaiveBayesClassifier.train(ttm.training_set)

## - Analyzing the accuracy of the train set
print(
    hp.nltk.classify.accuracy(spamClassifier, ttm.training_set)
)  # Train Accuracy :- 0.99


## - Analyzing the accuracy of the test set
print(
    hp.nltk.classify.accuracy(spamClassifier, ttm.testing_set)
)  # Test Accuracy :- 0.98


## Testing a example message with our newly trained classifier
m = "CONGRATULATIONS!! As a valued account holder you have been selected to receive a £900 prize reward! Valid 12 hours only."
print(
    "Classification result : ", spamClassifier.classify(ttm.extract_features(m.split()))
)

""" Now, let's see the probability the classifier attached to each features in our bow to be classified as spam or ham.
We can see that words like 'urgent, 'award', 'code', 'service' e.t.c have high prob. for classifying as spam """

## Priting the most informative features in the classifier
print(spamClassifier.show_most_informative_features(50))

## storing the classifier on disk for later usage
f = open("data/nb_spam_classifier.pickle", "wb")
pickle.dump(spamClassifier, f)
print("Classifier stored at ", f.name)
f.close()
