import message_preprocessing as mp
import preprocessing_helpers as hp

import train_test_split as tts

""" Perform custom vectorization to indicate the presence of features from bow in each documents messages,
True for all those bow features which are also present in the document """


## creating a LazyMap of feature presence for each of the 8K+ features with respect to each of the SMS messages
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in mp.bag_of_words:
        features["contains(%s)" % word] = word in document_words
    return features


## - creating the feature map of train and test data
training_set = hp.nltk.classify.apply_features(extract_features, tts.train_messages)
testing_set = hp.nltk.classify.apply_features(extract_features, tts.test_messages)

# print(training_set[:5])

# print("Training set size : ", len(training_set))
# print("Test set size : ", len(testing_set))
