import random
import read_dataset as rd
import preprocessing_helpers as hp

""" Preprocess the messages and create bag of words """

## - Performing the preprocessing steps on all messages
messages_set = hp.filter_message(rd.data_set)
# print(messages_set[:5])

# Preparing to create bag of words
## - creating the bow for the entire dataset
# Extract messages from messages_set
messages = [i[0] for i in messages_set]
bag_of_words = hp.create_bag_of_words(messages)
# print(len(bag_of_words))
