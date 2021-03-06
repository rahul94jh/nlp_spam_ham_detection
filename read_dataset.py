import pandas as pd

""" Read the input data (spam collection txt file) into a pandas dataframe """

## Reading the given dataset
spam = pd.read_csv("data/SMSSpamCollection.txt", sep="\t", names=["label", "message"])
# print(spam.head())

## Converting the read dataset in to a list of tuples, each tuple(row) contianing the message and it's label
data_set = []
for index, row in spam.iterrows():
    data_set.append((row.message, row.label))

# print(data_set[:5])
