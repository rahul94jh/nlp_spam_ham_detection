import random
import message_preprocessing as ms

""" Split the messages into train and test set """

# Get the message set
messages_set = ms.messages_set

## Preparing to create a train and test set
## - creating slicing index at 80% threshold
sliceIndex = int((len(messages_set) * 0.8))

## - shuffle the pack to create a random and unbiased split of the dataset
random.shuffle(messages_set)

# split the message set
train_messages, test_messages = messages_set[:sliceIndex], messages_set[sliceIndex:]

# print(len(train_messages))
# print(len(test_messages))
