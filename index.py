import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nlp
import random
import pandas as pd
import csv


def show_history(h):
    epochs_trained = len(h.history['loss'])
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), h.history.get('accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_accuracy'), label='Validation')
    plt.ylim([0., 1.])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    
def show_confusion_matrix(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(8, 8))
    sp = plt.subplot(1, 1, 1)
    ctx = sp.matshow(cm)
    plt.xticks(list(range(0, 6)), labels=classes)
    plt.yticks(list(range(0, 6)), labels=classes)
    plt.colorbar(ctx)
    plt.show()

    
#print('Using TensorFlow version', tf.__version__)
names = ['id', 'user', 'tweet_id', 'content', 'likes', 'retweets', 'replies']
test_dataset2 = pd.read_csv('/Users/achyutpaudel/Desktop/StockPerformanceAnalysis/Tweets/Tweet.csv', on_bad_lines='warn', skipinitialspace=True, names=names)
print(test_dataset2.head())
trueTest = test_dataset2['content']
testList = list()
for x in trueTest:
    testList.append(x)


dataset = nlp.load_dataset('emotion')

#Get specific data from dataset
train = dataset['train']
val = dataset['validation']


def get_tweet(data):
    # Get raw text and emotion label for the dataset
    tweets = [x['text'] for x in data]
    labels = [x['label'] for x in data]
    return tweets, labels

def print_tweet(t, l):
    # Provided a list of tweets and a list of labels
    # Print a random tweet and its corresponding label
    randomIndex = random.randint(0, len(t))
    print("tweet:\n", t[randomIndex])
    print("label:", l[randomIndex])
    print()

def print_tokenized(tokenizer, tweets):
    #Print a random tweet and its corresponding tokenized version
    randomIndex = random.randint(0, len(tweets))
    print("tweet:\n", tweets[randomIndex])
    print("tokenized:", tokenizer.texts_to_sequences(tweets)[randomIndex])
    print()

tweets, emLabels = get_tweet(train)

print_tweet(tweets, emLabels)

#Tokenize the tweets
from tensorflow.keras.preprocessing.text import Tokenizer

#Initialize the tokenizer to only consider the top 10000 most frequently used words.
#Anything not in the top 10000 will be considered a oov and will be tokenized as <UNK>.
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')

#Tokenize the tweets dataset that was previously imported
tokenizer.fit_on_texts(tweets)

#Test tokenizer..
print("======Tokenizer Test======")
print_tokenized(tokenizer, tweets)

#Pad and truncate sequences from previous step
#We want to pad the sequences to the same length.
lengths = [len(tweet.split(' ')) for tweet in tweets]

#Analyze the histogram to determine a maximum length for the tweet
    #plt.hist(lengths) #Uncomment to see histogram
    #plt.show() #Uncomment to show histogram

maxlength = 50


#Begin padding the sequences to the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_sequences(tokenizer, tweets):
    sequences = tokenizer.texts_to_sequences(tweets)
    padded = pad_sequences(sequences, truncating='post', padding='post', maxlen=maxlength)
    return padded

padded_train_seq = get_sequences(tokenizer, tweets)

print("======Padded Test======")
randomIndex = random.randint(0, len(padded_train_seq))
print(tweets[randomIndex])
print(padded_train_seq[randomIndex])
print()


#Prepare the labels
classes = set(emLabels)

#Helper arrays to convert a label to a numerical representation
class_to_index = dict((c, i) for i, c in enumerate(classes))
index_to_class = dict((v, k) for k, v in enumerate(classes))

#Convert our dataset labels to a numerical representation
names_to_ids = lambda labels: np.array([class_to_index.get(label) for label in labels])
train_labels = names_to_ids(emLabels)


#Build the model

#Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=16, input_length=maxlength),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(6, activation='softmax')
    ])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


#Prepare validation set
val_tweets, val_labels = get_tweet(val)
val_seq = get_sequences(tokenizer, val_tweets)
val_labels = names_to_ids(val_labels)


#Train the model
history = model.fit(
    padded_train_seq,
    train_labels,
    validation_data=(val_seq, val_labels),
    epochs=20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
    ]
)

#Results
    #show_history(history) #Uncomment to see the training history

#Test the model with a different dataset of tweets:
test_dataset = dataset['test']
test_tweets, test_labels = get_tweet(test_dataset)

#Evaluate the model on the test dataset. 
#test_tweets is the dataset that is being predicted.
test_seq = get_sequences(tokenizer, testList)
#test_labels = names_to_ids(test_labels)
#_ = model.evaluate(test_seq, test_labels)

#Print the confusion matrix to check the accuracy of the model
preds = model.predict(test_seq)
#show_confusion_matrix(test_labels, preds, index_to_class.values())
