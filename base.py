"""
░██████╗░██╗██╗░░░░░██████╗░███████╗██████╗░████████╗
██╔════╝░██║██║░░░░░██╔══██╗██╔════╝██╔══██╗╚══██╔══╝
██║░░██╗░██║██║░░░░░██████╦╝█████╗░░██████╔╝░░░██║░░░
██║░░╚██╗██║██║░░░░░██╔══██╗██╔══╝░░██╔══██╗░░░██║░░░
╚██████╔╝██║███████╗██████╦╝███████╗██║░░██║░░░██║░░░
░╚═════╝░╚═╝╚══════╝╚═════╝░╚══════╝╚═╝░░╚═╝░░░╚═╝░░░ v1.1

TODO:
- Uncomment
- Find new data
- Improve loading animations
- Choose different neural network
- If below a certainty, say IDK
- Speech to text
    * Implement "listening...." animation
    * Try / except statement in listening
- Text to speech

From there, we can train the model on specific input data
Make it flexible for problems by variably optimizing neural network
Make it more general, open to problem solving rather than conversation
"""

# loading animation

import os
import time
import threading
import itertools
import sys

os.system('clear')

def animate(birdword):
    for c in ['', '.', '..', '...', '..', '.']:
        print('{b}'.format(b=birdword) + c)
        time.sleep(0.5)
        os.system('clear')

# import packages

def loader():
    import nltk
    #nltk.download('punkt')
    #nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    import json
    import pickle
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    from keras.optimizers import SGD
    import random

# t = threading.Thread(name="load", target=loader)
# t.start()
# while t.isAlive():
#     animate("packages")
# print("Done!")
# time.sleep(0.5)

# intialize training
print("Loading packages...")
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import pyttsx3
import speech_recognition as sr
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#print("Done!")
#time.sleep(0.4)
#os.system('clear')


print("Loading training data...")
words= []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))
        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# lemmatize means to turn a word into its base meaning, or its lemma
# this is similar to stemming, which reduces an inflected word down to its root form.
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
# print (len(documents), "documents")
# print (len(classes), "classes", classes)
# print (len(words), "unique lemmatized words", words)
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# building deep learning model

training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
#print("Done!")
time.sleep(0.4)
#os.system('clear')

# Intitializing Neural network

print("Intializing Neural Network...")
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=False)

#print("Done!")
#time.sleep(0.4)
#os.system('clear')

# Preparation for GUI

print("Preparing GUI...")

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# cleans up any sentence inputted

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))
# takes the sentences that are cleaned up and creates a bag of words that are used for predicting classes

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
# error threshold of 0.25 to avoid too much overfitting.
# This function will output a list of intents and the probabilities, their likelihood of matching the correct intent

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result
# takes the list outputted and checks the json file and outputs the most response with the highest probability.

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

#print("Done!")
#time.sleep(0.4)
#os.system('clear')

# GUI

def listen():
    with sr.Microphone() as source2:
        audio2 = r.listen(source2)

r = sr.Recognizer()
r.pause_threshold = 0.5
print("Calibrating microphone (5 seconds)...")
with sr.Microphone() as source2:
    r.adjust_for_ambient_noise(source2, duration=5)
#print("Done!")
#time.sleep(0.4)

def gooey():
    os.system('clear')
    print("""
░██████╗░██╗██╗░░░░░██████╗░███████╗██████╗░████████╗
██╔════╝░██║██║░░░░░██╔══██╗██╔════╝██╔══██╗╚══██╔══╝
██║░░██╗░██║██║░░░░░██████╦╝█████╗░░██████╔╝░░░██║░░░
██║░░╚██╗██║██║░░░░░██╔══██╗██╔══╝░░██╔══██╗░░░██║░░░
╚██████╔╝██║███████╗██████╦╝███████╗██║░░██║░░░██║░░░
░╚═════╝░╚═╝╚══════╝╚═════╝░╚══════╝╚═╝░░╚═╝░░░╚═╝░░░ v1.1
    """)
    time.sleep(0.5)
    print("Built by kakn 27/12/2020. Hit enter to reply to Gilbert \n")
    engine = pyttsx3.init()
    print("GILBERT: What is your name?")
    engine.say("What is your name?")
    engine.runAndWait()
    enter = input()
    if enter == "":
        try:
            with sr.Microphone() as source2:
                #r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)
            name = r.recognize_google(audio2)
        except sr.UnknownValueError:
            engine.say("I'm sorry, I don't understand.")
            print("GILBERT: I'm sorry, I don't understand.")
            engine.runAndWait()
            gooey()
    #print(name)
    #time.sleep(0.5)
    print("GILBERT: Hello, {n}".format(n=name))
    engine.say("Hello, {n}".format(n=name))
    engine.runAndWait()
    convo = True
    while convo == True:
        try:
            enter = input()
            if enter == "":
                with sr.Microphone() as source2:
                    #r.adjust_for_ambient_noise(source2, duration=0.2)
                    audio2 = r.listen(source2)
                question = r.recognize_google(audio2)
            print('{n}: '.format(n=name.upper()) + question)
            if question == "terminate":
                engine.say("Goodbye.")
                break
                engine.runAndWait()
                convo = False
            answer = chatbot_response(question)
            print('GILBERT: ' + answer)
            engine.say(answer)
            engine.runAndWait()
            with open("data_stored.txt", "a") as text_file:
                text_file.write("{n}: ".format(n=name.upper()) + question + '\n' + "GILBERT: " + answer + '\n' + '\n')
        except sr.UnknownValueError:
            print("GILBERT: I'm sorry, I don't understand.")
            engine.say("I'm sorry, I don't understand.")
            engine.runAndWait()

# running functions
gooey()
