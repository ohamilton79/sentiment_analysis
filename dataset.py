import numpy as np
import os
from random import shuffle
import re
import pickle
from keras.preprocessing.text import Tokenizer

def extractRating(filename):
    """Extract the rating for the review
    from the filename
    """
    #Get the positions for the underscore and period
    underscorePos = filename.index("_")
    periodPos = filename.index(".")
    #The rating appears between the underscore and period
    rating = int(filename[underscorePos+1:periodPos])
    if rating > 6:
        return 1.0
    else:
        return 0.0

def getDirectory(rating, datasetDirectory):
    """Get the directory for a review
    based on whether it is positive or negative
    """

    #Positive ratings go from 0.7-1
    if rating == 1.0:
        return datasetDirectory + "pos/"

    #Negative ratings go from 0.1-0.4
    else:
        return datasetDirectory + "neg/"

def sanitiseText(text, sequenceLength):
    """Sanitise the text by removing invalid characters
    """
    #Convert to lowercase
    text = text.lower()
    #Remove HTML symbols
    text = re.sub('<.*?><.*?>', ' ', text)
    #Remove punctuation
    text = re.sub('[!\?£$%^\&*()\;:#\.,"`]', '', text)
    text = re.sub(' - ', ' ', text)
    text = re.sub(' -', ' ', text)
    text = re.sub('- ', ' ', text)
    #Trim the words to a specific sequence length
    words = text.split()
    return ' '.join(words[0:sequenceLength])

def generateWordMapping(texts):
    tokenizer = Tokenizer(num_words=30000)
    tokenizer.fit_on_texts(' '.join(texts).split())

    wordsToIndex = tokenizer.word_index
    
    #Save the tokenizer and word mapping objects to the disk
    with open("data/tokenizer.pkl", "wb") as tokenizerHandler:
        pickle.dump(tokenizer, tokenizerHandler)
        
    with open("data/wordsToIndex.pkl", "wb") as wordsToIndexHandler:
        pickle.dump(wordsToIndex, wordsToIndexHandler)
    

    #wordToVecMap = readGloveVectors()

    #embMatrix = np.zeros((len(wordsToIndex)+1, vecSpaceSize))

    #for word, index in wordsToIndex.items():
        #embVector = wordToVecMap.get(word)
        #if embVector is not None:
            #embMatrix[index, :] = embVector
            
    return wordsToIndex, tokenizer

#Read the review from the input files
def readDataset(datasetDirectory, sequenceLength, vecSpaceSize):
    #Read in the files
    positiveFiles = os.listdir(datasetDirectory + "pos/")
    negativeFiles = os.listdir(datasetDirectory + "neg/")
    #Concatenate the positive and negative training data items, and shuffle them
    allFiles = positiveFiles + negativeFiles
    shuffle(allFiles)

    #Collect the rating and review content from each file
    ratings = []
    reviews = []

    for filename in allFiles:
        #Get the rating for the file, scaled between 0 and 1
        rating = extractRating(filename)

        #Derive the directory stem from the rating
        directoryStem = getDirectory(rating, datasetDirectory)
        #Get the review within the file
        with open(directoryStem + filename, "r") as fileHandle:
            review = fileHandle.read()

        ratings.append(rating)
        #Sanitise the review text
        reviews.append(sanitiseText(review, sequenceLength))
        
    return reviews, ratings

