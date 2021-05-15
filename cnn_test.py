from model import loadModel
from dataset import generateWordMapping, readDataset
from keras.preprocessing.sequence import pad_sequences
import pickle

def loadWordMapping():
    """Loads the tokenizer and word mapping objects from the disk
    """
    with open("data/tokenizer.pkl", "rb") as tokenizerHandler:
        tokenizer = pickle.load(tokenizerHandler)
        
    with open("data/wordsToIndex.pkl", "rb") as wordsToIndexHandler:
        wordsToIndex = pickle.load(wordsToIndexHandler)
        
    return tokenizer, wordsToIndex

def performBatchTest(weightsFilename):
    tokenizer, wordsToIndex = loadWordMapping()
    datasetDirectory = "aclImdb/test/"
    sequenceLength = 300
    vecSpaceSize = 8

    print("Evaluating model performance...")
    #Get the testing dataset
    reviews, ratings = readDataset(datasetDirectory, sequenceLength, vecSpaceSize)
    
    #Load the model and its weights
    model = loadModel(len(wordsToIndex)+1, sequenceLength, vecSpaceSize)
    model.load_weights(weightsFilename)
                
    X = tokenizer.texts_to_sequences(reviews)
    X = pad_sequences(X, maxlen=sequenceLength, padding='post')
    loss, accuracy = model.evaluate(X, ratings)
    print("\tLoss: {}".format(loss))
    print("\tAccuracy: {}".format(accuracy))
    
def performIndividualTest(inputSentence, weightsFilename):
    tokenizer, wordsToIndex = loadWordMapping()
    sequenceLength = 300
    vecSpaceSize = 8
    
    #Load the model and its weights
    model = loadModel(len(wordsToIndex)+1, sequenceLength, vecSpaceSize)
    model.load_weights(weightsFilename)
                
    X = tokenizer.texts_to_sequences([inputSentence])
    X = pad_sequences(X, maxlen=sequenceLength, padding='post')
    result = model.predict(X)
    print("\tPredicted: {}".format(result))
    
