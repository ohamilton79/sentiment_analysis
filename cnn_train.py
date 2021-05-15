from dataset import generateWordMapping, readDataset
from model import loadModel
from cnn_test import performBatchTest
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

class EvaluationCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        """At the end of every 5th epoch, evaluate model performance
        """
        if epoch % 1 == 0:
            performBatchTest("data/weights-{}.hdf5".format(epoch+1))

'''def encode(text, encoding, maxLen):
    """Encode a piece of text as a list of integers of a specified length
    """
    encoded = np.array([encoding[word] for word in text.split()])
    #Pad with zeros if the max length isn't utilised
    paddedEncoding = np.zeros((maxLen))
    paddedEncoding[:encoded.shape[0]] = encoded

    return paddedEncoding'''

"""def readGloveVectors():
    wordToVecMap = {}

    loadedVectors = np.load("embeddings.npy", mmap_mode="r")
    with open("embeddings.vocab", "r", encoding="utf8") as fileRead:
        for index, word in enumerate(fileRead):
            wordToVecMap[word.strip()] = loadedVectors[index]
            
    return wordToVecMap"""

datasetDirectory = "aclImdb/train/"
sequenceLength = 300
vecSpaceSize = 8

reviews, ratings = readDataset(datasetDirectory, sequenceLength, vecSpaceSize)
#Get embedded matrix representing vocabulary
wordsToIndex, tokenizer = generateWordMapping(reviews)
#Generate model and output summary
model = loadModel(len(wordsToIndex)+1, sequenceLength, vecSpaceSize)
model.summary()
#Define weights checkpoint
filepath = "data/weights-{epoch:d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='min')

#Train the model
X = tokenizer.texts_to_sequences(reviews)
X = pad_sequences(X, maxlen=sequenceLength, padding='post')

#print(encodedReviews.shape)
model.fit(X, ratings, epochs=10, batch_size=32, shuffle=True, callbacks=[checkpoint, EvaluationCallback()])
print(model.metrics_names)

