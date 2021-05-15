import tensorflow as tf
from keras import Model, Input
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Activation, Flatten, Reshape, Concatenate
from keras.layers import BatchNormalization, AlphaDropout, Embedding, MaxPooling1D
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU

#Define the network architecture
def loadModel(vocabularySize, sequenceLength, vecSpaceSize):
    kernelSizes = [1, 2, 3, 4, 5]
    #kernelSizes2 = [3, 4, 5, 6, 7]
    #init = RandomNormal(stddev=0.02)
    inputs = Input(shape=(sequenceLength,))
    embedding = Embedding(vocabularySize, vecSpaceSize)(inputs)
    #reshape = Reshape((sequenceLength, vecSpaceSize, 1))(embedding)

    conved = [Conv1D(filters=32, kernel_size=kernelSize, activation="selu", kernel_initializer="lecun_normal")(embedding) for kernelSize in kernelSizes]
    
    #convedActivated = [LeakyReLU(alpha=0.20)(conv) for conv in conved]
    pooled = [MaxPooling1D(2, strides=2)(conv) for conv in conved]
    #batchNorms1 = [BatchNormalization()(pool) for pool in pooled]
    #dropouts = [Dropout(0.30)(pool) for pool in pooled]  

    conved2 = [Conv1D(filters=64, kernel_size=5, strides=3, activation="selu", kernel_initializer="lecun_normal")(pool) for pool in pooled]
    #convedActivated2 = [LeakyReLU(alpha=0.20)(conv) for conv in conved2]
    pooled2 = [MaxPooling1D(2, strides=2)(conv) for conv in conved2]
    
    #batchNorms2 = [BatchNormalization()(pool) for pool in pooled2]
    #dropouts = [Dropout(0.30)(pool) for pool in pooled]  

    #conved3 = [Conv1D(filters=32, kernel_size=3, strides=2, activation="relu")(batchNorm) for batchNorm in batchNorms2]
    #convedActivated2 = [LeakyReLU(alpha=0.20)(conv) for conv in conved2]
    #pooled3 = [MaxPooling1D(2, strides=2)(conv) for conv in conved3]
    #batchNorms2 = [BatchNormalization()(pool) for pool in pooled2]
    #dropouts2 = [Dropout(0.50)(pool) for pool in pooled2]
    flattened = [Flatten()(pool) for pool in pooled]

    merged = Concatenate()(flattened)
    dropoutDense1 = AlphaDropout(0.30)(merged)
    
    dense = Dense(10, activation="selu", kernel_initializer="lecun_normal")(dropoutDense1)
    #leakyReLU = LeakyReLU(alpha=0.20)(dense)
    #batchNorm3 = BatchNormalization()(leakyReLU)
    dropoutDense2 = AlphaDropout(0.20)(dense)

    outputs = Dense(1, activation="sigmoid", kernel_initializer="lecun_normal")(dropoutDense2)

    model = Model(inputs=inputs, outputs=outputs, name="sentiment_analysis")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
