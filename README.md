# sentiment_analysis
Using convolutional neural networks to perform sentiment analysis using the IMDb "Large Movie Review Dataset"

## Getting the dataset
The dataset can be downloaded by performing the following commands in your working directory (Linux):
```
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xvf aclImdb_v1.tar.gz
```

## Dependencies
The only dependencies are Tensorflow, Keras and h5py. Run the following:
```
pip install tensorflow keras 'h5py<3.0.0'
```
## Training
* When training the model, the word-to-integer mapping, tokenizer, and weights files will be stored in the `data` directory. 
* **WARNING**: This process will overwrite existing pre-trained weights.
* Run the following command to train the model:
```
python cnn_train.py
```

## Testing
* To test the model on the testing data provided by the IMDb dataset, run the following, specifying:
* * The location of the weights file you want to use relative to the working directory:
(`data/weights-2.hdf5` is the recommended value for the `path_to_weights_file` parameter)
```
python batch_test.py 'path_to_weights_file'
```
* To test the model's output in response to a given sentence as input, run the following, specifying:
* * The input sentence to test the model's output for
* * The location of the weights file you want to use relative to the working directory:
(`data/weights-2.hdf5` is the recommended value for the `path_to_weights_file` parameter)
```
python individual_test.py 'input_sentence' 'path_to_weights_file'
```

## Examples
```python
>>> python individual_test.py 'This movie is appalling. Waste of 2 hours of my life!' 'data/weights-2.hdf5'
Predicted: [[0.04231749]]
>>> python individual_test.py 'This movie is great. I loved the plot and characters!' 'data/weights-2.hdf5'
Predicted: [[0.8849266]]
>>> python individual_test.py "My friends liked this movie, but I wasn't keen on the main character" 'data/weights-2.hdf5'
Predicted: [[0.76930517]]
