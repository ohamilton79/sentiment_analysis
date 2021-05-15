from cnn_test import performIndividualTest
import sys

if len(sys.argv) == 3:
    #The first argument is the input sentence, the second is the weights filename
    performIndividualTest(sys.argv[1], sys.argv[2])
    
else:
    print("The input sentence and weights file to be used must be passed as arguments")