from cnn_test import performBatchTest
import sys

if len(sys.argv) == 2:
    performBatchTest(sys.argv[1])
    
else:
    print("The weights file to be used must be passed as an argument")