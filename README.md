# Speaker-Identification
This is a program to train a deep neural network for the task of speaker identification.

The program depends on the following installations :-
	
	1) TensorFlow or Theano (backend for Keras)
	
	2) Keras
	
	3) Scipy
	
	4) Numpy
	
	5) Scikit-learn
	
	6) Mlpy
	
	7) Scikits.talkbox

There are two modes to run the program :-
	
	1) Make changes in code itself for specifying training and testing data. 
		Change the string variables storing the pathnames of the train 'direc' and test directories 'testdirec' in the file runMode_1.py. 
		The train directory should contain folders named as numbers starting from 0 representing the Speaker No. or class label for classification. 
		Each such folder should contain voice samples (.wav files) of the corresponding user. 
		Same for the test directory. 
	
	2) Use command line arguments for specifying training and testing data. Run the file runMode_2.py by using the following arguments.
		python runMode_2.py train /path to train directory  for training
		python runMode_2.py test phone_number /path to test file for test

The basic architecture of the program remains the same for both the files.


