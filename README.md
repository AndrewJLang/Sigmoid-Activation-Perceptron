# Perceptron

## Instructions:

 - All program files were compiled and ran on Python 3.7.1

 - The external libraries used in this project were cv2 (OpenCV), csv, os, numpy, matplotlib.pyplot, random, pandas and math

 ## ColorHistogram.py 
  - This file is where the categorized images are written into a file that will be used for the perceptron part of the project.  Currently, method is commented out so file won't accidently have additional, unneccessary information in it.  The user only needs to specify the folder they wish to pull the images from and where they wish the histograms to be written to.

 ## Perceptron.py 
  - This file is where the images are compared to the perceptron's output using the sigmoid activation function. A set amount of datapoints are removed from the array that will be modeled and set aside for validation purposes. The best weights from the model are also stored so they can be used to categorized the validation data.
  - File writes accuracy vs epochs and error vs epochs to csv file to be graphed.
  - Step activation function is commented out as sigmoid provided better accuracy during testing.
  - User can specify learning rate, bias, epochs and batch size. Learning rate is only recommended value that user can change as bias, epochs and batch size were all tested and current values provide the best results.

## Note:
dataFile.txt is provided so ColorHistogram does not have to be run (if run, change file it is written to in ColorHistogram.py and which file it is being read from in Perceptron.py)