#Andrew Lang
#Perceptron implementation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math as e

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Perceptron Learning')

textData = []
with open('dataFile.txt') as my_file:
    textData = my_file.readlines()

hists = []

for i in range(len(textData)):
    newstr = textData[i].replace("[", "")
    newstr = newstr.replace("]", "")
    newstr = newstr.replace(".", ",")
    newstr = newstr.replace(":", ",")
    newstr = newstr[:-2]
    hist = [float(item) for item in newstr.split(',')]
    val = hist[0]
    hist = hist[1:]
    hist = np.true_divide(hist, np.max(hist))
    hist = np.insert(hist, 0, val, axis=0)
    hists.append(hist)

data = np.array(hists)

test = np.array(data)

#Create testing array and remove those values from the data array
test = test[:31]
data = data[31:]

#tested learning rates of 0.001, 0.01, 0.1, and 1.5 (all will get to 100% model accuracy within the given epoch count)
learning_rate = 0.1

L = data[:, 0]

batch_size = len(data)
epochs = 10
accuracy = 0
bias = 0

accuracyArr = []
#stores best W values that will be used in validation/testing
bestW = []

#Fill the array of W values with 0 to begin with
W = []
for i in range(len(data[0])):
    W.append(0)
    bestW.append(0)

#storage of info to be written to csv file
errors = []
classificationAccuracy = []
epochArray = []


sigmoid = 0

#sigmoid function - got a better validation using this method over step activation
for x in range(epochs):
    bestW = W
    accuracy = 0
    rand_batch = data[np.random.choice(data.shape[0], batch_size, replace=False), :]
    for i in range(batch_size):
        charge = bias + np.sum(np.multiply(rand_batch[i, 1:], W[1:]))
        sigmoid = np.divide(1, (1 + e.exp(-charge)))
        predict = 1 if sigmoid > 0.5 else 0
        if predict == rand_batch[i,0]:
            accuracy += 1
        else:
            Error = predict - rand_batch[i, 0]
            W_t = W
            X_t = np.concatenate(([1], rand_batch[i, 1:]))
            W_t = np.multiply(learning_rate, np.multiply(Error, X_t))
            W = np.subtract(W, W_t)
    print("Accuracy: ", (float(accuracy)/batch_size))
    classificationAccuracy.append(float(accuracy)/batch_size)
    errors.append(1-(float(accuracy)/batch_size))
    epochArray.append(x+1)


#step activation - currently commented out as sigmoid activation function proved to be a more accurate and viable approach
# for x in range(epochs):
#     bestW = W
#     accuracy = 0
#     rand_batch = data[np.random.choice(data.shape[0], batch_size, replace=False), :]
#     for i in range(batch_size):
#         charge = W[0] + np.sum(np.multiply(rand_batch[i , 1:], W[1:]))
#         predict = 1 if charge > 0 else 0
#         if predict == rand_batch[i,0]:
#             # print("You got it correct!")
#             accuracy += 1
#         else:
#             Error = predict - rand_batch[i,0]
#             W_t = W
#             X_t = np.concatenate(([1], rand_batch[i, 1:]))
#             W_t = np.multiply(learning_rate, np.multiply(Error, X_t))
#             W = np.subtract(W, W_t)
#             # print("Error: %f charge: %f predict: %f Actual: %f "%(Error, charge, predict, rand_batch[i][0]))
#     accuracyArr.append((float(accuracy))/batch_size)
#     print("Accuracy: %f"%((float(accuracy))/batch_size))


#validate that the model's weights are accurate on test datapoints
def validation():
    validationAccuracy = 0
    for i in range(len(test)):
        charge = bias + np.sum(np.multiply(test[i, 1:], bestW[1:]))
        predict = 1 if charge > 0 else 0
    
        if predict == test[i][0]:
            validationAccuracy += 1

    print("Validation accuracy: %f" %(float(validationAccuracy)/len(test)))

validation()

data_out_error = []
for x in range(len(errors)):
    data_out_error.append([errors[x], epochArray[x]])
out_error = pd.DataFrame(data_out_error)
# out_error.to_csv('Error_vs_Epoch.csv')

data_out_accuracy = []
for x in range(len(classificationAccuracy)):
    data_out_accuracy.append([classificationAccuracy[x], epochArray[x]])
out_accuracy = pd.DataFrame(data_out_accuracy)
# out_accuracy.to_csv('Accuracy_vs_Epoch.csv')