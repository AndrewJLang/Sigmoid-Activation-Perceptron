#Andrew Lang
#Histogram processing

import cv2
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

color = ('r', 'g', 'b')
# Specify datafile that histograms will be written to
dataFile = open("./dataFile.txt", "w")

# Pulls the r,g,b values from each image and writes into text file
# Includes if they are an aurora or not - see auroraStatus (1 = yes, 0 = no)
def imagePixelColors(fileLocation):
	try:
		for fileName in os.listdir("./" + fileLocation + "/"):
			auroraStatus = 0
			if (fileLocation == "./Aurora/"):
				auroraStatus = 1
			else:
				auroraStatus = 0
			img = cv2.imread("./" + fileLocation + "/" + fileName)
			hist = []
			for i,col in enumerate(color):
				series = cv2.calcHist([img],[i],None,[256],[0,256])
				hist.extend(series)
			c = ' '
			dataFile.write(f'{auroraStatus}: {c.join(map(str, hist))}\n')
			
	except FileNotFoundError:
		print("Improper file location")

# Method call is commented out so no accidental writing to a file is done
# imagePixelColors("./Aurora/")
# imagePixelColors("./No Aurora/")