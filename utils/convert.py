import csv
from imagenet_classes import IMAGENET2012_CLASSES

if True:
     LABELS = list(IMAGENET2012_CLASSES.values())
     for i in range(0, 1000):
          line = '\"' + str(i) + '\": \"' + LABELS[i] + '\",'
          print(line)
if False:
     with open('imagenet_superclass.csv', mode ='r') as file:    
          csvFile = csv.DictReader(file)
          for lines in csvFile: 
               line = '\"' + lines['Class Index'] + '\": \"' + lines['Class Name'] + '\",'
               print(line)