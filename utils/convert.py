import csv
from imagenet_classes import IMAGENET_CLASSES

if False:
     LABELS = list(IMAGENET2012_CLASSES.values())
     for i in range(0, 1000):
          line = '\"' + str(i) + '\": \"' + LABELS[i] + '\",'
          print(line)
if True:
     with open('caltech256_class.csv', mode ='r') as file:    
          csvFile = csv.DictReader(file)
          for lines in csvFile: 
               line = '\"' + str(int(lines['Class Index'])-1) + '\": \"' + lines['Class'] + '\",'
               print(line)