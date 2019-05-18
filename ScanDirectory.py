"""
Scans your desired directory for images and lists half of them to train.txt and half to val.txt
"""

import os

trainpath = '.../train.txt'
valpath = '.../val.txt'
wetpath = '***WETPATH***'
drypath = '***DRYPATH***'
output = open(trainpath, 'w')
imgcount = 0

# Lists images in the wet folder to both val.txt and train.txt
with os.scandir(wetpath) as entries:
    for entry in entries:
        if entry.is_file():
            if imgcount <= 100:
                output.write(wetpath)
                output.write(entry.name)
                output.write(" 1\n")
            else:
                output = open(valpath, 'w')
                output.write(drypath)
                output.write(entry.name)
                output.write(" 0\n")

output = open(trainpath, 'w')
imgcount = 0

# Lists images in the dry folder to both val.txt and train.txt
with os.scandir(drypath) as entries:
    for entry in entries:
        if entry.is_file():
            if imgcount <= 100:
                output.write(wetpath)
                output.write(entry.name)
                output.write(" 1\n")
            else:
                output = open(valpath, 'w')
                output.write(drypath)
                output.write(entry.name)
                output.write(" 0\n")