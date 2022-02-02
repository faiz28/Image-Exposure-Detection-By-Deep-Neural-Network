import csv
import os
DIR = './train/'


def model_perameter():
    file = open('model_perameter.csv')
    csvreader = csv.reader(file)
    rows = []
    count = 0
    for row in csvreader:
        if count == 0:
            count = 1
            continue
        rows.append(row)

    return rows


if __name__ == '__main__':
    perameter = model_perameter()

    for model_layer in perameter:
        
        for i in range(int(model_layer[3])):
            print("hello")
