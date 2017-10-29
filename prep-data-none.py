# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from numpy import genfromtxt
import csv
import re
import time
'''
If your input path is .../data/train, when you run the code(pre-data) successfully, 
in .../data folder you will get the results, they are folders:
Extended: All the input file to expand
Input: All the input files are integrated
Labels: All output files
Input.csv:  All the input files are integrated to a file
Output.csv: All the output files are integrated to a file
#  You can also explain this more in detail

'''
def replace(s):
    t = []
    for i in range(0, len(s[:,0])):
        t.append(np.sum(s[i, :]))
        if t[i] == 0:
            s[i, :] = np.full((1, len(s[i, :])), None)
    return s
def extend(x, y, file, extended_folder, names):
    z = []
    nn = 0
    n = len(y)
    for m in range(0,len(x)):
        if m<1 :
            z.append(y[nn,:])
            nn += 1
        elif x[m,0]> y[nn-1,0]:
            if x[m,0]> y[nn,0]:
                z.append(y[nn,:])
                if nn < n-1:
                    nn += 1
            else:
                z.append(y[nn-1,:])
        else:
            z.append(y[nn-1,:])
    np.savetxt(extended_folder + '/' + names.replace('.csv', '-%s.csv'%file), z, fmt='%s', delimiter=',') # Write to the folder

def merge(path, name):
    name_c = name.split('-')[0]
    name_b = name.split('-')[1]
    if name_b == 'eyes.csv':
        p1 = '%s/%s-eyes.csv'%(path, name_c)
        part1 = genfromtxt(p1, delimiter=',')
        part1 = replace(part1)
        m, n = part1.shape
        p2 = '%s/%s-kinect.csv' % (path, name_c)
        if os.path.exists(p2):
            part2 = genfromtxt(p2, delimiter=',')
            part2 = replace(part2[:, 1:])
        else:
            part2 = np.full((m, 27), None)  # If the file does not exist, fill it with 0
        p3 = '%s/%s-audio.csv' % (path, name_c)
        if os.path.exists(p3):
            part3 = genfromtxt(p3, delimiter=',')
            part3 = replace(part3[:, 1:])
        else:
            part3 = np.full((m, 36), None) # If the file does not exist, fill it with 0
        p4 = '%s/%s-face_nn.csv' % (path, name_c)
        if os.path.exists(p4):
            part4 = genfromtxt(p4, delimiter=',')
            part4 = replace(part4[:, 1:])
        else:
            part4 = np.full((m, 100), None) # If the file does not exist, fill it with 0
        parts = np.hstack((part1, part2, part3, part4))  # Horizontal connection
        np.savetxt(input_folder + '/%s.csv'%name_c, parts, fmt='%s', delimiter=',') # Write to the folder
if __name__ == '__main__':
    path = input('Input your data path, should be like : ../data/train or ../data/test \nPlease: ')
    path = str(path.replace('\\', '/'))
    if ('/data/train' not in path) and ('/data/test' not in path):
        print('The path is wrong, please cheack it again.')
        time.sleep(100)
        exit()
    else:
        s = re.findall('(.*?/data/)(.*)', path)[0]
        input_csv = '%sinput-%s.csv' % (s[0], s[1])
        output_csv = '%soutput-%s.csv' % (s[0], s[1])
        if '/data/train' in path:
            extended = 'extended-train'
            input = 'input-train'
            output = 'labels-train'
            label = path + '/' + 'labels'
        else:
            extended = 'extended-test'
            input = 'input-test'
            output = 'labels-test'
            label = path + '/' + 'prediction'
        os.makedirs(path.replace(s[1], extended))  # Create a folder
        extended_folder = path.replace(s[1], extended)
        os.makedirs(path.replace(s[1], input)) # Create a folder
        input_folder = path.replace(s[1], input)
        files = os.listdir(path) #得到文件夹下的所有文件名称
        print('Start extending....')
        labels = os.listdir(label)
        for label_name in labels: # Read the file
            read_labels = genfromtxt(label + '/' + label_name, delimiter=',', skip_header=1)
            for file in ['kinect', 'face_nn', 'audio', 'eyes']:
                part = os.listdir(path + '/' + file)
                for name in part:
                    if label_name == name:
                        folder = '%s/%s/%s'%(path, file, name)
                        read_others = genfromtxt(folder, delimiter=',', skip_header=1)
                        extend(read_labels, read_others, file, extended_folder, name) # All the input file to expand

        print('Finished. Start merging....')
        files = os.listdir(extended_folder)
        for name in files:
            merge(extended_folder, name) # All the input files are integrated

        print('Finished. Start the label transformation....')
        output_folder = path.replace(s[1], output)
        os.makedirs(output_folder)
        files = os.listdir(label)
        for file in files:   # Read the file
            label_read = genfromtxt('%s/%s' % (label, file), delimiter=',', skip_header=1)
            label_transform = np.dot(label_read[:, 1:7], np.arange(1, 7).T)  # Target classification mapping
            label_transform = np.vstack((label_transform, label_read[:, -1])).T
            np.savetxt(output_folder + '/' + file, label_transform, fmt='%s', delimiter=',')  # Write to the folder

        print('Finished. Start integrating to a file.....')
        files = os.listdir(input_folder)
        for i, file in enumerate(files):            # All the input and output files are integrated to a file
            if i == 0:
                ytext = genfromtxt(output_folder + '/' + file, delimiter=',')
                xtext = genfromtxt(input_folder + '/' + file, delimiter=',')
            else:
                ytext_1 = genfromtxt(output_folder + '/' + file, delimiter=',')
                xtext_1 = genfromtxt(input_folder + '/' + file, delimiter=',')
                xtext = np.vstack((xtext, xtext_1))
                ytext = np.vstack((ytext, ytext_1))
        np.savetxt(input_csv, xtext, fmt='%s', delimiter=',')  # Write to the folder
        np.savetxt(output_csv, ytext, fmt='%s', delimiter=',')  # Write to the folder
        print('Finished')
