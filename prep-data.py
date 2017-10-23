# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from numpy import genfromtxt
import csv
import time
def extend(x, y, path, extended_folder, names):
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
    np.savetxt(extended_folder + '/' + names, z, delimiter=',') # Write to the folder

def merge(path, name):
    name_b = name.split('-')[0]
    name_c = name.split('-')[1]
    if name_b == 'eyes':
        p1 = '%s/eyes-%s'%(path, name_c)
        part1 = genfromtxt(p1, delimiter=',')
        m, n = part1.shape
        p2 = '%s/kinect-%s' % (path, name_c)
        if os.path.exists(p2):
            part2 = genfromtxt(p2, delimiter=',')
            part2 = part2[:, 1:]
        else:
            part2 = np.zeros((m, 27))  # If the file does not exist, fill it with 0
        p3 = '%s/audio-%s' % (path, name_c)
        if os.path.exists(p3):
            part3 = genfromtxt(p3, delimiter=',')
            part3 = part3[:, 1:]
        else:
            part3 = np.zeros((m, 36)) # If the file does not exist, fill it with 0
        p4 = '%s/face_nn-%s' % (path, name_c)
        if os.path.exists(p4):
            part4 = genfromtxt(p4, delimiter=',')
            part4 = part4[:, 1:]
        else:
            part4 = np.zeros((m, 100)) # If the file does not exist, fill it with 0
        parts = np.hstack((part1, part2, part3, part4))  # Horizontal connection
        np.savetxt(input_folder + '/' + name_c, parts, delimiter=',') # Write to the folder

if __name__ == '__main__':
    path = input('Input your data path, should be like : ../data/train \nPlease: ')
    path = str(path.replace('\\', '/'))
    if '/data/train' not in path:
        print('The path is wrong, please cheack it again.')
        time.sleep(100)
        exit()
    else:
        os.makedirs(path.replace('train', 'extended'))  # Create a folder
        extended_folder = path.replace('train', 'extended')
        os.makedirs(path.replace('train', 'input')) # Create a folder
        input_folder = path.replace('train', 'input')
        files = os.listdir(path) #得到文件夹下的所有文件名称
        for file in files: #Read the file
            part = os.listdir(path + '/' + file)
            for single in part:  # os.listdir('.')遍历文件夹内的每个文件名，并返回一个包含文件名的list
                newname = file + '-' + single
                os.rename(path + '/' + file + '/' + single, path + '/' + file + '/' + newname)
        print('Start extending....')
        labels = os.listdir(path + '/' + 'labels')
        for label in labels: # Read the file
            read_labels = genfromtxt(path + '/' + 'labels/' + label, delimiter=',', skip_header=1)
            label = label.split('-')[1]
            for file in ['kinect', 'face_nn', 'audio', 'eyes']:
                part = os.listdir(path + '/' + file)
                for names in part:
                    name = names.split('-')[1]
                    if label == name:
                        folder = '%s/%s/%s'%(path, file, names)
                        read_others = genfromtxt(folder, delimiter=',', skip_header=1)
                        extend(read_labels, read_others, path, extended_folder, names) # All the input file to expand

        print('Finished. Start merging....')
        files = os.listdir(extended_folder)
        for name in files:
            merge(extended_folder, name) # All the input files are integrated

        print('Finished. Start the label transformation....')
        output_folder = path.replace('train', 'labels')
        os.makedirs(output_folder)
        files = os.listdir(path + '/labels')
        for file in files:   # Read the file
            name = file.split('-')[1]
            labels = genfromtxt('%s/labels/%s' % (path, file), delimiter=',', skip_header=1)
            label = np.dot(labels[:, 1:7], np.arange(1, 7).T)  # Target classification mapping
            label = np.vstack((label, labels[:, -1])).T
            np.savetxt(output_folder + '/' + name, label, delimiter=',')  # Write to the folder

        print('Finished. Start integrating to a file.....')
        files = os.listdir(input_folder)
        files_1 = os.listdir(output_folder)
        for i, file in enumerate(files):            # All the input and output files are integrated to a file
            if i == 0:
                ytext = genfromtxt(output_folder + '/' + file, delimiter=',')
                xtext = genfromtxt(input_folder + '/' + file, delimiter=',')
            else:
                ytext_1 = genfromtxt(output_folder + '/' + file, delimiter=',')
                xtext_1 = genfromtxt(input_folder + '/' + file, delimiter=',')
                xtext = np.vstack((xtext, xtext_1))
                ytext = np.vstack((ytext, ytext_1))
        np.savetxt('%s/input.csv'%path.split('train')[0], xtext, delimiter=',')  # Write to the folder
        np.savetxt('%s/output.csv'%path.split('train')[0], ytext, delimiter=',')  # Write to the folder
        print('Finished')
