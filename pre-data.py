# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
from numpy import genfromtxt
import csv

def extend(x, y, path, extended_folder, names):
    z = []
    nn = 0
    n = len(y)
    for m in range(0,len(x)):
        if m<1 :
            z.append(y[nn,:])
            nn += 1
        if x[m,0]> y[nn-1,0]:
            if x[m,0]> y[nn,0]:
                z.append(y[nn,:])
                if nn < n-1:
                    nn += 1
            else:
                z.append(y[nn-1,:])
        else:
            z.append(y[nn-1,:])
    np.savetxt(extended_folder + '/' + names, z, delimiter=',')

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
            part2 = np.zeros((m, 27))
        p3 = '%s/audio-%s' % (path, name_c)
        if os.path.exists(p3):
            part3 = genfromtxt(p3, delimiter=',')
            part3 = part3[:, 1:]
        else:
            part3 = np.zeros((m, 36))
        p4 = '%s/face_nn-%s' % (path, name_c)
        if os.path.exists(p4):
            part4 = genfromtxt(p4, delimiter=',')
            part4 = part4[:, 1:]
        else:
            part4 = np.zeros((m, 100))
        parts = np.hstack((part1, part2, part3, part4))
        np.savetxt(input_folder + '/' + name_c, parts, delimiter=',')

if __name__ == '__main__':

    path = "C:/Users/sheng/Desktop/data/train" # Change to your data file, shoule be like : ../data/train
    os.makedirs(path.replace('train', 'extended'))
    extended_folder = path.replace('train', 'extended')
    os.makedirs(path.replace('train', 'input'))
    input_folder = path.replace('train', 'input')
    files = os.listdir(path) #得到文件夹下的所有文件名称
    for file in files: #遍历文件夹
        part = os.listdir(path + '/' + file)
        for single in part:  # os.listdir('.')遍历文件夹内的每个文件名，并返回一个包含文件名的list
            newname = file + '-' + single
            os.rename(path + '/' + file + '/' + single, path + '/' + file + '/' + newname)
    print('Start extending....')
    labels = os.listdir(path + '/' + 'labels')
    for label in labels:
        read_labels = genfromtxt(path + '/' + 'labels/' + label, delimiter=',', skip_header=1)
        label = label.split('-')[1]
        for file in ['kinect', 'face_nn', 'audio', 'eyes']:
            part = os.listdir(path + '/' + file)
            for names in part:
                name = names.split('-')[1]
                if label == name:
                    folder = '%s/%s/%s'%(path, file, names)
                    read_others = genfromtxt(folder, delimiter=',', skip_header=1)
                    extend(read_labels, read_others, path, extended_folder, names)

    print('Finished. Start merging....')
    files = os.listdir(extended_folder)
    for name in files:
        merge(extended_folder, name)

    print('Finished. Start the label transformation....')
    output_folder = path.replace('train', 'labels')
    os.makedirs(output_folder)
    files = os.listdir(path + '/labels')
    for file in files:
        name = file.split('-')[1]
        labels = genfromtxt('%s/labels/%s' % (path, file), delimiter=',', skip_header=1)
        label = np.dot(labels[:, 1:7], np.arange(1, 7).T)
        label = np.vstack((label, labels[:, -1])).T
        np.savetxt(output_folder + '/' + name, label, delimiter=',')

    print('Finished. Start integrating to a file.....')
    files = os.listdir(input_folder)
    files_1 = os.listdir(output_folder)
    for i, file in enumerate(files):
        if i == 0:
            ytext = genfromtxt(output_folder + '/' + file, delimiter=',')
            xtext = genfromtxt(input_folder + '/' + file, delimiter=',')
        else:
            ytext_1 = genfromtxt(output_folder + '/' + file, delimiter=',')
            xtext_1 = genfromtxt(input_folder + '/' + file, delimiter=',')
            xtext = np.vstack((xtext, xtext_1))
            ytext = np.vstack((ytext, ytext_1))
    np.savetxt('%s/input.csv'%path.split('train')[0], xtext, delimiter=',')
    np.savetxt('%s/output.csv'%path.split('train')[0], ytext, delimiter=',')
    print('Finished')