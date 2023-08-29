#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:17:24 2022

@author: jankiz
"""

import os
import shutil

SRC_DATA_FOLDER = './images_v2/ALL/'
DST_DATA_FOLDER = './to_copy/'

IMAGE_LIST_FILE = './annotations_name.txt'

image_list_file = open(IMAGE_LIST_FILE, 'r')
count = 0
 
while True:
    count += 1
 
    # Get next line from file
    line = image_list_file.readline()
 
    # if line is empty
    # end of file is reached
    if not line:
        break

    print("Line{}: {}".format(count, line.strip()))
    shutil.copy(SRC_DATA_FOLDER + line.strip(), DST_DATA_FOLDER + line.strip())
 
image_list_file.close()