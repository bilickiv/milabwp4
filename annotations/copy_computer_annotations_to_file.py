#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:17:24 2022

@author: jankiz
"""

import xml.etree.ElementTree as ET
import cv2
import os
import time

tree = ET.parse('./annotations.xml')
root = tree.getroot()

IMAGE_TAG = 'image'
BOX_TAG = 'box'
LABEL_TAG = 'label'
POLYGON_TAG = 'polygon'

IMG_BASE_DIR = './images_v2/'

COMPUTER_ANNOTATIONS_PATH = './annotation-images/'

NAME_ATTR = 'name'
TOP_LEFT_X_ATTR = 'xtl'
TOP_LEFT_Y_ATTR = 'ytl'
BOTTOM_RIGHT_X_ATTR = 'xbr'
BOTTOM_RIGHT_Y_ATTR = 'ybr'

used_labels = set()

available_images = os.listdir(IMG_BASE_DIR)
available_images.sort()

ctr = 0

for main_tag in root:
    if main_tag.tag == IMAGE_TAG:
        # print(main_tag.tag, main_tag.attrib)
        # img = cv2.imread(IMG_BASE_DIR + 'ALL/' + main_tag.attrib[NAME_ATTR])
        # print(IMG_BASE_DIR + 'ALL/' + main_tag.attrib[NAME_ATTR])
        # print(img)
        
        for sub_tag in list(main_tag):
            print(list(main_tag), sub_tag)
            main_tag.remove(sub_tag)
            print(list(main_tag))
        
        print('----------------------')
        try:
            xml_str_file = open(COMPUTER_ANNOTATIONS_PATH + main_tag.attrib[NAME_ATTR] + '.xml')
            while True:
            
                # Get next line from file
                line = xml_str_file.readline()
                
                # if line is empty
                # end of file is reached
                if not line:
                    break

                line = line.strip()


                xml_content = ET.fromstring(line)
                main_tag.append(xml_content)
            
            xml_str_file.close()
        except:
            print('file is missing: ' + COMPUTER_ANNOTATIONS_PATH + main_tag.attrib[NAME_ATTR] + '.xml')
        
tree.write('./annotations_cell_polygons.xml')


        
