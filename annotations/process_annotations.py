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

tree = ET.parse('./ossn_annotations_v5.xml')
root = tree.getroot()

IMAGE_TAG = 'image'
BOX_TAG = 'box'
LABEL_TAG = 'label'

IMG_BASE_DIR = './images_v5/'

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
        img = cv2.imread(IMG_BASE_DIR + 'ALL/' + main_tag.attrib[NAME_ATTR])
        # print(IMG_BASE_DIR + 'ALL/' + main_tag.attrib[NAME_ATTR])
        # print(img)
        for idx, sub_tag in enumerate(main_tag):
            if sub_tag.tag == BOX_TAG:
                # print(type(float(sub_tag.attrib[TOP_LEFT_X_ATTR])))
                """print(float(sub_tag.attrib[TOP_LEFT_X_ATTR]),
                        float(sub_tag.attrib[TOP_LEFT_Y_ATTR]),
                        float(sub_tag.attrib[BOTTOM_RIGHT_X_ATTR]),
                        float(sub_tag.attrib[BOTTOM_RIGHT_Y_ATTR]))"""
                cropped_img = img[
                        int(float(sub_tag.attrib[TOP_LEFT_Y_ATTR])):int(float(sub_tag.attrib[BOTTOM_RIGHT_Y_ATTR])),
                        int(float(sub_tag.attrib[TOP_LEFT_X_ATTR])):int(float(sub_tag.attrib[BOTTOM_RIGHT_X_ATTR]))
                        ]
                # cv2.imshow('cropped image', cropped_img)
                if not os.path.exists(IMG_BASE_DIR + sub_tag.attrib[LABEL_TAG]):
                    os.makedirs(IMG_BASE_DIR + sub_tag.attrib[LABEL_TAG])
                
                cv2.imwrite(IMG_BASE_DIR + sub_tag.attrib[LABEL_TAG] + '/' + main_tag.attrib[NAME_ATTR] + '_' + str(idx) + '.jpg', cropped_img)
            # print(sub_tag.tag, sub_tag.attrib)