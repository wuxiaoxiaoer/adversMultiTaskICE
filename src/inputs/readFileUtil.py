# encoding: utf-8
import os
import numpy as np
import re

def readFile(dir):
    list = [];
    # curdir = os.getcwd()
    # parent_dir = os.path.dirname(curdir)
    file = open(dir, encoding='utf-8')
    i = 1
    while 1:
        line = file.readline()
        mat = re.compile(r'\t')
        split_list = mat.split(line)
        list.append(split_list)
        if not line:
            break
    # for l in list:
    #     print(l)
    return list