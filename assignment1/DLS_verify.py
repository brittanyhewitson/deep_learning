import cv2
import numpy as np
import argparse
import sklearn
import tensorflow as tf
import keras
from matplotlib import pyplot as plt 
import time
import os

f = open('output.txt', 'w')
f.write(time.strftime("%H:%M:%S\n"))
f.write(time.strftime("%d/%m/%Y\n"))
f.write(os.getcwd())
f.close()

