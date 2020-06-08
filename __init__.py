# coding='gbk'
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal
import time
import to_raster
import ogr, os, osr
from tqdm import tqdm
import datetime
from scipy import stats, linalg
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
import imageio
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl
import multiprocessing
from multiprocessing.pool import ThreadPool as TPool
import copy_reg
import types
from scipy.stats import gamma as gam
import math
import copy
import scipy
import sklearn
import ternary
import random
import h5py
import shutil

this_root = 'D:\\project_phenology\\'
data_root = 'D:\\project_phenology\\data\\'
results_root = 'D:\\project_phenology\\results\\'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from LY_Tools import *
from HANTS import *
T = Tools()
def sleep(t=1):
    time.sleep(t)