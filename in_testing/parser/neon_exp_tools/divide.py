import os
from matplotlib import pyplot as plt
import numpy as np


def cal_granularity(data, bit_num=8, signed=True):
    min_weight = np.min(data)
    max_weight = np.max(data)
    if signed:
        bit_num -= 1
    return (max_weight - min_weight) / (2.0 ** bit_num)


def divide_weight(data, key, bins=1024, bit_num=8, signed=True, plot=True):
    granularity = cal_granularity(data, bit_num, signed)
    data_coarse = np.round(np.true_divide(data, granularity)) * granularity
    data_offset = data - data_coarse
    if plot:
        fig = plt.figure(figsize=(48,16),dpi=300)

        data_plt = plt.subplot(1, 3, 1)
        data_plt.hist(data, bins=bins, range=(data.min(), data.max()))
        data_plt.set_title('origin-{}'.format(key))
        plt.yscale('log')
        data_coarse_plt = plt.subplot(1, 3, 2)
        data_coarse_plt.hist(data_coarse, bins=bins, range=(data_coarse.min(), data_coarse.max()))
        data_coarse_plt.set_title('coarse-{}'.format(key))
        plt.yscale('log') 
        data_offset_plt = plt.subplot(1, 3, 3)
        data_offset_plt.hist(data_offset, bins=bins, range=(data_offset.min(), data_offset.max()))
        data_offset_plt.set_title('offset-{}'.format(key))
        if not os.path.exists("./log"):
            os.mkdir("log")
        
        plt.tight_layout()
        plt.savefig("./log/divide-{}.png".format(key))
        plt.close()
    return data_coarse, data_offset, granularity