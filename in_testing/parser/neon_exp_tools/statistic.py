import os

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import torch
from astropy.modeling import fitting, models

from in_testing.parser.neon_exp_tools.set_opts import arch_node_id_list_dict
from hmquant.ptq.quantization.histogram_calib import HistogramBase
from hmquant.ptq.sequencer_module import Sequencer


def cal_std(sequencer_raw: Sequencer, args):
    """绘制RepVGG Block结构的weight对比和activation对比(!!!一定要在两步校准后才能使用!!!)

    Args:
        sequencer_raw (Sequencer): 浮点序列模型
        args (_type_): 配置参数文件字典
    """
    if args.arch in arch_node_id_list_dict.keys():
        rep_node_id_list = arch_node_id_list_dict[args.arch]
    else:
        print("Wrong Arch")
        exit()
    print(args.arch)
    print("node_id\tcenter-std\toutside-std")
    for name, node in sequencer_raw.nodes.items():
        if name in rep_node_id_list:  
            conv_weight_raw = node.op.weight
            weight_center = conv_weight_raw.data[:, :, 1, 1].view(conv_weight_raw.data.shape[0], conv_weight_raw.data.shape[1], 1, 1)
            weight_outside = conv_weight_raw.data.view(conv_weight_raw.data.shape[0], conv_weight_raw.data.shape[1], -1)
            weight_outside = torch.concat([weight_outside[:, :, :4], weight_outside[:, :, 5:]], dim=-1)
            std_center = torch.std(weight_center)
            std_outside = torch.std(weight_outside)
            print("{}\t{}\t{}".format(name, std_center, std_outside))


def safe_entropy_numpy(ref_p, can_q):
    """ours calculation of kl-div"""
    p_sum = np.sum(ref_p)
    q_sum = np.sum(can_q)
    ref_p = ref_p.astype(np.float64)
    can_q = can_q.astype(np.float64)
    mask = (ref_p != 0) & (can_q != 0)
    # mask = ref_p != 0
    tmp_sum1 = np.sum(ref_p[mask] * (np.log(p_sum * ref_p[mask])))
    tmp_sum2 = np.sum(ref_p[mask] * (np.log(q_sum * can_q[mask])))
    return (tmp_sum1 - tmp_sum2) / p_sum


def fit_gaussian1d(data, bins, key, percent=0.8, plot=False):
    """拟合高斯分布

    Args:
        data (np.adarray): 1D数组，被统计的数据。
        bins (int): 直方图内的条数。
        key (string): Conv的node编号
        percent (float, optional): 与直方图峰条中心点最近的占比percent的样本将被用于拟合高斯分布. Defaults to 0.8.
        plot (bool): 是否需要将拟合得到的高斯曲线与data的直方图画在一起看一下。

    Returns:
        _type_: _description_
    """
    hx, xedge = np.histogram(data,bins)
    
    top_idx = np.where(hx == np.max(hx))
    top_left_edge = xedge[top_idx[0]]
    top_right_edge = xedge[top_idx[0] + 1]
    top_mid = (top_left_edge + top_right_edge) / 2.0
    
    dist_to_topmid_array = np.abs(data - top_mid)
    sorte_dist_idx_array = np.argsort(dist_to_topmid_array)  # 根据value升序，返回对应元素的索引组成数组
    top_num = int(len(data) * percent)
    nearest_elements_array = data[sorte_dist_idx_array[:top_num].tolist()]
    
    hx_new, xedge_new = np.histogram(data, int(bins * percent))
    
    xedge_new = (xedge_new[1:]+xedge_new[:-1])/2

    g_init = models.Gaussian1D(amplitude=np.max(hx_new), mean=np.mean(nearest_elements_array), stddev=np.std(nearest_elements_array))
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, xedge_new, hx_new)
    
    
    if plot:
        plt.figure(figsize=(32,16),dpi=300)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.hist(data, bins=bins, range=(data.min(), data.max()))
        ax.plot(color="tab:blue")
        plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
        
        times = 3.0 
        x = np.linspace(g.mean.value - times * g.stddev.value, g.mean.value + times * g.stddev.value, int(bins * percent))
        y = g(x)
        ax.plot(x, y, color="tab:red")
        plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
        
        if not os.path.exists("./log"):
            os.mkdir("log")
                
        plt.tight_layout()
        plt.savefig("./log/gaussian1d-{}.png".format(key))
        plt.close()
    
    return g.mean.value, g.stddev.value, g


def simple_guassian(data, bins, key, percent=0.8, plot=True):
    """简单的高斯分布拟合方法。

    Args:
         data (np.adarray): 1D数组，被统计的数据。
        bins (int): 直方图内的条数。
        key (string): Conv的node编号
        percent (float, optional): 与直方图峰条中心点最近的占比percent的样本将被用于拟合高斯分布. Defaults to 0.8.
        plot (bool): 是否需要将拟合得到的高斯曲线与data的直方图画在一起看一下。 Defaults to True.

    Returns:
        _type_: _description_
    """
    hx, xedge = np.histogram(data,bins)
    
    top_idx = np.where(hx == np.max(hx))
    top_left_edge = xedge[top_idx[0]]
    top_right_edge = xedge[top_idx[0] + 1]
    top_mid = (top_left_edge + top_right_edge) / 2.0
    
    dist_to_topmid_array = np.abs(data - top_mid)
    sorte_dist_idx_array = np.argsort(dist_to_topmid_array)  # 根据value升序，返回对应元素的索引组成数组
    top_num = int(len(data) * percent)
    nearest_elements_array = data[sorte_dist_idx_array[:top_num].tolist()]
 
    mu = np.mean(nearest_elements_array)
    sigma = np.std(nearest_elements_array)
    if plot: 
        plt.figure(figsize=(32, 16), dpi=300)
        fig, ax1 = plt.subplots()
        ax1.hist(data, bins, color='tab:blue')
        plt.yscale('log')
        ax1.plot()
        
        ax2 = ax1.twinx()
        new_hx, new_xedge = np.histogram(nearest_elements_array, int(bins * percent)) 
        y = norm.pdf(new_xedge, mu, sigma)#拟合一条最佳正态分布曲线y 
        ax2.plot(new_xedge, y, color="tab:red") #绘制y的曲线
        
        if not os.path.exists("./log"):
            os.mkdir('log')
        plt.tight_layout()
        plt.savefig("./log/simplegaussian1d-{}.png".format(key))
        plt.close()
    return mu, sigma