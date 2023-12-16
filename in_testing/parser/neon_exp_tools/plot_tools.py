import os
import numpy as np
import torch
from hmquant.ptq.sequencer_module import Sequencer
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from in_testing.parser.neon_exp_tools.set_opts import arch_node_id_list_dict
import torch.nn.functional as F

DPI_VALUE = 400
FONT_SIZE = 9
FIG_SIZE = 2
FONT_TYPE = "Times New Roman"

def plot_hist(kernel_data: torch.Tensor, kernel_type: str, node_id: int):
    if not os.path.exists("./log"):
        os.mkdir("log")
    if not os.path.exists("./log/fpweights-{}".format(node_id)):
        os.mkdir("./log/fpweights-{}".format(node_id))
    type_dict = {
        "0": "Identity",
        "1": "1x1",
        "3": "3x3",
        "4": "Converted"
    }
    kernel_data = torch.flatten(kernel_data).cpu().detach().numpy()
    font = fm.FontProperties(size=FONT_SIZE, family=FONT_TYPE)
    figure = plt.figure(figsize=(1.2, 0.6))
    plt.hist(kernel_data, bins=1024, range=(kernel_data.min(), kernel_data.max()), linewidth=0.5)
    plt.xticks(fontproperties=font)
    plt.yticks(fontproperties=font)
    plt.title('Kernel-{}'.format(type_dict[kernel_type]), fontproperties=font)
    plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
    # plt.ylabel("Frequency", fontproperties=font)
    plt.savefig("./log/fpweights-{}/{}-{}-fpA.pdf".format(node_id, node_id, type_dict[kernel_type]), dpi=DPI_VALUE - 100, bbox_inches='tight')
    plt.close()
    

def plot_lines(step_data: list, value_data: list, arch: str, layer_idx: int):
    if not os.path.exists("./log"):
        os.mkdir("log")
    if not os.path.exists("./log/lines"):
        os.mkdir("./log/lines")
    figure = plt.figure(figsize=(FIG_SIZE, FIG_SIZE))
    font = fm.FontProperties(size=FONT_SIZE, family=FONT_TYPE)
    plt.title("Accuracy in {}".format(arch), fontproperties=font)
    # plt.xlabel("alpha", fontproperties=font)
    plt.ylabel("Accuracy(%)", fontproperties=font)
    plt.plot(step_data, value_data)
    plt.xticks(step_data[::1], fontproperties=font)
    plt.yticks(fontproperties=font)
    plt.savefig("./log/lines/{}-wei-{}-lines.png".format(arch, layer_idx), dpi=DPI_VALUE, bbox_inches='tight')
    plt.close()


def plot_weight_activation(sequencer: Sequencer, sequencer_raw: Sequencer, args):
    """绘制RepVGG Block结构的weight对比和activation对比(!!!一定要在两步校准后才能使用!!!)

    Args:
        sequencer (Sequencer): 替换PTQ算子并校准过后的序列模型
        sequencer_raw (Sequencer): 浮点序列模型
        args (_type_): 配置参数文件字典
    """
    if args.arch in arch_node_id_list_dict.keys():
        rep_node_id_list = arch_node_id_list_dict[args.arch]
    else:
        print("Wrong Arch")
        exit()
    bins = 256
    plt_row_num, plt_col_num = 1, 2
    if not os.path.exists("./log"):
        os.mkdir("log")
    for name, node in sequencer.nodes.items():
        if name in rep_node_id_list:  
            
            conv_weight_raw = sequencer_raw.nodes[name].op.weight
            weight_raw_cpu = torch.flatten(conv_weight_raw).cpu().detach().numpy() 
            
            if not args.no_quant:   
                conv = node.op
                if args.ptq_cfg in ["RepVGGKLConfig", "RepVGGMinMaxConfig", "OnlyRepVGGKLConfig"]:
                    conv_coarse_weight_ptq, conv_fine_weight_ptq, bias_sim = conv.quant_weight_bias()
                    conv_weight_ptq = torch.add(conv_coarse_weight_ptq, conv_fine_weight_ptq, alpha=1)
                else:
                    conv_weight_ptq, bias_sim = conv.quant_weight_bias()
                weight_ptq_cpu = torch.flatten(conv_weight_ptq).cpu().detach().numpy()
                out_raw_cpu = torch.flatten(conv.raw_out).cpu().detach().numpy()
                out_ptq_cpu = torch.flatten(conv.ptq_out).cpu().detach().numpy()
            
            figure_weight,(weight_raw_plt, weight_ptq_plt) = plt.subplots(plt_row_num, plt_col_num, figsize=(32, 16), dpi=300, sharex=True, sharey=True)
            weight_raw_plt.hist(weight_raw_cpu, bins=bins, range=(weight_raw_cpu.min(), weight_raw_cpu.max()))
            weight_raw_plt.set_title('weight raw -{}'.format(name))
            weight_raw_plt.set_yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
            if not args.no_quant: 
                weight_ptq_plt.hist(weight_ptq_cpu, bins=bins, range=(weight_ptq_cpu.min(), weight_ptq_cpu.max()))
                weight_ptq_plt.set_title('weight {} -{}-MSE:{}'.format(args.ptq_cfg, name, np.mean(np.square(weight_ptq_cpu - weight_raw_cpu))))
                weight_ptq_plt.set_yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
            plt.tight_layout()
            plt.savefig("./log/weight-raw-{}-{}.png".format(args.ptq_cfg, name))
            plt.close()
            figure_activation, (out_raw_plt, out_ptq_plt) = plt.subplots(plt_row_num, plt_col_num, figsize=(32, 16), dpi=300, sharex=True, sharey=True)
            out_raw_plt.hist(out_raw_cpu, bins=bins, range=(out_raw_cpu.min(), out_raw_cpu.max()))
            out_raw_plt.set_title('activation raw -{}'.format(name))
            out_raw_plt.set_yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
            if not args.no_quant:
                out_ptq_plt.hist(out_ptq_cpu, bins=bins, range=(out_ptq_cpu.min(), out_ptq_cpu.max()))
                out_ptq_plt.set_title('activation {} -{}-L1Error:{}'.format(args.ptq_cfg, name, np.true_divide(np.mean(np.abs(out_ptq_cpu - out_raw_cpu)), np.mean(np.abs(out_raw_cpu)))))
                out_ptq_plt.set_yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
            plt.tight_layout()
            plt.savefig("./log/activation-raw-{}-{}.png".format(args.ptq_cfg, name))
            plt.close()
            
            
def plot_inference_center_8neighbors_violin_plot(sequencer_raw: Sequencer, args):
    """绘制每一层RepVGGBlock Conv 的小提琴图, 横轴左侧是8 neighbors, 右侧是center.

    Args:
        sequencer_raw (Sequencer): 加载的浮点模型
        args (_type_): 脚本运行参数
    """
    if args.arch in arch_node_id_list_dict.keys():
        rep_node_id_list = arch_node_id_list_dict[args.arch]
    else:
        print("Wrong Arch")
        exit()
    
    if not os.path.exists("./log"):
        os.mkdir("log")
    if not os.path.exists("./log/violinplot"):
        os.mkdir("./log/violinplot")
    font = fm.FontProperties(size=FONT_SIZE, family=FONT_TYPE)
    count = 1
    for name, node in sequencer_raw.nodes.items():
        if name in rep_node_id_list:  
            conv_weight_raw = node.op.weight
            weight_center = conv_weight_raw.data[:, :, 1, 1].view(conv_weight_raw.data.shape[0], conv_weight_raw.data.shape[1], 1, 1)
            weight_outside = conv_weight_raw.data.view(conv_weight_raw.data.shape[0], conv_weight_raw.data.shape[1], -1)
            weight_outside = torch.concat([weight_outside[:, :, :4], weight_outside[:, :, 5:]], dim=-1)
            figure = plt.figure(figsize=(8, 16), dpi=300)
            plt.title("Conv Layer {}".format(name))
            weight_data = [torch.flatten(weight_outside).cpu().detach().numpy(), torch.flatten(weight_center).cpu().detach().numpy()]
            violinplot = plt.violinplot(weight_data, showmeans=True)
            plt.xticks(ticks=[1, 2], labels=["8 Neighbors", "Center"])
            plt.savefig("./log/violinplot/{}-conv-violin-raw-{}-{}.png".format(args.arch, args.ptq_cfg, name))
            plt.close()
            

def plot_inference_center_8neighbors_box_plot(sequencer_raw: Sequencer, args):
    """绘制每一层RepVGGBlock Conv的箱型图, 横轴左侧是

    Args:
        sequencer_raw (Sequencer): 加载的浮点模型
        args (_type_): 脚本运行参数
    """
    if args.arch in arch_node_id_list_dict.keys():
        rep_node_id_list = arch_node_id_list_dict[args.arch]
    else:
        print("Wrong Arch")
        exit()
    
    if not os.path.exists("./log"):
        os.mkdir("log")
    if not os.path.exists("./log/boxplot"):
        os.mkdir("./log/boxplot")
    font = fm.FontProperties(size=FONT_SIZE, family=FONT_TYPE)
    count = 1
    for name, node in sequencer_raw.nodes.items():
        if name in rep_node_id_list:  
            conv_weight_raw = node.op.weight
            weight_center = conv_weight_raw.data[:, :, 1, 1].view(conv_weight_raw.data.shape[0], conv_weight_raw.data.shape[1], 1, 1)
            weight_outside = conv_weight_raw.data.view(conv_weight_raw.data.shape[0], conv_weight_raw.data.shape[1], -1)
            weight_outside = torch.concat([weight_outside[:, :, :4], weight_outside[:, :, 5:]], dim=-1)
            figure = plt.figure(figsize=(FIG_SIZE, FIG_SIZE + 4))
            plt.title("Conv Layer {}".format(count), fontproperties=font)
            weight_data = [torch.flatten(weight_outside).cpu().detach().numpy(), torch.flatten(weight_center).cpu().detach().numpy()]
            boxplot = plt.violinplot(dataset=weight_data)
            # boxplot = plt.boxplot(weight_data, flierprops={'marker': '.', 'markeredgecolor': 'blue', 'markersize': 2})
            plt.xticks(ticks=[1, 2], labels=["around", "center"], fontproperties=font)
            plt.savefig("./log/boxplot/{}-conv-vio-raw-{}-{}.pdf".format(args.arch, args.ptq_cfg, name), dpi=DPI_VALUE - 100, bbox_inches='tight')
            plt.close()
            count += 1
            

def divide_weight_channelwise(weight, w_bit: int):
    w_qmax = (1 << (w_bit - 1)) - 1
    granularity_tensor = weight.abs().amax([1, 2, 3]) / w_qmax
    granularity_tensor = granularity_tensor.view(-1, 1, 1, 1)
    weight_coarse = torch.mul(torch.true_divide(weight, granularity_tensor).add(0.5).floor_(), granularity_tensor)
    weight_fine = torch.sub(input=weight, alpha=1, other=weight_coarse)
    return weight_coarse, weight_fine, granularity_tensor


def plot_inference_center_coarse_fine_hist(sequencer_raw: Sequencer, args):
    """绘制每一层RepVGGBlock Conv的箱型图, 横轴左侧是

    Args:
        sequencer_raw (Sequencer): 加载的浮点模型
        args (_type_): 脚本运行参数
    """
    if args.arch in arch_node_id_list_dict.keys():
        rep_node_id_list = arch_node_id_list_dict[args.arch]
    else:
        print("Wrong Arch")
        exit()
    
    if not os.path.exists("./log"):
        os.mkdir("log")
    if not os.path.exists("./log/center_hist"):
        os.mkdir("./log/center_hist")
    font = fm.FontProperties(size=FONT_SIZE, family=FONT_TYPE)
    font_2 = fm.FontProperties(size=FONT_SIZE-2, family=FONT_TYPE)
    count = 1
    for name, node in sequencer_raw.nodes.items():
        if name in rep_node_id_list:  
            conv_weight_raw = node.op.weight
            weight_center = conv_weight_raw.data[:, :, 1, 1].view(conv_weight_raw.data.shape[0], conv_weight_raw.data.shape[1], 1, 1)
            weight_outside = conv_weight_raw.data.view(conv_weight_raw.data.shape[0], conv_weight_raw.data.shape[1], -1)
            weight_outside = torch.concat([weight_outside[:, :, :4], weight_outside[:, :, 5:]], dim=-1)
            figure = plt.figure(figsize=(1, 1))
            plt.title("Conv-{}".format(count), fontproperties=font)
            weight_data = torch.flatten(weight_center).cpu().detach().numpy()
            plt.hist(weight_data, bins=1024, range=(weight_data.min(), weight_data.max()), linewidth=0.5)
            plt.xticks(fontproperties=font)
            plt.yticks(fontproperties=font)
            plt.title('Kernel-{}'.format("all"), fontproperties=font)
            plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
            plt.savefig("./log/center_hist/{}-{}.pdf".format(count, "all"), dpi=DPI_VALUE-100, bbox_inches='tight')
            plt.close()
            
            weight_coarse, weight_fine, _ = divide_weight_channelwise(weight_center, 8)
            weight_mix = conv_weight_raw.data - F.pad(weight_coarse, pad=[1, 1, 1, 1], value=0)
            weight_mix = torch.flatten(weight_mix).cpu().detach().numpy()
            figure = plt.figure(figsize=(1, 1))
            plt.title("Conv-{}".format(count), fontproperties=font)
            plt.hist(weight_mix, bins=1024, range=(weight_mix.min(), weight_mix.max()), linewidth=0.5)
            plt.xticks(fontproperties=font)
            plt.yticks(fontproperties=font)
            plt.title('Kernel-{}'.format("mix"), fontproperties=font)
            plt.yscale('log')
            plt.savefig("./log/center_hist/{}-{}.pdf".format(count, "mix"), dpi=DPI_VALUE-100, bbox_inches='tight')
            weight_coarse = torch.flatten(weight_coarse).cpu().detach().numpy()
            figure = plt.figure(figsize=(1, 1))
            plt.title("Conv-{}".format(count), fontproperties=font)
            plt.hist(weight_coarse, bins=1024, range=(weight_coarse.min(), weight_coarse.max()), linewidth=0.5)
            plt.xticks(fontproperties=font)
            plt.yticks(fontproperties=font)
            plt.title('Kernel-{}'.format("coarse"), fontproperties=font)
            plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
            plt.savefig("./log/center_hist/{}-{}.pdf".format(count, "coarse"), dpi=DPI_VALUE-100, bbox_inches='tight')
            plt.close()
            weight_fine = torch.flatten(weight_fine).cpu().detach().numpy()
            figure = plt.figure(figsize=(1, 1))
            plt.title("Conv-{}".format(count), fontproperties=font)
            plt.hist(weight_fine, bins=1024, range=(weight_fine.min(), weight_fine.max()), linewidth=0.5)
            plt.xticks(fontproperties=font_2)
            plt.yticks(fontproperties=font)
            plt.title('Kernel-{}'.format("fine"), fontproperties=font)
            plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
            plt.savefig("./log/center_hist/{}-{}.pdf".format(count, "fine"), dpi=DPI_VALUE-100, bbox_inches='tight')
            plt.close()
            weight_outside = torch.flatten(weight_outside).cpu().detach().numpy()
            figure = plt.figure(figsize=(1, 1))
            plt.title("Conv-{}".format(count), fontproperties=font)
            plt.hist(weight_outside, bins=1024, range=(weight_outside.min(), weight_outside.max()), linewidth=0.5)
            plt.yscale('log')
            plt.xticks(fontproperties=font)
            plt.yticks(fontproperties=font)
            plt.title('Kernel-{}'.format("around"), fontproperties=font)
            plt.savefig("./log/center_hist/{}-{}.pdf".format(count, "around"), dpi=DPI_VALUE-100, bbox_inches='tight')
            plt.close()
            count += 1
            

def plot_inference_center_channel_coarse_fine_hist(sequencer_raw: Sequencer, args):
    """绘制每一层RepVGGBlock Conv的箱型图, 横轴左侧是

    Args:
        sequencer_raw (Sequencer): 加载的浮点模型
        args (_type_): 脚本运行参数
    """
    if args.arch in arch_node_id_list_dict.keys():
        rep_node_id_list = arch_node_id_list_dict[args.arch]
    else:
        print("Wrong Arch")
        exit()
    
    if not os.path.exists("./log"):
        os.mkdir("log")
    if not os.path.exists("./log/center_ch_hist"):
        os.mkdir("./log/center_ch_hist")
    font = fm.FontProperties(size=FONT_SIZE, family=FONT_TYPE)
    count = 1
    for name, node in sequencer_raw.nodes.items():
        if name in rep_node_id_list:
            if name in rep_node_id_list[-2]:
                conv_weight_raw = node.op.weight
                for i in range(conv_weight_raw.data.shape[0]):
                    weight_center = conv_weight_raw.data[i, :, 1, 1].view(1, conv_weight_raw.data.shape[1], 1, 1)
                    figure = plt.figure(figsize=(1, 1))
                    weight_data = torch.flatten(weight_center).cpu().detach().numpy()
                    plt.hist(weight_data, bins=1024, range=(weight_data.min(), weight_data.max()), linewidth=0.5)
                    plt.xticks(fontproperties=font)
                    plt.yticks(fontproperties=font)
                    plt.title('{}-{}'.format(i, "all"), fontproperties=font)
                    # plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
                    plt.savefig("./log/center_ch_hist/{}-ch{}-{}.png".format(count, i, "all"), dpi=DPI_VALUE-100, bbox_inches='tight')
                    plt.close()
            
                    weight_coarse, weight_fine, _ = divide_weight_channelwise(weight_center, 8)
                    weight_coarse = torch.flatten(weight_coarse).cpu().detach().numpy()
                    figure = plt.figure(figsize=(1, 1))
                    plt.hist(weight_coarse, bins=1024, range=(weight_coarse.min(), weight_coarse.max()), linewidth=0.5)
                    plt.xticks(fontproperties=font)
                    plt.yticks(fontproperties=font)
                    plt.title('{}-{}'.format(i, "coarse"), fontproperties=font)
                    # plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
                    plt.savefig("./log/center_ch_hist/{}-ch{}-{}.png".format(count, i, "coarse"), dpi=DPI_VALUE-100, bbox_inches='tight')
                    plt.close()
                    weight_fine = torch.flatten(weight_fine).cpu().detach().numpy()
                    figure = plt.figure(figsize=(1, 1))
                    plt.hist(weight_fine, bins=1024, range=(weight_fine.min(), weight_fine.max()), linewidth=0.5)
                    plt.xticks(fontproperties=font)
                    plt.yticks(fontproperties=font)
                    plt.title('{}-{}'.format(i, "fine"), fontproperties=font)
                    # plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
                    plt.savefig("./log/center_ch_hist/{}-ch{}-{}.png".format(count, i, "fine"), dpi=DPI_VALUE-100, bbox_inches='tight')
                    plt.close()
            count += 1