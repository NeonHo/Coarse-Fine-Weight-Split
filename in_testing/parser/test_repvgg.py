import argparse
import sys
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
from dev.thirdparty.RepVGG.train.config import get_config
from dev.thirdparty.RepVGG.train.logger import create_logger
from dev.thirdparty.RepVGG.data.build import build_transform
from in_testing.parser.neon_exp_tools.rep_sequencer_module import RepSequencer
from in_testing.parser.neon_exp_tools.set_opts import set_conv_16bit, set_ops_specific_opname, set_conv_before_relu_onlypos, arch_node_id_list_dict
from in_testing.parser.neon_exp_tools.export_onnx import input_name, input_shape
from in_testing.parser.neon_exp_tools.plot_tools import plot_inference_center_8neighbors_box_plot, plot_inference_center_8neighbors_violin_plot, plot_lines, plot_weight_activation, plot_inference_center_coarse_fine_hist, plot_inference_center_channel_coarse_fine_hist
from timm.utils import accuracy, AverageMeter

from in_testing.parser.neon_exp_tools.statistic import cal_std
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from dev.thirdparty import RepVGG
from pathlib import Path
ROOT = Path(list(RepVGG.__path__)[0]).resolve()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import numpy
import torch
from torch import nn
from torchvision import datasets
import torch.backends.cudnn as cudnn
from hmquant.tools import parse_onnx
from hmquant.ptq.preprocess_utils import LoadedImageProcessor
from in_testing.parser.neon_exp_tools import repvgg_ptq
from hmquant.configs import get_ptq_cfg
from hmquant.ptq.quantization import seq_quantize, onnx_set_quant_op

from torch.utils.data import DataLoader
from hmquant.ptq.quant_analyzer import PTQQuantAnalyzerTb
import hmquant.tools.dataset.classic_classification as datasets
from dev.thirdparty.RepVGG.main import validate

# torch.manual_seed(0)
# numpy.random.seed(0)

def buid_calib_image_processor(cfg):
    generator = datasets.ImageNetLoaderGenerator(args.data_root, "imagenet", args.calib_num, args.batch_size, num_workers=8)
    loaded_image_processor = LoadedImageProcessor(cfg)
    return generator, loaded_image_processor


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        #   =============================== deepsup part
        if type(output) is dict:
            output = output['main']

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg

class RepVGGClassifier(nn.Module):
    def __init__(self, sequencer: RepSequencer) -> None:
        super().__init__()
        self.net = sequencer
    
    def forward(self, x):
        output = self.net.forward(x)
        return output[0]


def evaluate_repvgg(sequencer: RepSequencer, test_loader: DataLoader, config, logger):
    model = RepVGGClassifier(sequencer=sequencer)
    acc1, acc5, loss = validate(config=config, data_loader=test_loader, model=model)
    logger.info(f"Only eval. top-1 acc, top-5 acc, loss: {acc1:.3f}, {acc5:.3f}, {loss:.5f}")
    return acc1


def test_repvgg_sequence(args, config, logger):
    batch_input_shape = input_shape
    batch_input_shape[0] = args.batch_size
    torch.cuda.set_device("cuda:{}".format(args.device))
    nodes = parse_onnx(args.onnx_path, input_shapes={input_name: batch_input_shape})
    sequencer_raw = RepSequencer(nodes)
    if args.draw_raw_box:
        plot_inference_center_8neighbors_box_plot(sequencer_raw=sequencer_raw, args=args)
        exit()
    if args.draw_raw_violin:
        plot_inference_center_8neighbors_violin_plot(sequencer_raw=sequencer_raw, args=args)
        exit()
    if args.cal_std:
        cal_std(sequencer_raw=sequencer_raw, args=args)
        exit()
    if args.draw_coarse_fine_centers_hist:
        plot_inference_center_coarse_fine_hist(sequencer_raw=sequencer_raw, args=args)
        # plot_inference_center_channel_coarse_fine_hist(sequencer_raw=sequencer_raw, args=args)
        exit()
        
    # quantize
    ptq_cfg = get_ptq_cfg(args.ptq_cfg)
    if isinstance(ptq_cfg, repvgg_ptq.RepVGGKLConfig) or isinstance(ptq_cfg, repvgg_ptq.RepVGGMinMaxConfig) or isinstance(ptq_cfg, repvgg_ptq.OnlyRepVGGKLConfig):
        ptq_cfg.init(args)
    from copy import deepcopy

    sequencer = onnx_set_quant_op.set_quant_operators(ptq_cfg, deepcopy(sequencer_raw), debug=True)
    # set_conv_before_relu_onlypos(sequencer=sequencer, arch=args.arch)  # Neon Exp Neg filt
    if args.analyze:
        PTQQuantAnalyzerTb("repvgg").hook_modules(sequencer)
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda:{}'.format(args.device))
    sequencer.set_device(device)

    print(sequencer.nodes)
    
    # from torchvision.datasets import ImageFolder
    # transform_imagenet = build_transform(False, config)
    # imagenet_dataset = ImageFolder(os.path.join(args.data_root, "val"), transform=transform_imagenet)
    # imagenet_dataloader= DataLoader(imagenet_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    
    generator, loaded_image_processor = buid_calib_image_processor(cfg=ptq_cfg)
    
    if not args.no_quant:
        # make calibraiton data loader
        calib_loader = generator.calib_loader(num=args.batch_size)
        if args.quant_conv_16bit:
            set_conv_16bit(sequencer, arch_node_id_list_dict[args.arch])
        seq_quantize.crosslayer_ptq_calib_sequencer(
            sequencer,
            calib_loader,
            loaded_image_processor,
            args.n_next_layers,
            args.cpu,
        )
    else:
        sequencer.set_ops_mode("raw")
        
    if args.draw_hist:
        plot_weight_activation(sequencer=sequencer, sequencer_raw=sequencer_raw, args=args)
    
    if args.need_eval:
        if args.no_quant_softmax:
            set_ops_specific_opname(sequencer, ["Softmax"], "raw")
        # evaluate_repvgg(sequencer=sequencer, test_loader=imagenet_dataloader, config=config, logger=logger)
        evaluate_repvgg(sequencer=sequencer, test_loader=generator.test_loader(), config=config, logger=logger)
    if args.need_fp_eval:
        sequencer_raw.set_device(device=device)
        if args.need_filt_activation:
            sequencer_raw.activation_node_id_list.append(arch_node_id_list_dict[args.arch][args.filt_act_layer_idx])
            step_data = []
            acc_data = []
            for step in range(100, 0, -1):
                sequencer_raw.activation_threshold_float = float(step / 100.0)
                acc1 = evaluate_repvgg(sequencer=sequencer_raw, test_loader=generator.test_loader(), config=config, logger=logger)
                step_data.append(sequencer_raw.activation_threshold_float)
                acc_data.append(acc1)
            print(args.arch)
            print(step_data)
            print(acc_data)
            plot_lines(step_data, acc_data, arch=args.arch, layer_idx=sequencer_raw.activation_node_id_list[-1])
        elif args.need_filt_weight:
            sequencer_raw.weight_node_id_list.append(arch_node_id_list_dict[args.arch][args.filt_act_layer_idx])
            step_data = []
            acc_data = []
            for step in range(1000, 990, -2):
                sequencer_raw.weight_threshold_float = float(step / 1000.0)
                acc1 = evaluate_repvgg(sequencer=sequencer_raw, test_loader=generator.test_loader(), config=config, logger=logger)
                step_data.append(sequencer_raw.weight_threshold_float)
                acc_data.append(acc1)
            plot_lines(step_data, acc_data, arch=args.arch, layer_idx=sequencer_raw.weight_node_id_list[-1])
        else:
            evaluate_repvgg(sequencer=sequencer_raw, test_loader=generator.test_loader(), config=config, logger=logger)
    

def parse_option_houmo():
    parser = argparse.ArgumentParser('RepOpt-VGG training script built on the codebase of Swin Transformer', add_help=False)
    
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=["DATA.DATASET", "imagenet", "DATA.IMG_SIZE", "224"], nargs='+')
    parser.add_argument("--onnx_path", default="dev/thirdparty/RepVGG/weights/mmRepVGG-A1.onnx", type=str)
    parser.add_argument("--data_root", default="dev/thirdparty/datasets/imagenet", type=str)
    parser.add_argument("--batched_calib", action="store_true")
    parser.add_argument("--draw_hist", default=False, action="store_true", help='if use, need to add self.ptq_out = out_sim into calibration_step2 of conv_cimd.py')
    parser.add_argument("--draw_raw_box", default=False, action="store_true", help='plot box of center and 8 neighbors for each RepConv.')
    parser.add_argument("--draw_raw_violin", default=False, action="store_true", help='plot violin of center and 8 neighbors for each RepConv.')
    parser.add_argument("--draw_coarse_fine_centers_hist", default=False, action="store_true", help='plot hist of center and its coarse and fine for each RepConv.')
    parser.add_argument("--cal_std", default=False, action="store_true", help='calculate std of cneter and 8 neighbors for each RepConv.')
    parser.add_argument("--need_eval", default=False, action="store_true", help="if use, the quantized model will be eval in the end.")
    parser.add_argument("--need_fp_eval", default=False, action="store_true", help="if use, the FP model will be eval in the end.")
    parser.add_argument("--need_filt_activation", default=False, action="store_true", help="if use, the FP model's last rep conv activation will be filt step by step end with a line plot.")
    parser.add_argument("--filt_act_layer_idx", default=-1, type=int)
    parser.add_argument("--need_filt_weight", default=False, action="store_true", help="if use, the FP model's last rep conv weight will be filt step by step end with a line plot.")
    parser.add_argument("--no_quant_softmax", default=True, action="store_true", help="no quant for RepVGGConv.")
    parser.add_argument("--quant_conv_16bit", default=False, action="store_true")
    parser.add_argument("--no_rep_extract_center", default=False, action="store_true", help="True means without extract central parameters of repvgg conv.")
    parser.add_argument("--no_rep_perchannel", default=False, action="store_true", help="Let RepVGGBlock Conv quantization with per-channel mode.")
    parser.add_argument("--calib_num", default=32, type=int)
    parser.add_argument("--n_test", default=-1, type=int)
    parser.add_argument("--ptq_cfg", default="RepVGGMinMaxConfig", type=str)  # BaseHoumoConfig KLConfig
    parser.add_argument("--n_next_layers", default=1, type=int)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_quant", action="store_true")
    parser.add_argument('--arch', default=None, type=str, help='arch name, YOLOv6-s, RepVGG-A1, RepVGG-B1.')
    parser.add_argument('--batch-size', default=128, type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', default='dev/thirdparty/datasets/imagenet', type=str, help='path to dataset')
    parser.add_argument('--scales-path', default=None, type=str, help='path to the trained Hyper-Search model')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, full: cache all data, part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='output/RepVGG', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', default="test", help='tag of experiment')
    parser.add_argument('--eval', default=True, action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


if __name__ == "__main__":
    args, config = parse_option_houmo()
    cudnn.benchmark = True
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.ARCH}")
    
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    test_repvgg_sequence(args, config, logger)
