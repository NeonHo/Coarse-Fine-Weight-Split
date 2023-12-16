import sys

from in_testing.parser.neon_exp_tools.plot_tools import plot_weight_activation
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from pathlib import Path
from dev.thirdparty import YOLOv6
ROOT = Path(list(YOLOv6.__path__)[0]).resolve()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
import argparse
import torch
from torch import nn
from hmquant.tools import parse_onnx
from hmquant.ptq.sequencer_module import Sequencer
from hmquant.ptq.preprocess_utils import LoadedImageProcessor
from hmquant.configs import get_ptq_cfg
from hmquant.ptq.quantization import seq_quantize, onnx_set_quant_op
import torchvision.datasets as dset
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
import cv2
from hmquant.tools.utils import import_lib
from hmquant.ptq.quant_analyzer import PTQQuantAnalyzer
from hmquant.tools.evaluate import build_evaluator
from dev.thirdparty.YOLOv6.yolov6.core.evaler import Evaler
from dev.thirdparty.YOLOv6.yolov6.utils.config import Config
from dev.thirdparty.YOLOv6.yolov6.utils.events import LOGGER
from dev.thirdparty.YOLOv6.yolov6.core.evaler import Evaler
from dev.thirdparty.YOLOv6.yolov6.utils.general import increment_name
import dev.thirdparty.YOLOv6.yolov6.assigners.anchor_generator as anchor_generator
from dev.thirdparty.YOLOv6.yolov6.utils.general import dist2bbox
from dev.thirdparty.YOLOv6.yolov6.data.data_augment import letterbox
from in_testing.parser.neon_exp_tools.repvgg_ptq import RepVGGKLConfig, RepVGGMinMaxConfig, OnlyRepVGGKLConfig
from in_testing.parser.neon_exp_tools.set_opts import set_conv_before_relu_onlypos, yolov6s_rep_node_id_list
from in_testing.parser.neon_exp_tools.set_opts import set_conv_16bit, set_ops_specific_nodeid, set_ops_specific_opname
from in_testing.parser.neon_exp_tools.device import reload_device


class NewYoloModel(nn.Module):
    def __init__(self, sequencer) -> None:
        super().__init__()
        self.net = sequencer
        stride = [8, 16, 32]  # strides computed during build
        self.stride = torch.tensor(stride)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

    def forward(self, x):
        feats_0, feats_1, feats_2, cls_score_list, reg_dist_list = self.net.forward(
            x
        )  # raw_forward
        feats = [feats_0, feats_1, feats_2]
        anchor_points, stride_tensor = anchor_generator.generate_anchors(
            feats,
            self.stride,
            self.grid_cell_size,
            self.grid_cell_offset,
            device=x[0].device,
            is_eval=True,
        )
        pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format="xywh")
        pred_bboxes *= stride_tensor
        outputs = torch.cat(
            [
                pred_bboxes,
                torch.ones(
                    (pred_bboxes.shape[0], pred_bboxes.shape[1], 1),
                    device=pred_bboxes.device,
                    dtype=pred_bboxes.dtype,
                ),
                cls_score_list,
            ],
            axis=-1,
        )
        return [outputs, None]


res = dict()


def insert_hooks(name, seq):
    nodes = seq.nodes

    def collect_out(m, i, o):
        if res.get(name) is None:
            res[name] = dict()
        to_save = res[name]
        for k, n in nodes.items():
            if n.op is m:
                break
        to_save[k] = o

    return collect_out


def insert(seq, name):
    nodes = seq.nodes
    for k, v in nodes.items():
        op: nn.Module = v.op
        op.register_forward_hook(insert_hooks(name, seq))


def test_yolov6_sequence(args):
    torch.cuda.set_device("cuda:{}".format(args.device))
    nodes = parse_onnx(args.onnx_path)
    sequencer_raw = Sequencer(nodes)
    insert(sequencer_raw, "raw")
    # quantize
    ptq_cfg = get_ptq_cfg(args.ptq_cfg)
    if isinstance(ptq_cfg, RepVGGKLConfig) or isinstance(ptq_cfg, RepVGGMinMaxConfig) or isinstance(ptq_cfg, OnlyRepVGGKLConfig):
        ptq_cfg.init(args)
    from copy import deepcopy

    sequencer = onnx_set_quant_op.set_quant_operators(
        ptq_cfg, deepcopy(sequencer_raw), debug=True
    )
    set_conv_before_relu_onlypos(sequencer=sequencer, arch=args.arch)
    insert(sequencer, "now")
    if args.analyze:
        PTQQuantAnalyzer("yolov6").hook_modules(sequencer)
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    sequencer.set_device(device)

    print(sequencer.nodes)

    # task
    Evaler.check_task(args.task)
    # 一定是'val'，直接创建评估所需要的文件目录
    save_dir = args.save_dir
    save_dir = str(increment_name(osp.join(save_dir, args.name)))
    os.makedirs(save_dir, exist_ok=True)
    # check the threshold value, reload device/half/data according task
    Evaler.check_thres(args.conf_thres, args.iou_thres, args.task)  # 确保阈值设置合理。
    model = NewYoloModel(sequencer=sequencer)
    device = reload_device(
        device.type, model, args.task, args.device
    )  # 相当于设置os.environ['CUDA_VISIBLE_DEVICES']

    half = device.type != "cpu" and args.half
    data = (
        Evaler.reload_dataset(args.data, args.task)
        if isinstance(args.data, str)
        else args.data
    )  # 加载yaml文件

    # init
    val = Evaler(
        data,
        args.batch_size,
        args.img_size,
        args.conf_thres,
        args.iou_thres,
        device,
        half,
        save_dir,
        args.test_load_size,
        args.letterbox_return_int,
        args.force_no_pad,
        args.not_infer_on_rect,
        args.scale_exact,
        args.verbose,
        args.do_coco_metric,
        args.do_pr_metric,
        args.plot_curve,
        args.plot_confusion_matrix,
    )

    val.stride = int(model.stride.max())
    dataloader = None
    dataloader = val.init_data(dataloader, args.task)  # 创建dataloader

    # PTQ
    # calib_loader
    cfg = import_lib("demo/configs/yolov6s.py")
    metric_cfg = cfg.evaluation["metric"]
    conf_thres = metric_cfg.get("conf_thres", 0.001)  # confidence threshold
    iou_thres = metric_cfg.get("iou_thres", 0.65)  # NMS IOU threshold
    max_det = metric_cfg.get("max_det", 1000)  # maximum detections per image

    imgsz = cfg.test_pipeline["imgsz"]
    if args.cpu:
        args.device = "cpu"
    evaluator = build_evaluator(cfg)
    valData = dset.CocoDetection(
        root=cfg.dataset["val_path"], annFile=cfg.dataset["val_anno"]
    )

    loaded_data_processor = LoadedImageProcessor(ptq_cfg)

    if not args.no_quant:
        # make calib_loader
        calib_dataset = []
        for i, (img, tgt) in enumerate(tqdm(valData)):
            img = np.array(img)
            h0, w0 = img.shape[:2]  # origin shape
            force_load_size = 638
            img_size = 640
            if force_load_size:
                r = force_load_size / max(h0, w0)
            else:
                r = img_size / max(h0, w0)
            if r != 1:
                img = cv2.resize(
                    img,
                    (int(w0 * r), int(h0 * r)),
                    interpolation=cv2.INTER_AREA
                    if r < 1 and not False
                    else cv2.INTER_LINEAR,
                )
            h, w = img.shape[:2]
            shape = img_size
            img, ratio, pad = letterbox(
                img, shape, auto=False, scaleup=False, return_int=True
            )
            shapes = (h0, w0), (
                (h * ratio / h0, w * ratio / w0),
                pad,
            )  # for COCO mAP rescaling
            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            calib_dataset.append(torch.Tensor(img).unsqueeze(0) / 255)
            if len(calib_dataset) == args.calib_num:
                break
        calib_loader = [(torch.cat(calib_dataset, 0), None)]
        
        if args.quant_conv_16bit:
            set_conv_16bit(sequencer, yolov6s_rep_node_id_list)
        seq_quantize.crosslayer_ptq_calib_sequencer(
            sequencer,
            calib_loader,
            loaded_data_processor,
            args.n_next_layers,
            cpu=args.cpu,
        )
        
    if args.draw_hist:
        plot_weight_activation(sequencer=sequencer, sequencer_raw=sequencer_raw, args=args)
    
    if args.need_eval:
        if args.no_lut:
            set_ops_specific_opname(sequencer, ["Silu", "Sigmoid"], "raw")
        if args.no_repconv_after_cat:
            set_ops_specific_nodeid(sequencer, ["203", "192", "181", "169"], "raw")
        ptq_model = NewYoloModel(sequencer=sequencer)
        ptq_model.eval()
        pred_result, vis_outputs, vis_paths = val.predict_model(
            ptq_model, dataloader, args.task
        )
        val.eval_model(pred_result, ptq_model, dataloader, args.task)


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_path",
        default="dev/thirdparty/YOLOv6/weights/yolov6s_v2_reopt.onnx",  # _v2_reopt
        type=str,
    )
    parser.add_argument("--batched_calib", action="store_true")
    parser.add_argument("--draw_hist", default=False, action="store_true", help='if use, need to add self.ptq_out = out_sim into calibration_step2 of conv_cimd.py')
    parser.add_argument("--need_eval", default=False, action="store_true", help="if use, the quantized model will be eval in the end.")
    parser.add_argument("--no_lut", default=False, action="store_true", help="no Look up Table.")
    parser.add_argument("--no_repconv_after_cat", default=False, action="store_true", help="set conv in RepVGGBlock after Concat as raw forward.")
    parser.add_argument("--no_repconv", default=False, action="store_true", help="no quant for RepVGGConv.")
    parser.add_argument("--calib_num", default=32, type=int)
    parser.add_argument("--quant_conv_16bit", default=False, action="store_true")
    parser.add_argument(
        "--ptq_cfg", default="KLConfig", type=str
    )  # BaseHoumoConfig KLConfig
    parser.add_argument("--no_rep_extract_center", default=False, action="store_true", help="True means without extract central parameters of repvgg conv.")
    parser.add_argument("--no_rep_perchannel", default=False, action="store_true", help="Let RepVGGBlock Conv quantization with per-channel mode.")
    parser.add_argument('--arch', default=None, type=str, help='arch name, YOLOv6-s, RepVGG-A1, RepVGG-B1.')
    parser.add_argument("--n_next_layers", default=1, type=int)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_quant", action="store_true")
    parser.add_argument(
        "--data", type=str, default="./dev/thirdparty/YOLOv6/data/coco.yaml", help="dataset.yaml path"
    )
    parser.add_argument(
        "--weights", type=str, default="./dev/thirdparty/YOLOv6/weights/yolov6s.pt", help="model.pt path(s)"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--img_size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.03, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.65, help="NMS IoU threshold"
    )
    parser.add_argument("--task", default="val", help="val, test, or speed")
    parser.add_argument(
        "--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--half", default=False, action="store_true", help="whether to use fp16 infer"
    )
    parser.add_argument(
        "--save_dir", type=str, default="runs/val/", help="evaluation save dir"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="save evaluation results to save_dir/name",
    )
    parser.add_argument(
        "--test_load_size", type=int, default=640, help="load img resize when test"
    )
    parser.add_argument(
        "--letterbox_return_int",
        default=False,
        action="store_true",
        help="return int offset for letterbox",
    )
    parser.add_argument(
        "--scale_exact",
        default=False,
        action="store_true",
        help="use exact scale size to scale coords",
    )
    parser.add_argument(
        "--force_no_pad",
        default=False,
        action="store_true",
        help="for no extra pad in letterbox",
    )
    parser.add_argument(
        "--not_infer_on_rect",
        default=False,
        action="store_true",
        help="default to use rect image size to boost infer",
    )
    parser.add_argument(
        "--reproduce_640_eval",
        default=True,  # Neon
        action="store_true",
        help="whether to reproduce 640 infer result, overwrite some config",
    )
    parser.add_argument(
        "--eval_config_file",
        type=str,
        default="./dev/thirdparty/YOLOv6/configs/experiment/eval_640_repro.py",
        help="config file for repro 640 infer result",
    )
    parser.add_argument(
        "--do_coco_metric",
        default=True,
        type=boolean_string,
        help="whether to use pycocotool to metric, set False to close",
    )
    parser.add_argument(
        "--do_pr_metric",
        default=False,
        type=boolean_string,
        help="whether to calculate precision, recall and F1, n, set False to close",
    )
    parser.add_argument(
        "--plot_curve",
        default=True,
        type=boolean_string,
        help="whether to save plots in savedir when do pr metric, set False to close",
    )
    parser.add_argument(
        "--plot_confusion_matrix",
        default=False,
        action="store_true",
        help="whether to save confusion matrix plots when do pr metric, might cause no harm warning print",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="whether to print metric on each class",
    )
    parser.add_argument(
        "--config-file",
        default="",
        type=str,
        help="experiments description file, lower priority than reproduce_640_eval",
    )
    args = parser.parse_args()
    if args.config_file:
        assert os.path.exists(args.config_file), print(
            "Config file {} does not exist".format(args.config_file)
        )
        cfg = Config.fromfile(args.config_file)
        if not hasattr(cfg, "eval_params"):
            LOGGER.info("Config file doesn't has eval params config.")
        else:
            eval_params = cfg.eval_params
            for key, value in eval_params.items():
                if key not in args.__dict__:
                    LOGGER.info(f"Unrecognized config {key}, continue")
                    continue
                if isinstance(value, list):
                    if value[1] is not None:
                        args.__dict__[key] = value[1]
                else:
                    if value is not None:
                        args.__dict__[key] = value

    # load params for reproduce 640 eval result
    if args.reproduce_640_eval:
        assert os.path.exists(args.eval_config_file), print(
            "Reproduce config file {} does not exist".format(args.eval_config_file)
        )
        eval_params = Config.fromfile(args.eval_config_file).eval_params
        eval_model_name = os.path.splitext(os.path.basename(args.weights))[0]
        if eval_model_name not in eval_params:
            eval_model_name = "default"
        args.test_load_size = eval_params[eval_model_name]["test_load_size"]
        args.letterbox_return_int = eval_params[eval_model_name]["letterbox_return_int"]
        args.scale_exact = eval_params[eval_model_name]["scale_exact"]
        args.force_no_pad = eval_params[eval_model_name]["force_no_pad"]
        args.not_infer_on_rect = eval_params[eval_model_name]["not_infer_on_rect"]
        # force params
        args.img_size = 640
        args.conf_thres = 0.03
        args.iou_thres = 0.65
        args.task = "val"
        args.do_coco_metric = True

    LOGGER.info(args)
    test_yolov6_sequence(args)
