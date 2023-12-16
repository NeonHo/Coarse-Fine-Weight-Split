#!/bin/bash

python in_testing/parser/test_repvgg.py --device 3 --onnx_path dev/thirdparty/RepVGG/weights/1280mmRepVGG-A1.onnx --ptq_cfg KLConfig --arch RepVGG-A1 --batch-size 32 --calib_num 32 --no_quant --need_fp_eval --need_filt_weight --filt_act_layer_idx -1

python in_testing/parser/test_repvgg.py --device 3 --onnx_path dev/thirdparty/RepVGG/weights/1280mmRepVGG-A1.onnx --ptq_cfg KLConfig --arch RepVGG-A1 --batch-size 32 --calib_num 32 --no_quant --need_fp_eval --need_filt_weight --filt_act_layer_idx -2

python in_testing/parser/test_repvgg.py --device 3 --onnx_path dev/thirdparty/RepVGG/weights/1280mmRepVGG-A1.onnx --ptq_cfg KLConfig --arch RepVGG-A1 --batch-size 32 --calib_num 32 --no_quant --need_fp_eval --need_filt_weight --filt_act_layer_idx -3
