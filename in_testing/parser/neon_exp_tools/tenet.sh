#!/bin/bash

arch_list=(
    "RepVGG-A1" 
    "RepVGG-B1" 
    "YOLOv6-s"
)

onnx_list=(
    "RepVGG/weights/1280mmRepVGG-A1.onnx"
    "RepVGG/weights/2048mmRepVGG-B1.onnx"
    "YOLOv6/weights/yolov6s.onnx"
)

config_list=(
    "MinMaxConfig"
    "KLConfig"
    "RepVGGMinMaxConfig"
    "RepVGGKLConfig"
    "OnlyRepVGGKLConfig"
)

device_idx=0

batch_size=32

python_script="test.py"

echo "batch size = $batch_size"
for (( i=2; i<3; i++ ))
do
    if (( i<2 ))
    then
        python_script="test_repvgg.py"
    else
        python_script="test_yolov6.py"
    fi

    for (( j=0; j<5; j++ ))
    do 
        if [[ $j -ge 2 ]] && [[ $j -lt 5 ]]
        then
            echo "${arch_list[i]} quantized according to ${config_list[j]} with center extract: "
            python in_testing/parser/${python_script} --device $device_idx --onnx_path dev/thirdparty/${onnx_list[i]} --ptq_cfg ${config_list[j]} --arch ${arch_list[i]} --batch-size $batch_size --calib_num $batch_size --need_eval 
            echo "${arch_list[i]} quantized according to ${config_list[j]} without center extract: "
            python in_testing/parser/${python_script} --device $device_idx --onnx_path dev/thirdparty/${onnx_list[i]} --ptq_cfg ${config_list[j]} --arch ${arch_list[i]} --batch-size $batch_size --calib_num $batch_size --need_eval --no_rep_extract_center
        else
            echo "${arch_list[i]} quantized according to ${config_list[j]}: "
            python in_testing/parser/${python_script} --device $device_idx --onnx_path dev/thirdparty/${onnx_list[i]} --ptq_cfg ${config_list[j]} --arch ${arch_list[i]} --batch-size $batch_size --calib_num $batch_size --need_eval
        fi
    done
done
# python in_testing/parser/test_yolov6.py --device 3 --onnx_path dev/thirdparty/YOLOv6/weights/yolov6s.onnx --ptq_cfg RepVGGKLConfig --arch YOLOv6-s --batch-size 32 --calib_num 32 --need_eval