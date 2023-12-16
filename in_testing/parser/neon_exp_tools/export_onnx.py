import argparse
import torch

input_name = "input"
input_shape = [1, 3, 224, 224]

def parse_args():
    parser = argparse.ArgumentParser('Neon Load .pth and export .onnx', add_help=False)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--output_onnx_path", type=str)
    args, _ = parser.parse_known_args()
    return args
    


def main(args):
    model = torch.load(args.ckpt_path)
    model.eval()
    model.forward = model.simple_test
    dummy_input = torch.ones(input_shape)
    torch.onnx.export(
        model, dummy_input, args.output_onnx_path, export_params=True, opset_version=13, input_names=[input_name], output_names=["output"], 
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    

if __name__ == "__main__":
    args = parse_args()
    main(args=args)