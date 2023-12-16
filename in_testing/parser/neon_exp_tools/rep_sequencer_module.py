from collections import OrderedDict
from copy import deepcopy

import torch
from hmquant.ptq.sequencer_module import Sequencer
from hmquant.ptq.utils import map_aggregate
from in_testing.parser.neon_exp_tools.set_opts import arch_node_id_list_dict


class RepSequencer(Sequencer):
    def __init__(self, nodes: OrderedDict, convert_backend: str = "onnx", inplace=True, debug=False) -> None:
        super().__init__(nodes, convert_backend, inplace, debug)
        self.activation_threshold_float = 1.0  # max = α × max(|out|)过滤activation_node_id_list中指定的node的输出结果(out: Tensor)为[-max, max].
        self.weight_threshold_float = 1.0
        self.activation_node_id_list = []  # 被限制激活阈值的node的ID组成的list.
        self.weight_node_id_list = []  # 被限制weights阈值的node的ID组成的list.
        
    @torch.no_grad()
    def forward(self, *input, get_output_dict=False):
        """
        @input: a list or tuple
        @get_output_dict: bool, return dict or list
        """
        # Reset
        out_counts = deepcopy(self.out_counts)
        self.results = {}

        # Prepare inputs
        for name, tensor in zip(self.inputs_tensors, input):
            self.results[name] = tensor
            if isinstance(tensor, torch.Tensor):
                self.results_shape[name] = tensor.shape

        # node = next(iter(self.nodes.values()))  # first node
        # while node:
        outputs = []
        if get_output_dict:
            outputs_dict = dict()
        for node_id, node in self.nodes.items():
            if self.debug:
                print("Start", node.id, node)
            if node_id in self.weight_node_id_list:
                max = torch.mul(torch.max(torch.abs(node.op.weight.data)), self.weight_threshold_float)
                node.op.weight.data[node.op.weight.data > max] = max
                node.op.weight.data[node.op.weight.data < -max] = -max
            # ins = [self.results[i] for i in node.in_tensors]
            ins = map_aggregate(lambda i: self.results[i], node.in_tensors)
            # if "orm" in node.op_name:
            #     ins[0].interval = node.interval
            out = node.op(*ins, **node.kwargs)
            
            if node_id in self.activation_node_id_list:
                max = torch.mul(torch.max(torch.abs(out)), self.activation_threshold_float)
                out[out > max] = max
                out[out < -max] = -max
            
            if self.debug:
                if isinstance(out, torch.Tensor):
                    print(f"output of node {node_id}: shape {out.shape}")
            # if (len(node.op.out_consumers)>0) and "orm" in node.op.out_consumers[0].op_name:
            #     node.op.out_consumers[0].interval = out.abs().max() / (128 - 0.5)
            self.results[node.id] = out
            if node_id in self.outputs_tensors:
                outputs.append(out)
                if get_output_dict:
                    outputs_dict[node_id] = out
            if isinstance(out, torch.Tensor):
                self.results_shape[node.id] = out.shape
            # Remove unused tensors
            for id in node.in_tensors:
                out_counts[id].remove(node.id)
                if (len(out_counts[id]) == 0) and (id not in self.cache_id):
                    del self.results[id]
                    if self.debug:
                        print(f"Remove unused node[{id}] output")
            if self.debug:
                print(f"node {node.id} output {str(out)[:100]}")
            # import pdb;pdb.set_trace()
            # node = node.get_next()
        if get_output_dict:
            return outputs_dict
        return outputs