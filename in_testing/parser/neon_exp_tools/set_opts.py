from hmquant.ptq.nn_layers.base_interface import BaseInterface
from hmquant.ptq.sequencer_module import Sequencer


yolov6t_rep_node_id_list = [
    "119",
    "121", "123", "125", "127", "129",
    "131", "133", "135", "137", "139",
    "141", "143", "145", "147", "149",
    "151", "153", "155",
    "169",
    "171", "173", "175",
    "181", "183", "185", "187",
    "192", "194", "196", "198",
    "203", "205", "207", "209"
]

yolov6s_rep_node_id_list = [
    "119",
    "121", "123", "125", "127", "129",
    "131", "133", "135", "137", "139",
    "141", "143", "145", "147", "149",
    "151", "153", "155",
    "169",
    "171", "173", "175",
    "181", "183", "185", "187",
    "192", "194", "196", "198",
    "203", "205", "207", "209"
]

vgga0_rep_node_id_list = [
    "47", "49",
    "51", "53", "55", "57", "59",
    "61", "63", "65", "67", "69",
    "71", "73", "75", "77", "79",
    "81", "83", "85", "87", "89"
]


vgga1_rep_node_id_list = [
    "47", "49",
    "51", "53", "55", "57", "59",
    "61", "63", "65", "67", "69",
    "71", "73", "75", "77", "79",
    "81", "83", "85", "87", "89"
]

vggb0_rep_node_id_list = [
    "59", 
    "61", "63", "65", "67", "69",
    "71", "73", "75", "77", "79",
    "81", "83", "85", "87", "89",
    "91", "93", "95", "97", "99",
    "101", "103", "105", "107", "109",
    "111", "113"
]

vggb1_rep_node_id_list = [
    "59", 
    "61", "63", "65", "67", "69",
    "71", "73", "75", "77", "79",
    "81", "83", "85", "87", "89",
    "91", "93", "95", "97", "99",
    "101", "103", "105", "107", "109",
    "111", "113"
]

arch_node_id_list_dict = {
    "RepVGG-A0": vgga0_rep_node_id_list,
    "RepVGG-A1": vgga1_rep_node_id_list,
    "RepVGG-B0": vggb0_rep_node_id_list,
    "RepVGG-B1": vggb1_rep_node_id_list,
    "YOLOv6-t": yolov6t_rep_node_id_list,
    "YOLOv6-s": yolov6s_rep_node_id_list
}

def set_ops_specific_opname(sequencer: Sequencer, op_name_list: list, mode: str):
    """修改list中指定op_name的算子为mode.

    Args:
        sequencer (Sequencer): 模型序列结构本身.
        op_name_list (list): 指定op_name算子的列表.
        mode (str): 指定的推断模式, "raw"是原始推断，"quant_forward"是量化推断。
    """
    for name, node in sequencer.nodes.items():
        if isinstance(node.op, BaseInterface) and (node.op_name in op_name_list):
            node.op.mode = mode
            print("node.op {} mode={}".format(node.op_name, node.op.mode))


def set_ops_specific_nodeid(sequencer: Sequencer, node_id_list: list, mode: str):
    """修改list中指定node_id的算子为指定mode

    Args:
        sequencer (Sequencer): 模型序列结构本身.
        node_id_list (list): 指定node_id算子的列表.
        mode (_type_): 指定的推断模式, "raw"或"quant_forward"
    """
    for name, node in sequencer.nodes.items():
        if isinstance(node.op, BaseInterface) and (name in node_id_list):
            node.op.mode = mode
            print("node id {} mode={}".format(name, node.op.mode))
            

def set_conv_16bit(sequencer: Sequencer, node_id_list: list, bit_num: int = 16):
    for name, node in sequencer.nodes.items():
        if isinstance(node.op, BaseInterface) and (node.op_name == "Conv") and (name in node_id_list):
            node.op.w_bit = bit_num
            node.op.w_qmax = 1 << (bit_num - 1)
            print("node id {}\toperation {}\tweight bit {}".format(name, node.op_name, node.op.w_bit))
            

def set_conv_before_relu_onlypos(sequencer: Sequencer, arch: str):
    
    if arch in arch_node_id_list_dict.keys():
        rep_node_id_list = arch_node_id_list_dict[arch]
    else:
        print("Wrong Arch")
        exit()
    for name, node in sequencer.nodes.items():
        if isinstance(node.op, BaseInterface) and (node.op_name == "Conv") and (name in rep_node_id_list):
            node.op.o_abs = False
            print("node id {}\toperation {}\tOnly positive value in Output Activation will be Quantized.".format(name, node.op_name))