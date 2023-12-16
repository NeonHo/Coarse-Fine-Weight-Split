from demo.in_network.ptq_configs.imagenet import BaseImageNet
from hmquant.configs.base_ptq import BaseHoumoConfig, KLConfig
from hmquant.configs.register import ptq_config_register
from hmquant.ptq.nn_layers.conv import MinMaxQuantConv2d
from in_testing.parser.neon_exp_tools.rep_conv import RepVGGQuantConv2d
from in_testing.parser.neon_exp_tools.set_opts import arch_node_id_list_dict


@ptq_config_register
class MinMaxConfig(BaseHoumoConfig):
    def __init__(self) -> None:
        super().__init__() 
    
    def qconv_class(self, *args, **kwargs) -> MinMaxQuantConv2d:
        module = MinMaxQuantConv2d(*args, **kwargs)
        return module


@ptq_config_register
class RepVGGKLConfig(KLConfig):
    def __init__(self) -> None:
        super().__init__()
    
    def init(self, args):
        self.args = args
        if args.arch in arch_node_id_list_dict.keys():
            rep_node_id_list = arch_node_id_list_dict[args.arch]
        else:
            print("Wrong Arch")
            exit()
        for node_id in rep_node_id_list:  # 为node_fix_map指定RepVGGBlock的Conv指定RepVGGConv.
            self.node_fix_map[node_id] = self.rep_qconv_class
            
    def rep_qconv_class(self,*args,**kwargs):
        module = RepVGGQuantConv2d(*args,**kwargs)
        module.need_fake_channelwise_coarseweight = self.args.no_rep_perchannel
        module.extract_center = (not self.args.no_rep_extract_center)
        module.klconv = True
        return module


@ptq_config_register
class RepVGGMinMaxConfig(MinMaxConfig):
    def __init__(self) -> None:
        super().__init__()
    
    def init(self, args):
        self.args = args
        if args.arch in arch_node_id_list_dict.keys():
            rep_node_id_list = arch_node_id_list_dict[args.arch]
        else:
            print("Wrong Arch")
            exit()
        for node_id in rep_node_id_list:  # 为node_fix_map指定RepVGGBlock的Conv指定RepVGGConv.
            self.node_fix_map[node_id] = self.rep_qconv_class

    def rep_qconv_class(self,*args,**kwargs):
        module = RepVGGQuantConv2d(*args,**kwargs)
        module.need_fake_channelwise_coarseweight = self.args.no_rep_perchannel
        module.extract_center = (not self.args.no_rep_extract_center)
        module.klconv = False
        return module
            

@ptq_config_register
class OnlyRepVGGKLConfig(MinMaxConfig):
    def __init__(self) -> None:
        super().__init__()
    
    def init(self, args):
        self.args = args
        if args.arch in arch_node_id_list_dict.keys():
            rep_node_id_list = arch_node_id_list_dict[args.arch]
        else:
            print("Wrong Arch")
            exit()
        for node_id in rep_node_id_list:  # 为node_fix_map指定RepVGGBlock的Conv指定RepVGGConv.
            self.node_fix_map[node_id] = self.rep_qconv_class
            
    def rep_qconv_class(self,*args,**kwargs):
        module = RepVGGQuantConv2d(*args,**kwargs)
        module.need_fake_channelwise_coarseweight = self.args.no_rep_perchannel
        module.extract_center = (not self.args.no_rep_extract_center)
        module.klconv = True
        return module