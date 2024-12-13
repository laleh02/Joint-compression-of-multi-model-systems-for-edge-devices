import torch
from torch.optim import AdamW

class INQAdamW(AdamW):
    """Wrapper class for AdamW that ste quantization bit-widths and quantization order, necessary for the quantization scheduler in INQ"""

    def __init__(self, weight_bits=None, *args, **kwargs):

        assert(weight_bits >= 0)

        super(INQAdamW,self).__init__(*args,**kwargs)

        
        for group in self.param_groups:
            group['Ts'] = []
            for p in group['params']:
                if p.requires_grad is False:
                    group['Ts'].append(0)
                    continue

                T = torch.ones_like(p.data)
                group['Ts'].append(T)
                group['weight_bits'] = weight_bits
