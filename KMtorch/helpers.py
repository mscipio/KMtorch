
import pycuda
from pycuda.driver import PointerHolderBase as pycudaHolder
import torch
import numpy as np

__all__ = ['Holder','Utils']

class Holder(pycudaHolder):
    def __init__(self, t):
        super(pycudaHolder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()


class Utils():
    def checkInputs(self, var):
        if isinstance(var, torch.Tensor):
            if var.is_cuda:
                var_out = var.type(torch.cuda.FloatTensor)
            else:
                var_out = var.type(torch.cuda.FloatTensor).cuda()
        else:
            var_out = torch.from_numpy(np.asarray(var)).type(torch.cuda.FloatTensor).cuda()
        return var_out
