from .helpers import Utils, Holder
from .LMoptimizer import LMoptimizer

import pycuda.driver as drv
import numpy as np
from string import Template
from pycuda.compiler import SourceModule
import torch


class KMtorch:
    def __init__(self, Kparams, IFparams, IF, time, dimensions, activity=None,
                 block_num=256, model='bicompartment_3expIF_4k'):

        self.Nv, self.Nt, self.Nk = dimensions
        self.block = block_num
        self.grid = np.int(np.ceil((self.Nv + self.block - 1) / self.block))
        self.model = model
        self.utils = Utils()
        self.kpar = self.utils.checkInputs(Kparams)
        self.auxpar = self.torch_par2aux(self.kpar, self.Nv, self.Nk)
        self.IFpar = self.utils.checkInputs(IFparams)
        self.IF = self.utils.checkInputs(IF)
        self.time = self.utils.checkInputs(time)
        if activity is None:
            self.activity = torch.empty((self.Nv, self.Nt)).type(torch.cuda.FloatTensor).cuda()
        else:
            self.activity = self.utils.checkInputs(activity)
        self.activityMeas = self.activity.clone()
        self.jac = torch.empty((self.Nv, self.Nt, self.Nk)).type(torch.cuda.FloatTensor).cuda()
        self.mask = torch.ones((128, 128, 47)).type(torch.cuda.FloatTensor).cuda()
        self.dk = torch.tensor([0.006312816]).type(torch.cuda.FloatTensor).cuda()

        self.cudaModelLoading()

    def cudaModelLoading(self):
        drv.init()
        kernel_models = open('../__incoming_publications/functions/cuda_kernels/models.cu').read()
        kernel_code_template = Template(kernel_models)
        mod = SourceModule(
            kernel_code_template.substitute(max_threads_per_block=self.block,
                                            max_blocks_per_grid=self.grid,
                                            W=self.Nv,
                                            L=1,
                                            T=self.Nt,
                                            D=self.Nk,
                                            N=self.Nv))
        self.modelfun = mod.get_function(self.model)

    def evaluateModel(self, auxpar=None):
        if auxpar is None:
            auxpar = self.auxpar
        else:
            auxpar = self.utils.checkInputs(auxpar)
        self.modelfun(Holder(auxpar), Holder(self.IFpar),
                      Holder(self.IF), Holder(self.time), Holder(self.activity),
                      Holder(self.jac), Holder(self.dk), Holder(self.mask),
                      grid=(self.grid, 1, 1), block=(self.block, 1, 1))
        self.auxpar = auxpar
        self.kpar = self.torch_aux2par(self.auxpar, self.Nv, self.Nk)
        return self.activity, self.jac

    def fitMeasure(self, maxIteration=10, initialPoint=None):
        if initialPoint is None:
            initialPoint = self.auxpar
        else:
            initialPoint = self.utils.checkInputs(initialPoint)
        fData = self.activity.clone()
        self.activityMeas = self.activity.clone()
        LMoptim = LMoptimizer(function=self.evaluateModel, model_params=initialPoint,
                              measure=fData, learning_rate=1.0)
        LMoptim.run(maxIteration)
        self.activity = LMoptim.f
        self.auxpar = LMoptim.p
        self.kpar = self.torch_aux2par(self.auxpar, self.Nv, self.Nk)
        return LMoptim

    def torch_par2aux(self, k, voxels, params):
        # auxiliar parameters
        aux_par = torch.zeros(k.shape).reshape((voxels, params)).type(torch.cuda.FloatTensor).cuda()
        k = k.reshape((voxels, params))
        d = torch.abs(torch.sqrt((k[:, 2] + k[:, 3] + k[:, 4]) ** 2 - 4 * k[:, 2] * k[:, 4]));
        aux_par[:, 2] = (k[:, 2] + k[:, 3] + k[:, 4] + d) / 2  # L1
        aux_par[:, 4] = (k[:, 2] + k[:, 3] + k[:, 4] - d) / 2  # L2
        aux_par[:, 1] = (k[:, 1] * (k[:, 2] - k[:, 3] - k[:, 4] + d)) / (2 * d)  # B1
        aux_par[:, 3] = (k[:, 1] * (-k[:, 2] + k[:, 3] + k[:, 4] + d)) / (2 * d)  # B2
        aux_par[:, 0] = k[:, 0]  # vB
        return aux_par

    def torch_aux2par(self, aux_par, voxels, params):
        k = torch.zeros(aux_par.shape).reshape((voxels, params)).type(torch.cuda.FloatTensor).cuda()
        aux_par = torch.from_numpy(np.asarray(aux_par)).reshape((voxels, params)).type(torch.cuda.FloatTensor).cuda()
        n = aux_par[:, 1] * aux_par[:, 2] + aux_par[:, 3] * aux_par[:, 4]
        d = aux_par[:, 1] + aux_par[:, 3] + 1e-9
        k[:, 0] = aux_par[:, 0]  # vB
        k[:, 1] = d
        k[:, 2] = n / d
        k[:, 3] = (aux_par[:, 1] * aux_par[:, 3] * (aux_par[:, 2] - aux_par[:, 4]) ** 2) / (d * n)
        k[:, 4] = aux_par[:, 2] * aux_par[:, 4] * d / n
        return k
