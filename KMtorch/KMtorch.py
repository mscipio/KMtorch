from .helpers import Utils, Holder
from .optimizer import LMoptimizer

import pycuda.autoinit
import numpy as np
from string import Template
from pycuda.compiler import SourceModule
import torch
import os

__all__ = ['KMtorch','LMoptimizer']

class KMtorch:
    def __init__(self, Kparams, IFparams, IF, time, dimensions = (128,128,47,20,5), activity=None,
                 mask = None, block_num=256, model='bicompartment_3expIF_4k', prior='gaussian_MRF_prior', priorargs = {}):

        self.Nx, self.Ny, self.Nz, self.Nt, self.Nk = dimensions
        self.Nv = self.Nx * self.Ny * self.Nz
        self.volsize = (self.Nx, self.Ny, self.Nz, self.Nt)
        self.volvecsize = (self.Nv, self.Nt)
        self.parsize = (self.Nx, self.Ny, self.Nz, self.Nk)
        self.parvecsize = (self.Nv, self.Nk)
        self.masksize = (self.Nx, self.Ny, self.Nz)

        assert Kparams.shape == self.parsize or Kparams.shape == self.parvecsize
        assert time.shape[0] == self.Nt or time.shape[1] == self.Nt
        assert IF.shape[0] == self.Nt or IF.shape[1] == self.Nt
        if activity is not None:
            assert activity.shape == self.volsize or activity.shape == self.volvecsize
        if mask is not None:
            assert mask.shape == self.masksize

        self.block = block_num
        self.grid = np.int(np.ceil((self.Nv + self.block - 1) / self.block))
        self.model_type = model
        self.prior_type = prior
        self.utils = Utils()
        self.kpar = self.utils.checkInputs(Kparams)
        self.auxpar = self.torch_par2aux(self.kpar, self.Nv, self.Nk)
        self.IFpar = self.utils.checkInputs(IFparams)
        self.IF = self.utils.checkInputs(IF)
        self.time = self.utils.checkInputs(time)
        if activity is None:
            self.activity = torch.empty(self.volvecsize).type(torch.cuda.FloatTensor).cuda()
        else:
            self.activity = self.utils.checkInputs(activity).view(self.volvecsize)
        self.activityMeas = self.activity.clone()
        self.jac = torch.empty((self.Nv, self.Nt, self.Nk)).type(torch.cuda.FloatTensor).cuda()
        if mask is None:
            self.mask = torch.ones(self.masksize).type(torch.cuda.FloatTensor).cuda()
        else:
            self.mask = self.utils.checkInputs(mask).view(self.masksize)
        self.prior = torch.empty(self.parvecsize).type(torch.cuda.FloatTensor).cuda()

        self.dk = torch.Tensor([0.006312816]).type(torch.cuda.FloatTensor).cuda()
        self.checkPriorArgs(priorargs)
        print(self.priorargs)

        self.modulepath = os.path.dirname(os.path.abspath(__file__))
        self.cudapath = self.modulepath + '/cuda/'
        self.cudaModelLoading()

    def cudaModelLoading(self):

        kernel_models = open(self.cudapath+'models.cu').read()
        kernel_code_template = Template(kernel_models)
        mod = SourceModule(
            kernel_code_template.substitute(max_threads_per_block=self.block,
                                            max_blocks_per_grid=self.grid,
                                            W=self.Ny,
                                            L=self.Nz,
                                            T=self.Nt,
                                            D=self.Nk,
                                            N=self.Nv))
        self.modelfun = mod.get_function(self.model_type)

        kernel_priors = open(self.cudapath + 'priors.cu').read()
        kernel_code_template = Template(kernel_priors)
        mod = SourceModule(
            kernel_code_template.substitute(max_threads_per_block=self.block,
                                            max_blocks_per_grid=self.grid,
                                            W=self.Ny,
                                            L=self.Nz,
                                            D=self.Nk,
                                            N=self.Nv))
        self.logprior = mod.get_function(self.prior_type)

    def checkPriorArgs(self,priorargs):
        ppc = 2*torch.floor(torch.log10(torch.max(self.activityMeas))).cuda()
        ppn = ppc.cpu().numpy()
        if priorargs:
            if isinstance(priorargs, dict):
                self.priorargs = priorargs
                if 'beta' not in self.priorargs:
                    self.priorargs['beta'] = torch.from_numpy(np.asarray([0.1, 0.1, 0.1, 0.1, 0.1])*10**ppn).type(torch.cuda.FloatTensor).cuda()
                else:
                    self.priorargs['beta'] = self.utils.checkInputs(priorargs['beta'])*10**ppc
                if 'gamma' not in self.priorargs:
                    self.priorargs['gamma'] = torch.from_numpy(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0])*10**ppn).type(torch.cuda.FloatTensor).cuda()
                else:
                    self.priorargs['gamma'] = self.utils.checkInputs(priorargs['gamma'])*10**ppc
                if 'threshold' not in self.priorargs:
                    self.priorargs['threshold'] = torch.from_numpy(np.asarray([1, 10, 10, 1, 0])*10**ppn).type(torch.cuda.FloatTensor).cuda()
                else:
                    self.priorargs['threshold'] = self.utils.checkInputs(priorargs['threshold'])*10**ppc
            else:
                self.priorargs = {}
                self.priorargs['beta'] = torch.from_numpy(np.asarray([0.1, 0.1, 0.1, 0.1, 0.1])*10**ppn).type(torch.cuda.FloatTensor).cuda()
                self.priorargs['gamma'] = torch.from_numpy(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0])*10**ppn).type(torch.cuda.FloatTensor).cuda()
                self.priorargs['threshold'] = torch.from_numpy(np.asarray([1, 10, 10, 1, 0])*10**ppn).type(torch.cuda.FloatTensor).cuda()
        else:
            self.priorargs = {}
            self.priorargs['beta'] = torch.from_numpy(np.asarray([0.1, 0.1, 0.1, 0.1, 0.1])*10**ppn).type(torch.cuda.FloatTensor).cuda()
            self.priorargs['gamma'] = torch.from_numpy(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0])*10**ppn).type(torch.cuda.FloatTensor).cuda()
            self.priorargs['threshold'] = torch.from_numpy(np.asarray([1, 10, 10, 1, 0])*10**ppn).type(torch.cuda.FloatTensor).cuda()

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

    def computePrior(self, auxpar=None):
        if auxpar is None:
            auxpar = self.auxpar
        else:
            auxpar = self.utils.checkInputs(auxpar)

        self.logprior(Holder(auxpar), Holder(self.prior),
                      Holder(self.priorargs['beta']), Holder(self.priorargs['gamma']),
                      Holder(self.priorargs['threshold']),
                      grid=(self.grid, 1, 1), block=(self.block, 1, 1))
        return self.prior

    def fitMeasure(self, initialPoint=None, useprior=False, maxIteration=100, tol=1e-4):
        if initialPoint is None:
            initialPoint = self.auxpar
        else:
            initialPoint = self.utils.checkInputs(initialPoint)
        fData = self.activityMeas.clone()
        self.activityMeas = self.activity.clone()
        self.LMoptim = LMoptimizer(function=self.evaluateModel, priorfun = self.computePrior, useprior = useprior,
                                   model_params=initialPoint, measure=fData, learning_rate=1.0)
        self.LMoptim.run(maxIteration,tol)
        self.activity = self.LMoptim.f
        self.auxpar = self.LMoptim.p
        self.kpar = self.torch_aux2par(self.auxpar, self.Nv, self.Nk)
        torch.cuda.empty_cache()
        #return LMoptim

    def getFit(self,numpy=True):
        if numpy:
            return self.activity.cpu().view(self.volsize).numpy()
        else:
            return self.activity.view(self.volsize)

    def getMeasure(self,numpy=True):
        if numpy:
            return self.activityMeas.cpu().view(self.volsize).numpy()
        else:
            return self.activityMeas.view(self.volsize)

    def getParams(self,numpy=True):
        if numpy:
            return self.kpar.cpu().view(self.parsize).numpy()
        else:
            return self.kpar.view(self.parsize)

    @staticmethod
    def torch_par2aux(k, voxels, params):
        eps = 1e-9
        aux_par = torch.zeros(k.shape).reshape((voxels, params)).type(torch.cuda.FloatTensor).cuda()
        k = k.reshape((voxels, params))
        d = torch.abs(torch.sqrt((k[:, 2] + k[:, 3] + k[:, 4]) ** 2 - 4 * k[:, 2] * k[:, 4]))
        aux_par[:, 2] = (k[:, 2] + k[:, 3] + k[:, 4] + d) / 2  # L1
        aux_par[:, 4] = (k[:, 2] + k[:, 3] + k[:, 4] - d) / 2  # L2
        aux_par[:, 1] = (k[:, 1] * (k[:, 2] - k[:, 3] - k[:, 4] + d)) / (2 * d + eps)  # B1
        aux_par[:, 3] = (k[:, 1] * (-k[:, 2] + k[:, 3] + k[:, 4] + d)) / (2 * d + eps)  # B2
        aux_par[:, 0] = k[:, 0]  # vB
        return aux_par

    @staticmethod
    def torch_aux2par(aux_par, voxels, params):
        eps = 1e-9
        k = torch.zeros(aux_par.shape).reshape((voxels, params)).type(torch.cuda.FloatTensor).cuda()
        aux_par = torch.from_numpy(np.asarray(aux_par)).reshape((voxels, params)).type(torch.cuda.FloatTensor).cuda()
        n = aux_par[:, 1] * aux_par[:, 2] + aux_par[:, 3] * aux_par[:, 4]
        d = aux_par[:, 1] + aux_par[:, 3]
        k[:, 0] = aux_par[:, 0]  # vB
        k[:, 1] = d
        k[:, 2] = n / (d + eps)
        k[:, 3] = (aux_par[:, 1] * aux_par[:, 3] * (aux_par[:, 2] - aux_par[:, 4]) ** 2) / (d * n + eps)
        k[:, 4] = (aux_par[:, 2] * aux_par[:, 4] * d) / (n + eps)
        return k
