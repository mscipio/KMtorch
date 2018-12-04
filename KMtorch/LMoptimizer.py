# -*- coding: utf-8 -*-

from .helpers import Utils

import torch
import numpy as np

__all__ = ['LMoptimizer']


class LMoptimizer:
    # def __init__(self, function, initialPoint, tData, fData, Nv, Nt, Nk, learningRate=1, args = None):
    def __init__(self, function, model_params, measure, learning_rate=1):
        self.epsilon = 1e-10
        self.Nv, self.Nt = measure.shape
        tmp, self.Nk = model_params.shape

        self.function = function
        self.utils = Utils()
        self.p = self.utils.checkInputs(model_params).view((self.Nv, self.Nk, 1))  # k_params
        # self.tData = self.utils.checkInputs(tData)
        self.fData = self.utils.checkInputs(measure).view((self.Nv, self.Nt))
        self.lr = self.utils.checkInputs(learning_rate)

        self.f, self.J = self.function(self.p)  # (self.tData,self.k,Nv, Nt, Nk, self.fargs) #function and jacobian
        r = -(self.fData - self.f)
        self.c = self.getF(r)

        self.v = 2
        self.alpha = 1.0
        self.exitflag = None
        self.m = self.alpha * torch.max(self.pseudoHess(self.J)) * torch.ones((self.Nv,)).type(
            torch.cuda.FloatTensor).cuda()
        self.I = torch.eye(self.Nk).repeat(self.Nv, 1, 1).type(torch.cuda.FloatTensor).cuda()

    def pseudoHess(self, jacobian):
        """
        H = J'J
        """
        return torch.bmm(jacobian.permute(0, 2, 1), jacobian)

    def getF(self, function):
        # function = self.function_array(d)
        # 0.5 * torch.mm(function.T, function)
        return 0.5 * torch.einsum('ni,ni->n', (function, function))

    def batchedInv(self, batchedTensor):
        """
        Hinv = inv()
        """
        np_tensor = batchedTensor.cpu().numpy()
        np_inv = np.linalg.inv(np_tensor)
        return torch.from_numpy(np_inv).type(torch.cuda.FloatTensor).cuda()

    def updateLMfactor(self):
        return torch.einsum('n,nij->nij', (self.m, self.I))

    "This method will return the next point in the optimization process"

    def step(self):
        """
        if self.y==0: # finished. Y can't be less than zero
            return self.x, self.y
        """
        f, jac = self.function(self.p)  # function and jacobian
        r = -(self.fData - f).view((self.Nv, self.Nt, 1))
        H = self.pseudoHess(jac)
        pseudo_grad = torch.bmm(jac.permute(0, 2, 1), r).view((self.Nv, self.Nk, 1))  # gradient approximation
        lm_factor = self.updateLMfactor()
        preconditioner = self.batchedInv(H + lm_factor).view((self.Nv, self.Nk, self.Nk))
        d_lm = - torch.bmm(preconditioner, pseudo_grad).view((self.Nv, self.Nk, 1))  # moving direction
        p_new = (self.p + self.lr * d_lm).view((self.Nv, self.Nk, 1))  # line search
        f_new, jac_new = self.function(p_new)  # function and jacobian
        r_new = -(self.fData - f_new)

        c_new = self.getF(r_new)
        grain_numerator = (self.c - c_new)

        '''gain_divisor = 0.5* torch.bmm(d_lm.permute(0, 2, 1), 
                                     torch.einsum('n,nij->nij', (self.m, d_lm))-pseudo_grad) + self.epsilon
        gain = grain_numerator / gain_divisor
        print(gain, sep=' ', end=' | ', flush=True)'''

        idx_pos = (grain_numerator > 0).nonzero()
        idx_neg = (grain_numerator <= 0).nonzero()
        self.m[idx_pos] *= (0.1)
        self.move(p_new, f_new, c_new, idx_pos)  # ok, step acceptable
        self.m[idx_neg] *= self.v

        '''
        if grain_numerator > 0.0: # it's a good function approximation.
            self.gain = gain
            self.move(p_new,f_new,c_new) # ok, step acceptable
            self.m = self.m * (0.1)#max(1 / 3, 1 - (2 * gain - 1) ** 3)
            print(self.m, sep=' ', end='(/10)\n', flush=True)
        else:
            self.m *= self.v
            print(self.m, sep=' ', end='(x2)\n', flush=True)
        '''

    def move(self, nextP, nextF, nextC, idx):
        """
        Moving to the next point.
        Saves in Optimizer class next coordinates
        """
        self.f[idx] = nextF[idx]
        self.p[idx] = nextP[idx]
        self.c[idx] = nextC[idx]

    def run(self, maxit):
        currentIteration = 0
        self.exitflag = 0
        while currentIteration <= maxit:
            self.step()
            currentIteration += 1

            '''if np.isnan(self.m):
                self.exitflag = 1
                break

            if np.isinf(self.m):
                self.exitflag = 2
                break

            if np.abs(self.gain) < 1e-15:
                self.exitflag = 3
                break'''
