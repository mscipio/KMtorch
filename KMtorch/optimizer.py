# -*- coding: utf-8 -*-

from .helpers import Utils

import torch
import numpy as np
import timeit
import matplotlib.pyplot as plt

__all__ = ['LMoptimizer']


class LMoptimizer:
    # def __init__(self, function, initialPoint, tData, fData, Nv, Nt, Nk, learningRate=1, args = None):
    def __init__(self, function, priorfun, model_params, measure, useprior=False, learning_rate=1):
        self.epsilon = 1e-16
        self.Nv, self.Nt = measure.shape
        tmp, self.Nk = model_params.shape

        self.function = function
        self.priorfun = priorfun
        self.useprior = bool(useprior)
        self.utils = Utils()
        self.p = self.utils.checkInputs(model_params).view((self.Nv, self.Nk, 1))  # aux_params
        # self.tData = self.utils.checkInputs(tData)
        self.fData = self.utils.checkInputs(measure).view((self.Nv, self.Nt))
        self.lr = self.utils.checkInputs(learning_rate)

        self.f, self.J = self.function(self.p)  # (self.tData,self.k,Nv, Nt, Nk, self.fargs) #function and jacobian
        r = (self.fData - self.f)
        self.c = self.getF(r)

        self.v = 5
        self.alpha = 1.0
        self.exitflag = None
        self.m = self.alpha * torch.max(self.pseudoHess(self.J)) * torch.ones((self.Nv,)).type(
            torch.cuda.FloatTensor).cuda()
        self.idx_m = (self.m > 1e4 * torch.max(self.fData)).nonzero()
        self.I = torch.eye(self.Nk).repeat(self.Nv, 1, 1).type(torch.cuda.FloatTensor).cuda()

    def pseudoHess(self, jacobian):
        """
        H = J'J
        """
        return torch.bmm(jacobian.permute(0, 2, 1), jacobian)

    def getF(self, error):
        # function = self.function_array(d)
        # 0.5 * torch.mm(function.T, function)
        return 0.5 * torch.einsum('nt,nt->n', (error, error))

    def batchedInv(self, batchedTensor):
        """
        Hinv = inv()
        """
        if batchedTensor.shape[0] >= 256 * 256 - 1:
            temp = []
            #Hinv_t = torch.empty_like(batchedTensor)
            for i,h in enumerate(torch.split(batchedTensor, 256 * 256 - 1)):
                try:
                    temp.append(torch.inverse(h))
                except:
                    print(self.m[i*(256 * 256 - 1)+47233])
                    print(h[47233])
            return torch.cat(temp)  # .type(torch.cuda.FloatTensor).cuda()
        else:
            return torch.inverse(batchedTensor)

        ## legacy
        #np_tensor = batchedTensor.cpu().numpy()
        #np_inv = np.linalg.inv(np_tensor)
        #return torch.from_numpy(np_inv).type(torch.cuda.FloatTensor).cuda()

    def updateLMfactor(self):
        # diag = self.I*self.H
        # diag[diag<=0] = self.epsilon
        # return torch.einsum('n,nij->nij', (self.m, diag))
        return torch.einsum('n,nij->nij', [self.m, self.I])

    def plot_debug(self, f, m):
        plt.plot(self.fData.cpu().numpy()[1208769, :], 'r.-')
        plt.plot(f.cpu().numpy()[1208769, :], 'b.-')
        plt.title(str(m[1208769]))
        plt.show()
        print(str(self.p[1208769,:]))
        #plt.plot(self.fData.cpu().view([256, 256, 47, 40]).numpy()[100, 118, 23, :], 'r.-')
        #plt.plot(f.cpu().view(256, 256, 47, 40).numpy()[100, 118, 23, :], 'b.-')
        #plt.title(str(m.view(256, 256, 47)[100, 118, 23]))
        #plt.show()

    def step(self):
        """
        This method will return the next point in the optimization process
        """
        start = timeit.default_timer()
        #f, jac = self.function(self.p)  # function and jacobian
        f, jac = self.f, self.J
        print('\nfunction: %.3f s' % (timeit.default_timer() - start))
        self.plot_debug(f, self.m)

        start = timeit.default_timer()
        r = (self.fData - f).view((self.Nv, self.Nt, 1))
        print('error: %.3f s' % (timeit.default_timer() - start))
        start = timeit.default_timer()
        self.H = self.pseudoHess(jac)
        print('hessian: %.3f s' % (timeit.default_timer() - start))
        start = timeit.default_timer()
        pseudo_grad = torch.bmm(jac.permute(0, 2, 1), r).view((self.Nv, self.Nk, 1))  # gradient approximation
        print('grad: %.3f s' % (timeit.default_timer() - start))

        if self.useprior:
            print('update prior')
            prior = self.priorfun(self.p)  # function and jacobian
            print('Max grad: %f' % (torch.max(pseudo_grad)))
            print('Max prior: %f' % (torch.max(prior)))
            pseudo_grad += prior.view((self.Nv, self.Nk, 1))

        start = timeit.default_timer()
        lm_factor = self.updateLMfactor()
        print('LMfactor: %.3f s' % (timeit.default_timer() - start))
        start = timeit.default_timer()
        preconditioner = self.batchedInv(self.H + lm_factor).view((self.Nv, self.Nk, self.Nk))
        #preconditioner = torch.inverse(self.H + lm_factor).view((self.Nv, self.Nk, self.Nk))
        print('inversion: %.3f s' % (timeit.default_timer() - start))
        start = timeit.default_timer()
        d_lm = - torch.bmm(preconditioner, pseudo_grad).view((self.Nv, self.Nk, 1))  # moving direction
        print('delta: %.3f s' % (timeit.default_timer() - start))
        start = timeit.default_timer()
        p_new = self.check_nan_par(self.p + self.lr * d_lm).view((self.Nv, self.Nk, 1))  # line search
        print('par update: %.3f s' % (timeit.default_timer() - start))
        start = timeit.default_timer()
        self.move(p_new)
        print('move: %.3f s' % (timeit.default_timer() - start))

    def check_nan_par(self, par_new):
        par_new[torch.isnan(par_new)] = self.p[torch.isnan(par_new)]
        par_new[par_new <= 0] = self.epsilon
        return par_new

    def move(self, p_new):
        """
        Moving to the next point.
        Saves in Optimizer class next coordinates
        """
        f_new, jac_new = self.function(p_new)  # function and jacobian
        #f_new[torch.isnan(f_new)] = self.epsilon
        r_new = (self.fData - f_new)
        c_new = self.getF(r_new)
        self.gain_numerator = (self.c - c_new)
        '''gain_divisor = 0.5* torch.bmm(d_lm.permute(0, 2, 1), 
                                     torch.einsum('n,nij->nij', (self.m, d_lm))-pseudo_grad) + self.epsilon
        gain = gain_numerator / gain_divisor
        print(gain, sep=' ', end=' | ', flush=True)
        '''
        idx_pos = (self.gain_numerator >  0).nonzero()
        self.not_converged = len(idx_pos)
        idx_neg = (self.gain_numerator <= 0).nonzero()
        self.m[idx_pos] *= 0.1  # max(1 / 3, 1 - (2 * gain - 1) ** 3)
        self.m[idx_neg] *= self.v
        self.m[self.m <= 1e-20] = 1e-20
        self.m[self.m  >  1e20] = 1e20
        self.f[idx_pos,:] = f_new[idx_pos,:]
        self.J[idx_pos,:,:] = jac_new[idx_pos,:,:]
        self.p[idx_pos,:] = p_new[idx_pos,:]
        self.c[idx_pos] = c_new[idx_pos]

    def check_convergence(self, tol=1e-4):
        gainmax = torch.max((self.gain_numerator))
        cmax = torch.sqrt(torch.max(self.c))
        # print(gainmax, sep=' ', end=' | ', flush=True)
        print(cmax, sep=' ', end=' | ', flush=True)
        # print(gainmax / cmax, sep=' ', end='\n', flush=True)
        return gainmax / cmax < tol

    def run(self, maxit=100, tol=1e-4):
        currentIteration = 0
        self.exitflag = 1
        start = timeit.default_timer()
        while currentIteration <= maxit:
            print('|', sep=' ', end='', flush=True)
            self.step()
            currentIteration += 1

            if torch.isnan(self.m).nonzero().shape[0]:
                self.exitflag = 2
                break
            if torch.isinf(self.m).nonzero().shape[0]:
                self.exitflag = 3
                break
            if self.not_converged == 0:
                self.exitflag = 0
                break
            if self.check_convergence(tol):
                self.exitflag = 0
                break

        elapsed = timeit.default_timer() - start
        print(f'\nFitting completed after %i iterations. \nElapsed time: %.3f s\n' % (currentIteration, elapsed))
