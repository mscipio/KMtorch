#include <stdio.h>
// Macro for converting subscripts to linear index:
#define INDEX_VOL_TIME(i, t) i*${Nt}+t
#define INDEX_JAC_TIME(i, t, p) i*${Nt}*${Nk}+t*${Nk}+p
#define INDEX_PARAM(i, p) i*${Nk}+p
#define INDEX_MASK(x, y, z) x*${W}*${L}+y*${L}+z
#define eps 1e-9

/******************************************************************************************************************************* 
                          MODEL FUNCTIONS DECLARATION (see the end of this file for the body of the functions)
*******************************************************************************************************************************/
__device__ void bicomp_3expIF(unsigned int idx, float *aux_par, float *ifparams, float *IF, float *times, float *func, float *jac, float *dk, float *mask);
__device__ void bicomp_2expIF(unsigned int idx, float *aux_par, float *ifparams, float *IF, float *times, float *func, float *jac, float *dk, float *mask);

/******************************************************************************************************************************* 
                                                       PET
*******************************************************************************************************************************/

// BICOMPARTMENT MODEL WITH IF MODELED AS SUM OF 3 EXP (like in Feng model #2)
__global__ void bicompartment_3expIF_4k(float *aux_par, float *inputfunpar, float *IF, float *time, float *func, float *jac, float *dk, float *mask)
{ 
    // Obtain the linear index corresponding to the current thread:
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    //*** Uncomment the lines below to use shared memory 
    __shared__ float times[${Nt}];
    __shared__ float ifparams[7];
    if (threadIdx.x < ${Nt}) {
            times[threadIdx.x] = time[threadIdx.x]; 
            if (threadIdx.x < 7) 
                ifparams[threadIdx.x] = inputfunpar[threadIdx.x];
    }
    __syncthreads();
    //*** Comment the lines above and uncomment below to disable shared memory 
    //float *times = time;
    //float *ifparams = inputfun; 
    //*** 
    
    bicomp_3expIF(idx, aux_par, ifparams, IF, times, func, jac, dk, mask);
    //__syncthreads();
}

// BICOMPARTMENT MODEL WITH IF MODELED AS SUM OF 3 EXP (like in Feng model #2)
__global__ void bicompartment_3expIF_3k(float *aux_par, float *inputfunpar, float *IF, float *time, float *func, float *jac, float *dk, float *mask)
{ 
    // Obtain the linear index corresponding to the current thread:
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    //*** Uncomment the lines below to use shared memory 
    __shared__ float times[${Nt}];
    __shared__ float ifparams[7];
    if (threadIdx.x < ${Nt}) {
            times[threadIdx.x] = time[threadIdx.x]; 
            if (threadIdx.x < 7) 
                ifparams[threadIdx.x] = inputfunpar[threadIdx.x];
    }
    __syncthreads();
    //*** Comment the lines above and uncomment below to disable shared memory 
    //float *times = time;
    //float *ifparams = inputfun; 
    //*** 
    
    bicomp_3expIF(idx, aux_par, ifparams, IF, times, func, jac, dk, mask);
    //__syncthreads();
    
    // deactivate the jacobian for the 4th kinetic constant we don't want to update
    for (uint tt=0; tt<${Nt}; ++tt) {
	    jac[INDEX_JAC_TIME(idx,tt,4)] = 0.0;          
    }
    //__syncthreads();
}

// MONOOMPARTMENT MODEL WITH IF MODELED AS SUM OF 2 EXP (like in Feng model #4)
__global__ void monocompartment_3expIF(float *aux_par, float *inputfunpar, float *IF, float *time, float *func, float *jac, float *dk, float *mask)
{ 
    // Obtain the linear index corresponding to the current thread:
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    //*** Uncomment the lines below to use shared memory 
    __shared__ float times[${Nt}];
    __shared__ float ifparams[5];
    if (threadIdx.x < ${Nt}) {
            times[threadIdx.x] = time[threadIdx.x]; 
            if (threadIdx.x < 5) 
                ifparams[threadIdx.x] = inputfunpar[threadIdx.x];
    }
    __syncthreads();
    //*** Comment the lines above and uncomment below to disable shared memory 
    //float *times = time;
    //float *ifparams = inputfun; 
    //*** 
    
    bicomp_3expIF(idx, aux_par, ifparams, IF, times, func, jac, dk, mask);
    //__syncthreads();
    
    // deactivate the jacobian for the 4th kinetic constant we don't want to update
    for (uint tt=0; tt<${Nt}; ++tt) {
	    jac[INDEX_JAC_TIME(idx,tt,3)] = 0; 
	    jac[INDEX_JAC_TIME(idx,tt,4)] = 0; 
    }
    //__syncthreads();
}

/******************************************************************************************************************************* 
                                                      DCE-MRI
*******************************************************************************************************************************/

// BICOMPARTMENT MODEL WITH IF MODELED AS SUM OF 2 EXP (like in Feng model #4)
__global__ void bicompartment_2expIF_4k(float *aux_par, float *inputfunpar, float *IF, float *time, float *func, float *jac, float *dk, float *mask)
{ 
    // Obtain the linear index corresponding to the current thread:
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    //*** Uncomment the lines below to use shared memory 
    __shared__ float times[${Nt}];
    __shared__ float ifparams[5];
    if (threadIdx.x < ${Nt}) {
            times[threadIdx.x] = time[threadIdx.x]; 
            if (threadIdx.x < 5) 
                ifparams[threadIdx.x] = inputfunpar[threadIdx.x];
    }
    __syncthreads();
    //*** Comment the lines above and uncomment below to disable shared memory 
    //float *times = time;
    //float *ifparams = inputfun; 
    //*** 
    
    bicomp_2expIF(idx, aux_par, ifparams, IF, times, func, jac, dk, mask);
    //__syncthreads();
}

// BICOMPARTMENT MODEL WITH IF MODELED AS SUM OF 2 EXP (like in Feng model #4)
__global__ void bicompartment_2expIF_3k(float *aux_par, float *inputfunpar, float *IF, float *time, float *func, float *jac, float *dk, float *mask)
{ 
    // Obtain the linear index corresponding to the current thread:
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    //*** Uncomment the lines below to use shared memory 
    __shared__ float times[${Nt}];
    __shared__ float ifparams[5];
    if (threadIdx.x < ${Nt}) {
            times[threadIdx.x] = time[threadIdx.x]; 
            if (threadIdx.x < 5) 
                ifparams[threadIdx.x] = inputfunpar[threadIdx.x];
    }
    __syncthreads();
    //*** Comment the lines above and uncomment below to disable shared memory 
    //float *times = time;
    //float *ifparams = inputfun; 
    //*** 
    
    bicomp_2expIF(idx, aux_par, ifparams, IF, times, func, jac, dk, mask);
    //__syncthreads();
    
    // deactivate the jacobian for the 4th kinetic constant we don't want to update
    for (uint tt=0; tt<${Nt}; ++tt) {
	    jac[INDEX_JAC_TIME(idx,tt,4)] = 0;           
    }
    //__syncthreads();
}

// MONOOMPARTMENT MODEL WITH IF MODELED AS SUM OF 2 EXP (like in Feng model #4)
__global__ void monocompartment_2expIF(float *aux_par, float *inputfunpar, float *IF, float *time, float *func, float *jac, float *dk, float *mask)
{ 
    // Obtain the linear index corresponding to the current thread:
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    //*** Uncomment the lines below to use shared memory 
    __shared__ float times[${Nt}];
    __shared__ float ifparams[5];
    if (threadIdx.x < ${Nt}) {
            times[threadIdx.x] = time[threadIdx.x]; 
            if (threadIdx.x < 5) 
                ifparams[threadIdx.x] = inputfunpar[threadIdx.x];
    }
    __syncthreads();
    //*** Comment the lines above and uncomment below to disable shared memory 
    //float *times = time;
    //float *ifparams = inputfun; 
    //*** 
    
    bicomp_2expIF(idx, aux_par, ifparams, IF, times, func, jac, dk, mask);
    //__syncthreads();
    
    // deactivate the jacobian for the 4th kinetic constant we don't want to update
    if (idx < ${Nv}) {
        for (uint tt=0; tt<${Nt}; ++tt) {
            jac[INDEX_JAC_TIME(idx,tt,3)] = eps;
            jac[INDEX_JAC_TIME(idx,tt,4)] = eps;
        }
    }
    //__syncthreads();
}


/******************************************************************************************************************************* 
                                                 COMPARTMENTAL MODELS IMPLEMENTATION
*******************************************************************************************************************************/

// ANALYTIC FORMULATION OF A BICOMPARTMENT MODEL WITH IF MODELED AS SUM OF 3 EXP (like in Feng model #2)
__device__ void bicomp_3expIF(unsigned int idx, float *aux_par, float *ifparams, float *IF, float *times, float *func, float *jac, float *dk, float *mask)
{     
    //float s;
    //float d;
    float delta0;
    float delta;
    float p[4];
    float Ahat[3];
    float Abar[3]; 
    float sum[${Nt}];
    float TAC[${Nt}];
    float Jb[${Nt}];
    float Jl[${Nt}];

    unsigned int x = idx/(${W}*${L});
    unsigned int y = (idx%(${W}*${L}))/${L};
    unsigned int z = (idx%(${W}*${L}))%${L};

    //__syncthreads();

    // Compute output of bicompartmental model and Jacobian using analytical expression.  
    if (idx < ${Nv}) {
        /* Auxiliary parameters
	
        s = k[INDEX_PARAM(idx,2)] + k[INDEX_PARAM(idx,3)] + k[INDEX_PARAM(idx,4)];
        d = abs(sqrt(s*s - 4*k[INDEX_PARAM(idx,2)]*k[INDEX_PARAM(idx,4)]));
        p[1] = (s + d) / 2;   //L1
        p[3] = (s - d) / 2;   //L2
        p[0] = (k[INDEX_PARAM(idx,1)] * ( p[1] - k[INDEX_PARAM(idx,3)] - k[INDEX_PARAM(idx,4)])) / d;  //B1
        p[2] = (k[INDEX_PARAM(idx,1)] * (-p[3] + k[INDEX_PARAM(idx,3)] + k[INDEX_PARAM(idx,4)])) / d;  //B2 */
	    p[1] = aux_par[INDEX_PARAM(idx,2)];
        p[3] = aux_par[INDEX_PARAM(idx,4)];
        p[0] = aux_par[INDEX_PARAM(idx,1)];
        p[2] = aux_par[INDEX_PARAM(idx,3)];
        Abar[0] = -ifparams[2]-ifparams[3]; 
        Abar[1] =  ifparams[2]; 
        Abar[2] =  ifparams[3];   
        
        for (uint tt=0; tt<${Nt}; ++tt) { // reset the values of TAC and JAC for current voxel/thread
            func[INDEX_VOL_TIME(idx,tt)] = 0;
            jac[INDEX_JAC_TIME(idx,tt,0)] = 0;
            jac[INDEX_JAC_TIME(idx,tt,1)] = 0;
            jac[INDEX_JAC_TIME(idx,tt,2)] = 0;
            jac[INDEX_JAC_TIME(idx,tt,3)] = 0;
            jac[INDEX_JAC_TIME(idx,tt,4)] = 0;
            jac[INDEX_JAC_TIME(idx,tt,5)] = 0;
            TAC[tt] = 0;
        }
        //__syncthreads();
        if ((idx*mask[INDEX_MASK(x, y, z)]!= 0) || (idx == 0 && mask[INDEX_MASK(x, y, z)]!= 0))
        {
        for (uint ii=0; ii<=2; ii+=2) {   //i = 1:2:4 % 2 compartiments
            delta0  = p[ii+1] + ifparams[4];
            Ahat[0] = -ifparams[2]-ifparams[3]-(ifparams[1]/(delta0 + eps));
            Ahat[1] = ifparams[2]; 
            Ahat[2] = ifparams[3]; 
          
            for (uint tt=0; tt<${Nt}; ++tt) { // reset temporary variables for i-th compartment
                sum[tt]=0;
                Jb[tt] =0;
                Jl[tt] =0;
            }
            for (uint jj=0; jj<3; ++jj) {
                delta  = p[ii+1]+ifparams[4+jj];
            
                for (uint tt=0; tt<${Nt}; ++tt) { 
                    if (times[tt]>=ifparams[0]) {
                        sum[tt] += Ahat[jj] * (1.0f / (delta + eps)) * ( exp(ifparams[4+jj]*(times[tt]-ifparams[0]))-exp(-p[ii+1]*(times[tt]-ifparams[0])) );
                        Jb[tt]  += Ahat[jj] * (1.0f / (delta + eps)) * ( exp(ifparams[4+jj]*(times[tt]-ifparams[0]))-exp(-p[ii+1]*(times[tt]-ifparams[0])) );
                        Jl[tt]  += Abar[jj] * (1.0f / (delta*delta + eps)) * ( exp(-p[ii+1]*(times[tt]-ifparams[0]))-exp(ifparams[4+jj]*(times[tt]-ifparams[0]))) + Abar[jj] * (1.0f / (delta + eps))*(times[tt]-ifparams[0]) * exp(-p[ii+1]*(times[tt]-ifparams[0]));
                    }
                }
            }
            
            for (uint tt=0; tt<${Nt}; ++tt) {
                if (times[tt]>=ifparams[0]) {
                    TAC[tt] += p[ii] * (sum[tt] + ((ifparams[1]*(times[tt]-ifparams[0])) / (delta0 + eps)) *exp(ifparams[4]*(times[tt]-ifparams[0])))* exp(-dk[0]*(times[tt]-ifparams[0]));
                    jac[INDEX_JAC_TIME(idx,tt,ii+1)] = -(1-aux_par[INDEX_PARAM(idx,0)]) * (Jb[tt]
                                                                                              + ((ifparams[1]*(times[tt]-ifparams[0])) / (delta0 + eps))
                                                                                                   *exp(ifparams[4]*(times[tt]-ifparams[0]))
                                                                                         )* exp(-dk[0]*(times[tt]-ifparams[0]));
                    jac[INDEX_JAC_TIME(idx,tt,ii+2)] = -(1-aux_par[INDEX_PARAM(idx,0)]) * (p[ii] * (Jl[tt]
                                                                                                        - ifparams[1] * (times[tt]-ifparams[0]) * (1.0f / (delta0*delta0 + eps))
                                                                                                              * ( exp(ifparams[4]*(times[tt]-ifparams[0]))  + exp(-p[ii+1]*(times[tt]-ifparams[0])) )
                                                                                                        + 2*ifparams[1] * (1.0f / (delta0*delta0*delta0 + eps))
                                                                                                              * ( exp(ifparams[4]*(times[tt]-ifparams[0]))  - exp(-p[ii+1]*(times[tt]-ifparams[0])) )
                                                                                                  )
                                                                                         )* exp(-dk[0]*(times[tt]-ifparams[0]));
                    
                    /*jac[INDEX_JAC_TIME(idx,tt,ii+1)] = (1-aux_par[INDEX_PARAM(idx,0)]) * (Jb[tt] + ((ifparams[1]*(times[tt]-ifparams[0]))/delta0) *exp(ifparams[4]*(times[tt]-ifparams[0])));
                    jac[INDEX_JAC_TIME(idx,tt,ii+2)] = (1-aux_par[INDEX_PARAM(idx,0)]) * (p[ii] * (Jl[tt] 
                                                                    + ( exp(-p[ii+1]*(times[tt]-ifparams[0]))-exp(ifparams[4]*(times[tt]-ifparams[0]))) 
                                                                           * (ifparams[1] *(times[tt]-ifparams[0]) * (1.0f / (delta0*delta0)) 
                                                                    + 2*ifparams[1] * (1.0f / (delta0*delta0*delta0))) ));*/
                }
            }
            
        }

        //__syncthreads();
        for (uint tt=0; tt<${Nt}; ++tt) {
            jac[INDEX_JAC_TIME(idx,tt,0)] = -IF[tt] + TAC[tt];
            TAC[tt]  = ((1-aux_par[INDEX_PARAM(idx,0)]) * TAC[tt]) + (aux_par[INDEX_PARAM(idx,0)] * IF[tt]);
            if (TAC[tt] < 0.0) {
                    TAC[tt] = 1e-16;
            } 
            func[INDEX_VOL_TIME(idx,tt)] = TAC[tt];            
        }
        //__syncthreads();
        }
    }
}

// ANALYTIC FORMULATION OF A BICOMPARTMENT MODEL WITH IF MODELED AS SUM OF 2 EXP (like in Feng model #4) -- NO DECAY CORRECTION FOR DCE-MRI
__device__ void bicomp_2expIF(unsigned int idx, float *aux_par, float *ifparams, float *IF, float *times, float *func, float *jac, float *dk, float *mask)
{
    float p[4];
    float Ahat[2];
    float Abar[2]; 
    float sum[${Nt}];
    float TAC[${Nt}];
    float Jb[${Nt}];
    float Jl[${Nt}];
    float Jd[${Nt}];
    float delay;
	float fv;
    
    unsigned int x = idx/(${W}*${L});
    unsigned int y = (idx%(${W}*${L}))/${L};
    unsigned int z = (idx%(${W}*${L}))%${L};
   // __syncthreads();

    // Compute output of bicompartmental model and Jacobian using analytical expression.  
    if (idx < ${Nv}) {
        // Auxiliary parameters
        p[1] = aux_par[INDEX_PARAM(idx,2)];
        p[3] = aux_par[INDEX_PARAM(idx,4)];
        p[0] = aux_par[INDEX_PARAM(idx,1)];
        p[2] = aux_par[INDEX_PARAM(idx,3)];
        Abar[0] = -ifparams[2]; 
        Abar[1] =  ifparams[2];
        fv = aux_par[INDEX_PARAM(idx,0)];
        //15Dec218 modified to allow estimate of a specific delay for each TAC (different from IF delay ifparams[0])
        if (aux_par[INDEX_PARAM(idx,5)] < ifparams[0]){
            aux_par[INDEX_PARAM(idx,5)] = ifparams[0];
        }
        delay = aux_par[INDEX_PARAM(idx,5)];
        //delay = ifparams[0];
        
        for (uint tt=0; tt<${Nt}; ++tt) { // reset the values of TAC and JAC for current voxel/thread
            for (uint kk=0; kk<${Nk}; ++kk) { // reset the values of TAC and JAC for current voxel/thread
                jac[INDEX_JAC_TIME(idx,tt,kk)] = eps;
            }
            func[INDEX_VOL_TIME(idx,tt)]  = eps;
            TAC[tt] = eps;
        }
        //__syncthreads();

        if ((idx*mask[INDEX_MASK(x, y, z)]!= 0) || (idx == 0 && mask[INDEX_MASK(x, y, z)]!= 0))
        {
          for (uint ii=0; ii<=2; ii+=2) {   //i = 1:2:4 % 2 compartiments
              float l1 = -ifparams[3];
			  float A1 = ifparams[1];
              float delta0  = 1.0f / (p[ii+1] - l1 + eps) ;
              Ahat[0] = -ifparams[2]-(A1 * delta0);
              Ahat[1] = ifparams[2];

              for (uint tt=0; tt<${Nt}; ++tt) { // reset temporary variables for i-th compartment
                  sum[tt]= eps;
                  Jb[tt] = eps;
                  Jl[tt] = eps;
                  Jd[tt] = eps;
              }
              for (uint jj=0; jj<2; ++jj) {
                  float l = -ifparams[3+jj];
				  float delta  = 1.0f / (p[ii+1] - l + eps);

                  for (uint tt=0; tt<${Nt}; ++tt) {
                      if (times[tt]>=delay) {
                          float t = times[tt] - delay;
                          sum[tt] += Ahat[jj] * delta * ( exp(-l*t)-exp(-p[ii+1]*t) );
                          Jb[tt]  += Ahat[jj] * delta * ( exp(-l*t)-exp(-p[ii+1]*t) );
                          Jl[tt]  += Abar[jj] * delta * (delta * ( exp(-p[ii+1]*t)-exp(-l*t)) +  t*exp(-p[ii+1]*t) );
                          Jd[tt]  += Ahat[jj] * delta * (l*exp(-l*t) - p[ii+1]*exp(-p[ii+1]*t) );
                      }
                  }
              }

              // TODO TRIPLE CHECK THESE DERIVATIVES
              for (uint tt=0; tt<${Nt}; ++tt) {
                  if (times[tt]>=delay) {
                      float t = times[tt] - delay;
                      TAC[tt] += p[ii] * (sum[tt] + (A1*t*delta0) *exp(-l1*t))* exp(-dk[0]*t);
                      jac[INDEX_JAC_TIME(idx,tt,ii+1)] = -(1-fv) * (Jb[tt] + (A1*t*delta0 *exp(-l1*t)))* exp(-dk[0]*t);
                      jac[INDEX_JAC_TIME(idx,tt,ii+2)] = -(1-fv) * (p[ii] * (Jl[tt]
                                                                               - ( exp(-p[ii+1]*t)+exp(-l1*t)) * A1 * t * delta0 * delta0
                                                                               + (-exp(-p[ii+1]*t)+exp(-l1*t)) * 2*A1 * delta0 * delta0 * delta0
                                                                           )
                                                                   ) * exp(-dk[0]*t);
                      jac[INDEX_JAC_TIME(idx,tt,5)]   += -(1-fv) * (p[ii] * ( Jd[tt] + (A1 * delta0 * exp(-l1*t) * (l1*t - 1 ))
                                                                            )
                                                                   )* exp(-dk[0]*t);
                      //Jd[tt]  += p[ii] * ( Jd[tt] + (A1 * delta0 * exp(-l1*t) * (l1*t - 1 ))); //TODO: CHECK THIS
                  }
              }

          }
          //__syncthreads();
          for (uint tt=0; tt<${Nt}; ++tt) {
              if (times[tt] >= delay) {
                  //float t = times[tt] - delay;
                  jac[INDEX_JAC_TIME(idx,tt,0)] = -IF[tt] + TAC[tt];
                  //jac[INDEX_JAC_TIME(idx,tt,5)] = -(1-fv) * Jd[tt] * exp(-dk[0]*t);
                  TAC[tt]  = ((1-fv) * TAC[tt]) + (fv * IF[tt]);
                  if (TAC[tt] < eps) {
                      TAC[tt] = eps;
                  }
              }
              func[INDEX_VOL_TIME(idx,tt)] = TAC[tt];
          }
          //__syncthreads();
	    }
	    else{
		    aux_par[INDEX_PARAM(idx,0)] = eps;
  		    aux_par[INDEX_PARAM(idx,1)] = eps;
  		    aux_par[INDEX_PARAM(idx,2)] = eps;
  		    aux_par[INDEX_PARAM(idx,3)] = eps;
  		    aux_par[INDEX_PARAM(idx,4)] = eps;
  		    aux_par[INDEX_PARAM(idx,5)] = eps;
		}
    }
}

