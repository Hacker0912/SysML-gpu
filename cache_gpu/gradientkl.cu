#ifndef _GRADIENTKL_H_
#define _GRADIENTKL_H_

#include <cuda.h>
#include "math.h"


__global__ void G_lrcache(double* dY, double* dH, double* cuda_cache, double* dmul, int cur_index, long row_num, int pitch)
{
    // F_partial += gradientCompute(Y[i],H[i],lm)*cache[cur_index][i];    
    long Idx =  blockIdx.x * blockDim.x + threadIdx.x;
    if(Idx < row_num) {    
        dmul[Idx] = -(dY[Idx] / (1 + exp(dY[Idx] * dH[Idx])))*cuda_cache[cur_index*pitch/sizeof(double) + Idx];        
    	//dmul[Idx] = -(dY[Idx] / (1 + exp(dY[Idx] * dH[Idx])))* (*(cuda_cache + Idx*pitch/sizeof(double)+cur_index));
	}
}

__global__ void G_lrkl(double* dY, double* dH, double* dX, double* dmul, long row_num)
{
    // F_partial += gradientCompute(Y[i],H[i],lm)*dX[i];    
    long Idx =  blockIdx.x * blockDim.x + threadIdx.x;
    if(Idx < row_num) {
        dmul[Idx] = -(dY[Idx] / (1 + exp(dY[Idx] * dH[Idx])))* dX[Idx];        
    }
}

__global__ void G_lrloss(double* dY, double* dH, double* dmul, long row_num)
{
    // F += lossCompute(Y[i],H[i],lm);
    long Idx =  blockIdx.x * blockDim.x + threadIdx.x;   
    if(Idx < row_num) {
        dmul[Idx] = log(1+exp(-dY[Idx]*dH[Idx]));
    } 
}

__global__ void H_cache(double* dH, double* cuda_cache, double diff, int cur_index, int pitch, long row_num)
{
    long Idx =  blockIdx.x * blockDim.x + threadIdx.x;
    if(Idx < row_num) { 
        dH[Idx] = dH[Idx] + diff * cuda_cache[cur_index*pitch/sizeof(double) + Idx];        
    }    
}

__global__ void Hkl(double* dH, double* dX, double diff, long row_num)
{
    long Idx =  blockIdx.x * blockDim.x + threadIdx.x;
    if(Idx < row_num) {
        dH[Idx] = dH[Idx] + diff * dX[Idx];        
    }    
}

#endif
