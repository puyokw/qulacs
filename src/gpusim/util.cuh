#ifndef _QCUDASIM_UTIL_CUH_
#define _QCUDASIM_UTIL_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <cuda_runtime.h>
//#include <cuda.h>

#ifdef __cplusplus
#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <complex>
//#include <sys/time.h>
#else
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#endif

#include <cuComplex.h>

#include "util.h"
#include "util_common.h"

inline void checkCudaErrors(const cudaError error)
{
	if (error != cudaSuccess){
		printf("Error: %s:%d, ", __FILE__, __LINE__);
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));
		exit(1);
	}
}

inline void memcpy_quantum_state_HostToDevice(CTYPE* state_cpu, GTYPE* state_gpu, ITYPE dim){
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&state_gpu, dim * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(state_gpu, state_cpu, dim * sizeof(CTYPE), cudaMemcpyHostToDevice));
}

__global__ void init_qstate(GTYPE* state_gpu, ITYPE dim){
	ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < dim) {
		state_gpu[idx] = make_cuDoubleComplex(0.0, 0.0);
	}
	if (idx == 0) state_gpu[idx] = make_cuDoubleComplex(1.0, 0.0);
}
/*
inline void initialize_quantum_state(GTYPE *state, ITYPE dim){
	GTYPE* state_gpu = reinterpret_cast<cuDoubleComplex*>(state);
	cudaError cudaStatus;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	init_qstate << <grid, block >> >(state_gpu, dim);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "init_state_gpu failed: %s\n", cudaGetErrorString(cudaStatus));
	}
}
*/
inline void release_quantum_state(GTYPE* state_gpu){
	//GTYPE* state_gpu = reinterpret_cast<cuDoubleComplex*>(state);
	cudaFree(state_gpu);
}
/*
CTYPE* get_quantum_state(void* state_gpu, ITYPE dim){
	GTYPE* psi_gpu = reinterpret_cast<GTYPE*>(state_gpu);
	CTYPE* state_cpu=(CTYPE*)malloc(sizeof(CTYPE)*dim);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, psi_gpu, dim * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	state_gpu = reinterpret_cast<void*>(psi_gpu);
	return state_cpu;
}
*/
inline void print_quantum_state(GTYPE* state_gpu, ITYPE dim){
	CTYPE* state_cpu=(CTYPE*)malloc(sizeof(CTYPE)*dim);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, state_gpu, dim * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	for(int i=0;i<dim;++i){
		std::cout << i << " : " << state_cpu[i].real() << "+i" << state_cpu[i].imag() << '\n'; 
	}
	std::cout << '\n';
	free(state_cpu);
}

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

inline __device__ int popcount64(ITYPE b) {
	b -= (b >> 1) & 0x5555555555555555ULL;
	b = ((b >> 2) & 0x3333333333333333ULL) + (b & 0x3333333333333333ULL);
	b = ((b >> 4) + b) & 0x0F0F0F0F0F0F0F0FULL;
	return (b * 0x0101010101010101ULL) >> 56;
}

inline __device__ int popcount32(unsigned int b) {
	unsigned int w = b >> 32;
	unsigned int v = b;
	v -= (v >> 1) & 0x55555555;
	w -= (w >> 1) & 0x55555555;
	v = ((v >> 2) & 0x33333333) + (v & 0x33333333);
	w = ((w >> 2) & 0x33333333) + (w & 0x33333333);
	v = ((v >> 4) + v + (w >> 4) + w) & 0x0F0F0F0F;
	return (v * 0x01010101) >> 24;
}

inline __device__ double atomicAdd_double(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

inline __device__ double __shfl_down_double(double var, unsigned int srcLane, int width = 32) {
	int2 a = *reinterpret_cast<int2*>(&var);
	a.x = __shfl_down(a.x, srcLane, width);
	a.y = __shfl_down(a.y, srcLane, width);
	return *reinterpret_cast<double*>(&a);
}

inline __device__ int warpReduceSum(int val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}

inline __device__ double warpReduceSum_double(double val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_down_double(val, offset);
	return val;
}

__global__ void deviceReduceWarpAtomicKernel(int *in, int* out, ITYPE N) {
	int sum = int(0);
	for (ITYPE i = blockIdx.x * blockDim.x + threadIdx.x;
		i < N;
		i += blockDim.x * gridDim.x) {
		sum += in[i];
	}
	sum = warpReduceSum(sum);
	if ((threadIdx.x & (warpSize - 1)) == 0)
		atomicAdd(out, sum);
}

__global__ void deviceReduceWarpAtomicKernel(double *in, double* out, ITYPE N) {
	double sum = double(0.0);
	for (ITYPE i = blockIdx.x * blockDim.x + threadIdx.x;
		i < N;
		i += blockDim.x * gridDim.x) {
		sum += in[i];
	}
	sum = warpReduceSum_double(sum);
	if ((threadIdx.x & (warpSize - 1)) == 0)
		atomicAdd_double(out, sum);
}

__global__ void deviceReduceWarpAtomicKernel(GTYPE *in, GTYPE* out, ITYPE N) {
	GTYPE sum = make_cuDoubleComplex(0.0, 0.0);
	for (ITYPE i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
		sum = cuCadd(sum, in[i]);
	}
	sum.x = warpReduceSum_double(sum.x);
	sum.y = warpReduceSum_double(sum.y);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(out[0].x), sum.x);
		atomicAdd_double(&(out[0].y), sum.y);
	}
}

inline __device__ void deviceReduceWarpAtomicKernel_device(GTYPE *in, GTYPE* out, ITYPE N) {
	GTYPE sum = make_cuDoubleComplex(0.0, 0.0);
	for (ITYPE i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
		sum = cuCadd(sum, in[i]);
	}
	sum.x = warpReduceSum_double(sum.x);
	sum.y = warpReduceSum_double(sum.y);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(out[0].x), sum.x);
		atomicAdd_double(&(out[0].y), sum.y);
	}
}

inline __device__ ITYPE insert_zero_to_basis_index_device(ITYPE basis_index, unsigned int qubit_index){
    // ((basis_index >> qubit_index) << (qubit_index+1) )+ (basis_index % basis_mask)
	ITYPE temp_basis = (basis_index >> qubit_index) << (qubit_index+1);
    return (temp_basis + (basis_index & ( (1ULL<<qubit_index) -1)));
}

#endif // #ifndef _QCUDASIM_UTIL_CUH_
