#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
//#include "util.h"
#include "util.cuh"
#include "util_common.h"
#include "update_ops_cuda.h"

__global__ void H_gate_gpu(unsigned int target_qubit_index, GTYPE *psi, ITYPE dim) {
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;
	ITYPE basis0, basis1;
	GTYPE tmp;

	if (j < (dim >> 1)){
		//basis0 = ((j & ~((ONE<< i)-1)) << 1) + (j & ((ONE<< i)-1));
		//basis1 = basis0 + (ONE<< i);
		basis0 = (j >> target_qubit_index);
		basis0 = basis0 << (target_qubit_index + 1);
		basis0 += j & ((1ULL << target_qubit_index) - 1);
		basis1 = basis0 ^ (1ULL << target_qubit_index);

		tmp = psi[basis0];
		psi[basis0] = cuCadd(tmp, psi[basis1]);
		psi[basis1] = cuCadd(tmp, make_cuDoubleComplex(-1*psi[basis1].x, -1*psi[basis1].y));
		psi[basis0] = make_cuDoubleComplex(psi[basis0].x / sqrt(2.0), psi[basis0].y / sqrt(2.0));
		psi[basis1] = make_cuDoubleComplex(psi[basis1].x / sqrt(2.0), psi[basis1].y / sqrt(2.0));
	}
}

extern "C"
__host__ cudaError H_gate_host(unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE dim){
	cudaError cudaStatus;
	ITYPE half_dim = dim >> 1;
	int block = half_dim <= 1024 ? half_dim : 1024;
	int grid = dim / block;

	H_gate_gpu << <grid, block >> >(target_qubit_index, psi_gpu, dim);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "H_gate_host launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

extern "C"
void H_gate(unsigned int target_qubit_index, void *psi, ITYPE DIM){
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(psi);
	GTYPE *psi_gpu;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, DIM * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, state_cpu, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));

	H_gate_host(target_qubit_index, psi_gpu, DIM);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "H_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, psi_gpu, DIM * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	psi = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(psi_gpu);
	//free(state_cpu);
}

__global__ void X_gate_gpu(unsigned int target_qubit_index, GTYPE *psi, ITYPE dim) {
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;
	ITYPE basis0, basis1;
	GTYPE tmp;

	if (j < (dim>>1)){
		//basis0 = ((j & ~((ONE<< i)-1)) << 1) + (j & ((ONE<< i)-1));
		//basis1 = basis0 + (ONE<< i);
		basis0 = (j >> target_qubit_index);
		basis0 = basis0 << (target_qubit_index + 1);
		basis0 += j & ((1ULL << target_qubit_index) - 1);
		basis1 = basis0 ^ (1ULL << target_qubit_index);

		tmp = psi[basis0];
		psi[basis0] = psi[basis1];
		psi[basis1] = tmp;
    }
}

extern "C"
__host__ cudaError X_gate_host(unsigned int target_qubit_index, void *state_gpu, ITYPE dim){
	GTYPE* psi_gpu = reinterpret_cast<GTYPE*>(state_gpu);
    cudaError cudaStatus;
	ITYPE half_dim = dim >> 1;
	int block = half_dim <= 1024 ? half_dim : 1024;
	int grid = dim / block;
    
	X_gate_gpu << <grid, block >> >(target_qubit_index, psi_gpu, dim);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "X_gate_host launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	state_gpu = reinterpret_cast<void*>(psi_gpu);
	return cudaStatus;
}

extern "C"
void X_gate(unsigned int target_qubit_index, void *psi, ITYPE dim){
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(psi);
	GTYPE *psi_gpu;
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, dim * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, state_cpu, dim * sizeof(CTYPE), cudaMemcpyHostToDevice));

	X_gate_host(target_qubit_index, psi_gpu, dim);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "X_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, psi_gpu, dim * sizeof(CTYPE), cudaMemcpyDeviceToHost));
    psi = reinterpret_cast<void*>(state_cpu);
	
Error:
	cudaFree(psi_gpu);
	//free(state_cpu);
}

__global__ void Y_gate_gpu(unsigned int target_qubit_index, GTYPE *psi, ITYPE dim) {
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;
	ITYPE basis0, basis1;
	GTYPE tmp;

	if (j < (dim>>1)){
		basis0 = (j >> target_qubit_index);
		basis0 = basis0 << (target_qubit_index + 1);
		basis0 += j & ((1ULL << target_qubit_index) - 1);
		basis1 = basis0 ^ (1ULL << target_qubit_index);

		tmp = psi[basis0];
		psi[basis0] = make_cuDoubleComplex(cuCimag(psi[basis1]), -cuCreal(psi[basis1]));
		psi[basis1] = make_cuDoubleComplex(-cuCimag(tmp), cuCreal(tmp));
	}
}

extern "C"
__host__ cudaError Y_gate_host(unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE dim){
	cudaError cudaStatus;
	ITYPE half_dim = dim >> 1;
	int block = half_dim <= 1024 ? half_dim : 1024;
	int grid = dim / block;

	Y_gate_gpu << <grid, block >> >(target_qubit_index, psi_gpu, dim);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Y_gate_host launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

extern "C"
void Y_gate(unsigned int target_qubit_index, void *psi, ITYPE dim){
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(psi);
	GTYPE *psi_gpu;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, dim * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, state_cpu, dim * sizeof(CTYPE), cudaMemcpyHostToDevice));

	Y_gate_host(target_qubit_index, psi_gpu, dim);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Y_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, psi_gpu, dim * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	psi = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(psi_gpu);
	//free(state_cpu);
}

__global__ void Z_gate_gpu(unsigned int target_qubit_index, GTYPE *psi, ITYPE DIM) {
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;
	ITYPE basis0, basis1;
	if (j < (DIM>>1)){
		basis0 = insert_zero_to_basis_index_device(j, target_qubit_index);
		basis1 = basis0^(1ULL<<target_qubit_index);
		psi[basis1] = make_cuDoubleComplex(-cuCreal(psi[basis1]), -cuCimag(psi[basis1]));
	}
}

extern "C"
__host__ cudaError Z_gate_host(unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE dim){
	cudaError cudaStatus;
	ITYPE half_dim = dim >> 1;
	int block = half_dim <= 1024 ? half_dim : 1024;
	int grid = dim / block;

	Z_gate_gpu << <grid, block >> >(target_qubit_index, psi_gpu, dim);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Z_gate_host launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

extern "C"
void Z_gate(unsigned int target_qubit_index, void *psi, ITYPE DIM){
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(psi);
	GTYPE *psi_gpu;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, DIM * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, state_cpu, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));

	Z_gate_host(target_qubit_index, psi_gpu, DIM);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Z_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, psi_gpu, DIM * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	psi = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(psi_gpu);
	//free(state_cpu);
}
