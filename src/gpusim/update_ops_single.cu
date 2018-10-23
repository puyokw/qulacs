#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
//#include "util.h"
#include "util.cuh"
#include "util_common.h"
#include "update_ops_cuda.h"


__device__ void single_qubit_dense_matrix_gate_device(unsigned int target_qubit_index, GTYPE matrix[4], GTYPE *state, ITYPE dim){
	ITYPE basis0, basis1;
	ITYPE half_dim = dim >> 1;
	GTYPE tmp;
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;

	if (j < half_dim){
		//basis0 = ((j & ~((ONE<< i)-1)) << 1) + (j & ((ONE<< i)-1));
		//basis1 = basis0 + (ONE<< i);
		basis0 = (j >> target_qubit_index);
		basis0 = basis0 << (target_qubit_index + 1);
		basis0 += j & ((1LL << target_qubit_index) - 1);
		basis1 = basis0 ^ (1LL << target_qubit_index);

		tmp = state[basis0];
		state[basis0] = cuCadd(cuCmul(matrix[0], tmp), cuCmul(matrix[1], state[basis1]));
		state[basis1] = cuCadd(cuCmul(matrix[2], tmp), cuCmul(matrix[3], state[basis1]));
	}
}

__global__ void single_qubit_dense_matrix_gate_gpu(unsigned int target_qubit_index,GTYPE matrix[4], GTYPE *state_gpu, ITYPE dim){
	single_qubit_dense_matrix_gate_device(target_qubit_index, matrix, state_gpu, dim);
}

__host__ cudaError single_qubit_dense_matrix_gate_host(unsigned int target_qubit_index, CTYPE matrix[4], GTYPE *state_gpu, ITYPE dim) {
	GTYPE* matrix_gpu;
	cudaError cudaStatus;
	checkCudaErrors(cudaMalloc((void**)&matrix_gpu, 4 * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(matrix_gpu, matrix, 4 * sizeof(CTYPE), cudaMemcpyHostToDevice));

	ITYPE half_dim = dim >> 1;
	unsigned int block = half_dim <= 1024 ? half_dim : 1024;
	unsigned int grid = dim / block;
	single_qubit_dense_matrix_gate_gpu << <grid, block >> >(target_qubit_index, matrix_gpu, state_gpu, dim);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "single_qubit_dense_matrix_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	cudaFree(matrix_gpu);
	return cudaStatus;
}

//extern "C" DllExport void single_qubit_dense_matrix_gate(unsigned int target_qubit_index, const CTYPE matrix[4], CTYPE *state, ITYPE dim);
void single_qubit_dense_matrix_gate(unsigned int target_qubit_index, CTYPE matrix[4], void *state, ITYPE DIM)
{
	GTYPE *state_gpu;
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(state);
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&state_gpu, DIM * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(state_gpu, state_cpu, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));

	cudaStatus = single_qubit_dense_matrix_gate_host(target_qubit_index, matrix, state_gpu, DIM);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "single_qubit_dense_matrix_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, state_gpu, DIM * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	state = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(state_gpu);
}

__device__ void single_qubit_diagonal_matrix_gate_device(unsigned int target_qubit_index, GTYPE diagonal_matrix[2], GTYPE *state, ITYPE dim) {
    ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(state_index<dim){
		state[state_index] = cuCmul(diagonal_matrix[(state_index >> target_qubit_index) & 1], state[state_index]);
	}
}

__global__ void single_qubit_diagonal_matrix_gate_gpu(unsigned int target_qubit_index,GTYPE matrix[2], GTYPE *state_gpu, ITYPE dim) {
	single_qubit_diagonal_matrix_gate_device(target_qubit_index, matrix, state_gpu, dim);
}

__host__ cudaError single_qubit_diagonal_matrix_gate_host(unsigned int target_qubit_index, const CTYPE diagonal_matrix[2], GTYPE *state_gpu, ITYPE dim) {
	GTYPE* diagonal_matrix_gpu;
	cudaError cudaStatus;
	checkCudaErrors(cudaMalloc((void**)&diagonal_matrix_gpu, 2 * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(diagonal_matrix_gpu, diagonal_matrix, 2 * sizeof(CTYPE), cudaMemcpyHostToDevice));

	//ITYPE half_dim = dim >> 1;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	single_qubit_diagonal_matrix_gate_gpu << <grid, block >> >(target_qubit_index, diagonal_matrix_gpu, state_gpu, dim);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "single_qubit_diagonal_matrix_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	cudaFree(diagonal_matrix_gpu);
	return cudaStatus;
}

void single_qubit_diagonal_matrix_gate(unsigned int target_qubit_index, const CTYPE diagonal_matrix[2], void *state, ITYPE dim) {
	GTYPE *state_gpu;
	CTYPE *state_cpu = reinterpret_cast<CTYPE*>(state);
	cudaError cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&state_gpu, dim * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(state_gpu, state_cpu, dim * sizeof(CTYPE), cudaMemcpyHostToDevice));

	cudaStatus = single_qubit_diagonal_matrix_gate_host(target_qubit_index, diagonal_matrix, state_gpu, dim);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "single_qubit_diagonal_matrix_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, state_gpu, dim * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	state = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(state_gpu);
}

__device__ void single_qubit_control_single_qubit_dense_matrix_gate_device(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, GTYPE matrix[4], GTYPE *state, ITYPE dim) {
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	//state[state_index] = cuCmul(diagonal_matrix[(state_index >> target_qubit_index) & 1], state[state_index]);
    const ITYPE loop_dim = dim>>2;
    // target mask
    const ITYPE target_mask = 1ULL << target_qubit_index;
    const ITYPE control_mask = (1ULL << control_qubit_index) * control_value;
    // insert index
    const unsigned int min_qubit_index = (control_qubit_index<target_qubit_index) ? control_qubit_index : target_qubit_index;
    const unsigned int max_qubit_index = (control_qubit_index>target_qubit_index) ? control_qubit_index : target_qubit_index;

	if(state_index<loop_dim){
        // create base index
		ITYPE basis_c_t0 = state_index;

        basis_c_t0 = insert_zero_to_basis_index_device(basis_c_t0, min_qubit_index);
        basis_c_t0 = insert_zero_to_basis_index_device(basis_c_t0, max_qubit_index);
        // flip control
        basis_c_t0 ^= control_mask;
        // gather index
        ITYPE basis_c_t1 = basis_c_t0 ^ target_mask;
        // fetch values
        GTYPE cval_c_t0 = state[basis_c_t0];
        GTYPE cval_c_t1 = state[basis_c_t1];
        // set values
        state[basis_c_t0] = cuCadd(cuCmul(matrix[0], cval_c_t0), cuCmul(matrix[1], cval_c_t1));
        state[basis_c_t1] = cuCadd(cuCmul(matrix[2], cval_c_t0), cuCmul(matrix[3], cval_c_t1));
    }
}

__global__ void single_qubit_control_single_qubit_dense_matrix_gate_gpu(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, GTYPE matrix_gpu[4], GTYPE *state_gpu, ITYPE dim) {
	single_qubit_control_single_qubit_dense_matrix_gate_device(control_qubit_index, control_value, target_qubit_index, matrix_gpu, state_gpu, dim);
}

__host__ cudaError single_qubit_control_single_qubit_dense_matrix_gate_host(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, CTYPE matrix[4], GTYPE *state_gpu, ITYPE dim) {
	GTYPE* matrix_gpu;
	cudaError cudaStatus;
	checkCudaErrors(cudaMalloc((void**)&matrix_gpu, 4 * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(matrix_gpu, matrix, 4 * sizeof(CTYPE), cudaMemcpyHostToDevice));

	ITYPE quad_dim = dim>>2;
	unsigned int block = quad_dim <= 1024 ? quad_dim : 1024;
	unsigned int grid = quad_dim / block;
	single_qubit_control_single_qubit_dense_matrix_gate_gpu << <grid, block >> >(control_qubit_index, control_value, target_qubit_index, matrix_gpu, state_gpu, dim);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "single_qubit_control_single_qubit_dense_matrix_gate_host launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	Error:
	cudaFree(matrix_gpu);
	return cudaStatus;
}

void single_qubit_control_single_qubit_dense_matrix_gate(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, CTYPE matrix[4], void *state, ITYPE dim) {
	GTYPE *state_gpu;
	CTYPE *state_cpu = reinterpret_cast<CTYPE*>(state);
	cudaError cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&state_gpu, dim * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(state_gpu, state_cpu, dim * sizeof(CTYPE), cudaMemcpyHostToDevice));
    
	cudaStatus = single_qubit_control_single_qubit_dense_matrix_gate_host(control_qubit_index, control_value, target_qubit_index, matrix, state_gpu, dim);
    
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "single_qubit_control_single_qubit_dense_matrix_gate_host launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
    
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, state_gpu, dim * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	state = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(state_gpu);

}
/*
__device__ void single_qubit_phase_gate_device(unsigned int target_qubit_index, GTYPE phase, GTYPE *state_gpu, ITYPE dim){
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	// target tmask
	const ITYPE mask = 1ULL << target_qubit_index;
	
	// loop varaibles
	const ITYPE loop_dim = dim>>1;
	
	if(state_index<loop_dim){
		// create index
		ITYPE basis_1 = insert_zero_to_basis_index_device(state_index, target_qubit_index) ^ mask;
	
		// set values
		state_gpu[basis_1] = cuCmul(state_gpu[basis_1], phase);
	}
}

__global__ void single_qubit_phase_gate_gpu(unsigned int target_qubit_index, GTYPE phase, GTYPE *state_gpu, ITYPE dim){
	single_qubit_phase_gate_device(target_qubit_index, phase, state_gpu, dim);
}

__host__ cudaError single_qubit_phase_gate_host(unsigned int target_qubit_index, CTYPE phase, GTYPE *state_gpu, ITYPE dim){
	GTYPE phase_gtype;
	cudaError cudaStatus;

	phase_gtype = make_cuDoubleComplex(phase.real(), phase.imag());
	ITYPE half_dim = dim>>1;
	unsigned int block = half_dim <= 1024 ? half_dim : 1024;
	unsigned int grid = half_dim / block;
	single_qubit_phase_gate_gpu << <grid, block >> >(target_qubit_index, phase_gtype, state_gpu, dim);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "single_qubit_phase_gate_host launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	return cudaStatus;
}

void single_qubit_phase_gate(unsigned int target_qubit_index, CTYPE phase, void *state, ITYPE dim){
	GTYPE *state_gpu;
	CTYPE *state_cpu = reinterpret_cast<CTYPE*>(state);
	cudaError cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&state_gpu, dim * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(state_gpu, state_cpu, dim * sizeof(CTYPE), cudaMemcpyHostToDevice));
    
	cudaStatus = single_qubit_phase_gate_host(target_qubit_index, phase, state_gpu, dim);
    
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "single_qubit_phase_gate launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
    
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, state_gpu, dim * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	state = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(state_gpu);	
}
*/
__host__ cudaError RX_gate_host(UINT target_qubit_index, double angle, GTYPE* state_gpu, ITYPE dim){
    return single_qubit_Pauli_rotation_gate_host(target_qubit_index, 1, angle, state_gpu, dim);
}

void RX_gate(UINT target_qubit_index, double angle, void* state, ITYPE dim){
	single_qubit_Pauli_rotation_gate(target_qubit_index, 1, angle, state, dim);
}

__host__ cudaError RY_gate_host(UINT target_qubit_index, double angle, GTYPE* state_gpu, ITYPE dim){
    return single_qubit_Pauli_rotation_gate_host(target_qubit_index, 2, angle, state_gpu, dim);
}

void RY_gate(UINT target_qubit_index, double angle, void* state, ITYPE dim){
    single_qubit_Pauli_rotation_gate(target_qubit_index, 2, angle, state, dim);
}

__host__ cudaError RZ_gate_host(UINT target_qubit_index, double angle, GTYPE* state_gpu, ITYPE dim){
    return single_qubit_Pauli_rotation_gate_host(target_qubit_index, 3, angle, state_gpu, dim);
}

void RZ_gate(UINT target_qubit_index, double angle, void* state, ITYPE dim){
    single_qubit_Pauli_rotation_gate(target_qubit_index, 3, angle, state, dim);
}

// [[1,0],[0,i]]
extern "C"
__host__ cudaError S_gate_host(UINT target_qubit_index, GTYPE* state_gpu, ITYPE dim){
	CTYPE diagonal_matrix[2];
	diagonal_matrix[0]=CTYPE(1.0,0.0);
	diagonal_matrix[1]=CTYPE(0.0,1.0);
	return single_qubit_diagonal_matrix_gate_host(target_qubit_index, diagonal_matrix, state_gpu, dim);
	//CTYPE phase=std::complex<double>(0.0,1.0);
	//return single_qubit_phase_gate_host(target_qubit_index, phase, state_gpu, dim);
}

void S_gate(UINT target_qubit_index, void* state, ITYPE dim){
	CTYPE diagonal_matrix[2];
	diagonal_matrix[0]=CTYPE(1.0,0.0);
	diagonal_matrix[1]=CTYPE(0.0,1.0);
	single_qubit_diagonal_matrix_gate(target_qubit_index, diagonal_matrix, state, dim);
	//CTYPE phase=std::complex<double>(0.0,1.0);
	//single_qubit_phase_gate(target_qubit_index, phase, state, dim);
}

// [[1,0],[0,-i]]
extern "C"
__host__ cudaError Sdag_gate_host(UINT target_qubit_index, GTYPE* state_gpu, ITYPE dim){
	CTYPE diagonal_matrix[2];
	diagonal_matrix[0]=CTYPE(1.0,0.0);
	diagonal_matrix[1]=CTYPE(0.0,-1.0);
	return single_qubit_diagonal_matrix_gate_host(target_qubit_index, diagonal_matrix, state_gpu, dim);
	//CTYPE phase=std::complex<double>(0.0, -1.0);
	//return single_qubit_phase_gate_host(target_qubit_index, phase, state_gpu, dim);
}

void Sdag_gate(UINT target_qubit_index, void* state, ITYPE dim){
	CTYPE diagonal_matrix[2];
	diagonal_matrix[0]=CTYPE(1.0,0.0);
	diagonal_matrix[1]=CTYPE(0.0,-1.0);
	single_qubit_diagonal_matrix_gate(target_qubit_index, diagonal_matrix, state, dim);
	//CTYPE phase=std::complex<double>(0.0, -1.0);
	//single_qubit_phase_gate(target_qubit_index, phase, state, dim);
}

// [[1,0],[0,exp(i*pi/4)]] , (1+i)/sprt(2)
extern "C"
__host__ cudaError T_gate_host(UINT target_qubit_index, GTYPE* state_gpu, ITYPE dim){
	CTYPE diagonal_matrix[2];
	diagonal_matrix[0]=CTYPE(1.0,0.0);
	diagonal_matrix[1]=CTYPE(1.0/sqrt(2),1.0/sqrt(2));
	return single_qubit_diagonal_matrix_gate_host(target_qubit_index, diagonal_matrix, state_gpu, dim);
	//CTYPE phase=std::complex<double>(1.0/sqrt(2), 1.0/sqrt(2));
	//return single_qubit_phase_gate_host(target_qubit_index, phase, state_gpu, dim);
}

void T_gate(UINT target_qubit_index, void* state, ITYPE dim){
	CTYPE diagonal_matrix[2];
	diagonal_matrix[0]=CTYPE(1.0,0.0);
	diagonal_matrix[1]=CTYPE(1.0/sqrt(2),1.0/sqrt(2));
	single_qubit_diagonal_matrix_gate(target_qubit_index, diagonal_matrix, state, dim);
	//CTYPE phase=std::complex<double>(1.0/sqrt(2), 1.0/sqrt(2));
	//single_qubit_phase_gate(target_qubit_index, phase, state, dim);
}
// [[1,0],[0,-exp(i*pi/4)]], (1-i)/sqrt(2)
extern "C"
__host__ cudaError Tdag_gate_host(UINT target_qubit_index, GTYPE* state_gpu, ITYPE dim){
	CTYPE diagonal_matrix[2];
	diagonal_matrix[0]=CTYPE(1.0, 0.0);
	diagonal_matrix[1]=CTYPE(1.0/sqrt(2), -1.0/sqrt(2));
	return single_qubit_diagonal_matrix_gate_host(target_qubit_index, diagonal_matrix, state_gpu, dim);
	//CTYPE phase=std::complex<double>(1.0/sqrt(2), -1.0/sqrt(2));
    //return single_qubit_phase_gate_host(target_qubit_index, phase, state_gpu, dim);
}

void Tdag_gate(UINT target_qubit_index, void* state, ITYPE dim){
	CTYPE diagonal_matrix[2];
	diagonal_matrix[0]=CTYPE(1.0, 0.0);
	diagonal_matrix[1]=CTYPE(1.0/sqrt(2), -1.0/sqrt(2));
	single_qubit_diagonal_matrix_gate(target_qubit_index, diagonal_matrix, state, dim);
	//CTYPE phase=std::complex<double>(1.0/sqrt(2), -1.0/sqrt(2));
    //single_qubit_phase_gate(target_qubit_index, phase, state, dim);
}

__host__ cudaError sqrtX_gate_host(UINT target_qubit_index, GTYPE* state_gpu, ITYPE dim){
	CTYPE SQRT_X_GATE_MATRIX[4] = {
		std::complex<double>(0.5, 0.5), std::complex<double>(0.5, -0.5),
		std::complex<double>(0.5, -0.5), std::complex<double>(0.5, 0.5)
	};
	return single_qubit_dense_matrix_gate_host(target_qubit_index, SQRT_X_GATE_MATRIX, state_gpu, dim);
}

void sqrtX_gate(UINT target_qubit_index, void* state, ITYPE dim){
	CTYPE SQRT_X_GATE_MATRIX[4] = {
		std::complex<double>(0.5, 0.5), std::complex<double>(0.5, -0.5),
		std::complex<double>(0.5, -0.5), std::complex<double>(0.5, 0.5)
	};
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_X_GATE_MATRIX, state, dim);
}

__host__ cudaError sqrtXdag_gate_host(UINT target_qubit_index, GTYPE* state_gpu, ITYPE dim){
	CTYPE SQRT_X_DAG_GATE_MATRIX[4] = 
	{
		std::complex<double>(0.5, -0.5), std::complex<double>(0.5, 0.5),
		std::complex<double>(0.5, 0.5), std::complex<double>(0.5, -0.5)
	};
    return single_qubit_dense_matrix_gate_host(target_qubit_index, SQRT_X_DAG_GATE_MATRIX, state_gpu, dim);
}

void sqrtXdag_gate(UINT target_qubit_index, void* state, ITYPE dim){
	CTYPE SQRT_X_DAG_GATE_MATRIX[4] = 
	{
		std::complex<double>(0.5, -0.5), std::complex<double>(0.5, 0.5),
		std::complex<double>(0.5, 0.5), std::complex<double>(0.5, -0.5)
	};
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_X_DAG_GATE_MATRIX, state, dim);
}

__host__ cudaError sqrtY_gate_host(UINT target_qubit_index, GTYPE* state_gpu, ITYPE dim){
	CTYPE SQRT_Y_GATE_MATRIX[4] =
	{
		std::complex<double>(0.5, 0.5), std::complex<double>(-0.5, -0.5),
		std::complex<double>(0.5, 0.5), std::complex<double>(0.5, 0.5)
	};
    return single_qubit_dense_matrix_gate_host(target_qubit_index, SQRT_Y_GATE_MATRIX, state_gpu, dim);
}

void sqrtY_gate(UINT target_qubit_index, void* state, ITYPE dim){
	CTYPE SQRT_Y_GATE_MATRIX[4] =
	{
		std::complex<double>(0.5, 0.5), std::complex<double>(-0.5, -0.5),
		std::complex<double>(0.5, 0.5), std::complex<double>(0.5, 0.5)
	};
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_Y_GATE_MATRIX, state, dim);
}

__host__ cudaError sqrtYdag_gate_host(UINT target_qubit_index, GTYPE* state_gpu, ITYPE dim){
	CTYPE SQRT_Y_DAG_GATE_MATRIX[4] =
	{
		std::complex<double>(0.5, -0.5), std::complex<double>(0.5, -0.5),
		std::complex<double>(-0.5, 0.5), std::complex<double>(0.5, -0.5)
	};
    return single_qubit_dense_matrix_gate_host(target_qubit_index, SQRT_Y_DAG_GATE_MATRIX, state_gpu, dim);
}

void sqrtYdag_gate(UINT target_qubit_index, void* state, ITYPE dim){
	CTYPE SQRT_Y_DAG_GATE_MATRIX[4] =
	{
		std::complex<double>(0.5, -0.5), std::complex<double>(0.5, -0.5),
		std::complex<double>(-0.5, 0.5), std::complex<double>(0.5, -0.5)
	};
    single_qubit_dense_matrix_gate(target_qubit_index, SQRT_Y_DAG_GATE_MATRIX, state, dim);
}

 
/*
void multi_qubit_control_single_qubit_dense_matrix_gate(const UINT* control_qubit_index_list, const UINT* control_value_list, UINT control_qubit_index_count, 
    UINT target_qubit_index, const CTYPE matrix[4], void *state, ITYPE dim)
*/