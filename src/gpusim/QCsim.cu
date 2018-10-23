#include "device_launch_parameters.h"
#include <cuda_runtime.h>
//#include <cuda.h>

#ifdef __cplusplus
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
//#include "util.h"
#include "util.cuh"
#include "util_common.h"
#include "update_ops_cuda.h"

extern "C" int sum_i(int *a, ITYPE N);
extern "C" double sum_d(double *a, ITYPE N);
extern "C" CTYPE sum_c(CTYPE *a, ITYPE N);

extern "C" DllExport void P0_gate(unsigned int target_qubit_index, CTYPE *state, ITYPE dim);
extern "C" DllExport void P1_gate(unsigned int target_qubit_index, CTYPE *state, ITYPE dim);
extern "C" DllExport void normalize(double norm, CTYPE* state, ITYPE dim);

extern "C" DllExport void multi_qubit_Pauli_gate_whole_list(const unsigned int* Pauli_operator_type_list, unsigned int qubit_count, CTYPE* state, ITYPE dim_);
extern "C" DllExport void multi_qubit_Pauli_gate_partial_list(const unsigned int* target_qubit_index_list, const unsigned int* Pauli_operator_type_list, unsigned int target_qubit_index_count, CTYPE* state, ITYPE dim);
extern "C" DllExport void multi_qubit_Pauli_rotation_gate_whole_list(const unsigned int* Pauli_operator_type_list, unsigned int qubit_count, double angle, CTYPE* state, ITYPE dim_);
extern "C" DllExport void multi_qubit_Pauli_rotation_gate_partial_list(const unsigned int* target_qubit_index_list, const unsigned int* Pauli_operator_type_list, unsigned int target_qubit_index_count, double angle, CTYPE* state, ITYPE dim);
extern "C" DllExport void multi_qubit_dense_matrix_gate(const unsigned int* target_qubit_index_list, unsigned int target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim);
extern "C" DllExport void single_qubit_control_multi_qubit_dense_matrix_gate(unsigned int control_qubit_index, unsigned int control_value, const unsigned int* target_qubit_index_list, unsigned int target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim);
extern "C" DllExport void multi_qubit_control_multi_qubit_dense_matrix_gate(const unsigned int* control_qubit_index_list, const unsigned int* control_value_list, unsigned int control_qubit_index_count, const unsigned int* target_qubit_index_list, unsigned int target_qubit_index_count, const CTYPE* matrix, CTYPE* state, ITYPE dim);


__global__ void inner_product_gpu(GTYPE *ret, GTYPE *psi, GTYPE *phi, ITYPE DIM);
extern "C" DllExport CTYPE inner_product(void *psi, void *phi, ITYPE DIM);
extern "C" DllExport CTYPE state_inner_product(void *state_bra, void *state_ket, ITYPE dim);

__global__ void expectation_value_single_qubit_Pauli_operator_gpu(GTYPE *ret, GTYPE U[4], GTYPE *psi, unsigned int target_qubit_index, ITYPE DIM);
__host__ double expectation_value_single_qubit_Pauli_operator_host(unsigned int operator_index, unsigned int targetQubitIndex, GTYPE *psi_gpu, ITYPE DIM);
extern "C" DllExport double expectation_value_single_qubit_Pauli_operator(unsigned int operator_index, unsigned int targetQubitIndex, void *psi, ITYPE DIM);
__device__ void multi_Z_gate_device(ITYPE bit_mask, ITYPE DIM, GTYPE *psi_gpu);
__global__ void multi_Z_gate_gpu(ITYPE bit_mask, ITYPE DIM, GTYPE *psi_gpu);
__global__ void multi_Pauli_gate_gpu(int* gates, ITYPE bit_mask_XY, int* num_pauli_op, ITYPE DIM, GTYPE *psi_gpu, int n_qubits);
__host__ cudaError multi_Pauli_gate_host(int* gates, GTYPE *psi_gpu, ITYPE DIM, int n_qubits);
extern "C" DllExport void multi_Pauli_gate(int* gates, void *psi, ITYPE DIM, int n_qubits);
__device__ GTYPE multi_Z_get_expectation_value_device(ITYPE idx, ITYPE bit_mask, ITYPE DIM, GTYPE *psi_gpu);
__global__ void multi_Z_gate_gpu(ITYPE bit_mask, ITYPE DIM, GTYPE *psi_gpu);
__host__ cudaError multi_Z_gate_host(int* gates, GTYPE *psi_gpu, ITYPE DIM, int n_qubits);
extern "C" DllExport void multi_Z_gate(int* gates, void *psi, ITYPE DIM, int n_qubits);
__device__ GTYPE multipauli_get_expectation_value_device(ITYPE idx, ITYPE* bit_mask_gpu, int* num_pauli_op, ITYPE DIM, GTYPE *psi_gpu, int n_qubits);
extern "C" DllExport __host__ double multipauli_get_expectation_value_host(unsigned int* gates, GTYPE *psi_gpu, ITYPE DIM, int n_qubits);
//__host__ cudaError multipauli_get_expectation_value_host(double ret, int* gates, GTYPE *psi_gpu, ITYPE DIM, int n_qubits);
extern "C" DllExport double multipauli_get_expectation_value(unsigned int* gates, void *psi, ITYPE DIM, int n_qubits);


// extern "C" DllExport double state_norm(const CTYPE *state, ITYPE dim) ;
// extern "C" DllExport double measurement_distribution_entropy(const CTYPE *state, ITYPE dim);
// extern "C" DllExport double M0_prob(unsigned int target_qubit_index, const CTYPE* state, ITYPE dim);
// extern "C" DllExport double M1_prob(unsigned int target_qubit_index, const CTYPE* state, ITYPE dim);
// extern "C" DllExport double marginal_prob(const unsigned int* sorted_target_qubit_index_list, const unsigned int* measured_value_list, unsigned int target_qubit_index_count, const CTYPE* state, ITYPE dim);
// extern "C" DllExport double expectation_value_single_qubit_Pauli_operator(unsigned int target_qubit_index, unsigned int Pauli_operator_type, const CTYPE *state, ITYPE dim);
// extern "C" DllExport double expectation_value_multi_qubit_Pauli_operator_whole_list(const unsigned int* Pauli_operator_type_list, unsigned int qubit_count, const CTYPE* state, ITYPE dim);
// extern "C" DllExport double expectation_value_multi_qubit_Pauli_operator_partial_list(const unsigned int* target_qubit_index_list, const unsigned int* Pauli_operator_type_list, unsigned int target_qubit_index_count, const CTYPE* state, ITYPE dim);


extern "C"
int sum_i(int *a, ITYPE N)
{
	int *a_gpu, *out_gpu;
	int out=0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&a_gpu, N * sizeof(int)));
	checkCudaErrors(cudaMemcpy(a_gpu, a, N * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&out_gpu, sizeof(int)));
	checkCudaErrors(cudaMemcpy(out_gpu, &out, sizeof(int), cudaMemcpyHostToDevice));
	
	// Launch a kernel on the GPU with one thread for each element.
	unsigned int block = N <= 1024 ? N : 1024;
	unsigned int grid = N / block;
	deviceReduceWarpAtomicKernel << <grid, block >> >(a_gpu, out_gpu, N);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "deviceReduceWarpAtomicKernel(sum_i) launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(&out, out_gpu, sizeof(int), cudaMemcpyDeviceToHost));
	
Error:
	cudaFree(out_gpu);
	cudaFree(a_gpu);
	return out;
}

extern "C"
double sum_d(double *a, ITYPE N)
{
	double *a_gpu, *out_gpu;
	double out=0.0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&a_gpu, N * sizeof(double)));
	checkCudaErrors(cudaMemcpy(a_gpu, a, N * sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&out_gpu, sizeof(double)));
	checkCudaErrors(cudaMemcpy(out_gpu, &out, sizeof(double), cudaMemcpyHostToDevice));

	unsigned int block = N <= 1024 ? N : 1024;
	unsigned int grid = N / block;
	deviceReduceWarpAtomicKernel << <block, grid>> >(a_gpu, out_gpu, N);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "deviceReduceWarpAtomicKernel(sum_d) launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(&out, out_gpu, sizeof(double), cudaMemcpyDeviceToHost));

Error:
	cudaFree(out_gpu);
	cudaFree(a_gpu);
	return out;
}

extern "C"
CTYPE sum_c(CTYPE *a, ITYPE N)
{
	GTYPE *a_gpu, *out_gpu;
	CTYPE out = CTYPE(0, 0);
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&a_gpu, N * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(a_gpu, a, N * sizeof(CTYPE), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&out_gpu, sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(out_gpu, &out, sizeof(CTYPE), cudaMemcpyHostToDevice));
	
	// Launch a kernel on the GPU with one thread for each element.
	unsigned int block = N <= 1024 ? N : 1024;
	unsigned int grid = N / block;
	deviceReduceWarpAtomicKernel << <grid, block >> >(a_gpu, out_gpu, N);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "deviceReduceWarpAtomicKernel(sum_c) launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(&out, out_gpu, sizeof(CTYPE), cudaMemcpyDeviceToHost));
	
Error:
	cudaFree(out_gpu);
	cudaFree(a_gpu);
	return out;
}

__global__ void CZ_gate_gpu(unsigned int control_qubit_indexr, unsigned int target_qubit_index, GTYPE *psi, ITYPE DIM) {
	ITYPE bigger, smaller;
	ITYPE tmp, tmp1, tmp2;
	ITYPE quarter_DIM = DIM / 4;
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;
	if (control_qubit_indexr > target_qubit_index) {
		bigger = control_qubit_indexr;
		smaller = target_qubit_index;
	}
	else {
		bigger = target_qubit_index;
		smaller = control_qubit_indexr;
	}
	if (j < quarter_DIM){
		tmp = j & ((1LL << smaller) - 1);
		tmp1 = j & ((1LL << (bigger - 1)) - 1);
		tmp2 = j - tmp1;

		tmp1 = tmp1 - tmp;
		tmp1 = (tmp1 << 1);

		tmp2 = (tmp2 << 2);

		tmp = tmp + (1LL << smaller) + tmp1 + (1LL << bigger) + tmp2;

		psi[tmp] = make_cuDoubleComplex(-cuCreal(psi[tmp]), -cuCimag(psi[tmp]));
	}
}

extern "C"
__host__ cudaError CZ_gate_host(unsigned int control_qubit_indexr, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE dim){
	cudaError cudaStatus;
	ITYPE quad_dim = dim >> 2;
	int block = quad_dim <= 1024 ? quad_dim : 1024;
	int grid = dim / block;

	CZ_gate_gpu << <grid, block >> >(control_qubit_indexr, target_qubit_index, psi_gpu, dim);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CZ_gate_host launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

__global__ void CNOT_gate_gpu(unsigned int control_qubit_indexr, unsigned int target_qubit_index, GTYPE *psi, ITYPE dim)
{
	unsigned int left, right;
	ITYPE head, body, tail;
	ITYPE basis10, basis11;
	GTYPE tmp_psi;
	ITYPE quarter_dim = dim / 4;
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;

	if (target_qubit_index > control_qubit_indexr){
		left = target_qubit_index;
		right = control_qubit_indexr;
	}
	else {
		left = control_qubit_indexr;
		right = target_qubit_index;
	}

	if (j < quarter_dim){
		head = j >> (left - 1);
		body = (j & ((1LL << (left - 1)) - 1)) >> right; // (j % 2^(k-1)) >> i
		tail = j & ((1LL << right) - 1); // j%(2^i)

		// ONE<<control
		basis10 = (head << (left + 1)) + (body << (right + 1)) + (1LL << control_qubit_indexr) + tail;
		// ONE<<target, ONE<<control
		basis11 = (head << (left + 1)) + (body << (right + 1)) + (1LL << target_qubit_index) + (1LL << control_qubit_indexr) + tail;

		tmp_psi = psi[basis11];
		psi[basis11] = psi[basis10];
		psi[basis10] = tmp_psi;
	}
}

extern "C"
__host__ cudaError CNOT_gate_host(unsigned int control_qubit_indexr, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE dim){
	cudaError cudaStatus;
	ITYPE quad_dim = dim >> 2;
	int block = quad_dim <= 1024 ? quad_dim : 1024;
	int grid = dim / block;

	CNOT_gate_gpu << <grid, block >> >(control_qubit_indexr, target_qubit_index, psi_gpu, dim);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CNOT_gate_host launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

__global__ void SWAP_gate_gpu(unsigned int i, unsigned int k, GTYPE *psi_gpu, ITYPE dim) {
	unsigned int left, right;
	ITYPE head, body, tail;
	ITYPE basis01, basis10;
	GTYPE tmp;
	ITYPE quarter_dim = dim / 4;
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;

	if (k > i) {
		left = k;
		right = i;
	}
	else {
		left = i;
		right = k;
	}

	if (j < quarter_dim){
		head = j >> (left - 1);
		body = (j & ((1LL << (left - 1)) - 1)) >> right; // (j % 2^(k-1)) >> i
		tail = j & ((1LL << right) - 1); // j%(2^i)

		basis01 = (head << (left + 1)) + (body << (right + 1)) + (1LL << k) + tail;
		basis10 = (head << (left + 1)) + (body << (right + 1)) + (1LL << i) + tail;

		tmp = psi_gpu[basis01];
		psi_gpu[basis01] = psi_gpu[basis10];
		psi_gpu[basis10] = tmp;
	}
}

extern "C"
__host__ cudaError SWAP_gate_host(unsigned int control_qubit_indexr, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE dim){
	cudaError cudaStatus;
	ITYPE quad_dim = dim >> 2;
	int block = quad_dim <= 1024 ? quad_dim : 1024;
	int grid = dim / block;

	SWAP_gate_gpu << <grid, block >> >(control_qubit_indexr, target_qubit_index, psi_gpu, dim);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "SWAP_gate_host launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

__host__ cudaError single_qubit_Pauli_rotation_gate_host(unsigned int target_qubit_index, unsigned int op_idx, double angle, void *state_gpu, ITYPE dim) {
    cudaError cudaStatus;
	GTYPE* psi_gpu = reinterpret_cast<GTYPE*>(state_gpu);
	cuDoubleComplex PAULI_MATRIX_gpu[4][4] = {
		{ make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0) },
		{ make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0) },
		{ make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, -1), make_cuDoubleComplex(0, 1), make_cuDoubleComplex(0, 0) },
		{ make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(-1, 0) }
	};

	GTYPE rotation_gate[4];
	GTYPE *rotation_gate_gpu;

	rotation_gate[0] = make_cuDoubleComplex(
		cos(angle) - sin(angle)* cuCimag(PAULI_MATRIX_gpu[op_idx][0]),
		sin(angle) * cuCreal(PAULI_MATRIX_gpu[op_idx][0])
		);
	rotation_gate[1] = make_cuDoubleComplex(
		-sin(angle)* cuCimag(PAULI_MATRIX_gpu[op_idx][1]),
		sin(angle) * cuCreal(PAULI_MATRIX_gpu[op_idx][1])
		);
	rotation_gate[2] = make_cuDoubleComplex(
		-sin(angle)* cuCimag(PAULI_MATRIX_gpu[op_idx][2]),
		sin(angle) * cuCreal(PAULI_MATRIX_gpu[op_idx][2])
		);
	rotation_gate[3] = make_cuDoubleComplex(
		cos(angle) - sin(angle)* cuCimag(PAULI_MATRIX_gpu[op_idx][3]),
		sin(angle) * cuCreal(PAULI_MATRIX_gpu[op_idx][3])
		);

	// cudaStatus = cudaMalloc((void**)&rotation_gate_gpu, 4 * sizeof(GTYPE));
	checkCudaErrors(cudaMalloc((void**)&rotation_gate_gpu, 4 * sizeof(GTYPE)));
	checkCudaErrors(cudaMemcpy(rotation_gate_gpu, rotation_gate, 4 * sizeof(GTYPE), cudaMemcpyHostToDevice));

	ITYPE half_dim = dim >> 1;
	unsigned int block = half_dim <= 1024 ? half_dim : 1024;
	unsigned int grid = dim / block;
	single_qubit_dense_matrix_gate_gpu << <grid, block >> >(target_qubit_index, rotation_gate_gpu, psi_gpu, dim);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "single_qubit_Pauli_rotation_gate_host launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	state_gpu = reinterpret_cast<void*>(psi_gpu);
Error:
	cudaFree(rotation_gate_gpu);
	return cudaStatus;
}

__host__ cudaError U1_gate_host(double lambda, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE DIM){
	CTYPE U_gate[4];
	cudaError cudaStatus;

	U_gate[0] = std::complex<double>(1.0, 0.0); // make_cuDoubleComplex(1.0, 0.0);
	U_gate[1] = std::complex<double>(0.0, 0.0); //make_cuDoubleComplex(0.0, 0.0);
	U_gate[2] = std::complex<double>(0.0, 0.0); // make_cuDoubleComplex(0.0, 0.0);
	U_gate[3] = std::complex<double>(cos(lambda), sin(lambda)); // make_cuDoubleComplex(cos(lambda), sin(lambda));

	cudaStatus=single_qubit_dense_matrix_gate_host(target_qubit_index, U_gate, psi_gpu, DIM);
	// Check for any errors launching the kernel
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "U1_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

extern "C"
void U1_gate(double lambda, unsigned int target_qubit_index, void *psi, ITYPE DIM) {
	GTYPE *psi_gpu;
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(psi);

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, DIM * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, state_cpu, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));

	U1_gate_host(lambda, target_qubit_index, psi_gpu, DIM);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, psi_gpu, DIM * sizeof(CTYPE), cudaMemcpyDeviceToHost));

	psi = reinterpret_cast<void*>(state_cpu);
	cudaFree(psi_gpu);
	//free(state_cpu);
}

__host__ cudaError U2_gate_host(double lambda, double phi, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE DIM){
	GTYPE *U_gpu;
	GTYPE U_gate[4];
	cudaError cudaStatus;
	double sqrt2_inv = 1.0 / sqrt(2.0);
	GTYPE exp_val1 = make_cuDoubleComplex(cos(phi), sin(phi));
	GTYPE exp_val2 = make_cuDoubleComplex(cos(lambda), sin(lambda));

	U_gate[0] = make_cuDoubleComplex(sqrt2_inv, 0.0);
	U_gate[1] = make_cuDoubleComplex(-cos(lambda) / sqrt(2.0), -sin(lambda) / sqrt(2.0));
	U_gate[2] = make_cuDoubleComplex(cos(phi) / sqrt(2.0), sin(phi) / sqrt(2.0));
	U_gate[3] = cuCmul(exp_val1, exp_val2);
	U_gate[3] = make_cuDoubleComplex(U_gate[3].x / sqrt(2.0), U_gate[3].y / sqrt(2.0));

	checkCudaErrors(cudaMalloc((void**)&U_gpu, 4 * sizeof(GTYPE)));
	checkCudaErrors(cudaMemcpy(U_gpu, U_gate, 4 * sizeof(GTYPE), cudaMemcpyHostToDevice));

	ITYPE half_dim = DIM >> 1;
	int block = half_dim <= 1024 ? half_dim : 1024;
	int grid = DIM / block;
	single_qubit_dense_matrix_gate_gpu << <grid, block >> >(target_qubit_index, U_gpu, psi_gpu, DIM);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "U2_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	cudaFree(U_gpu);
	return cudaStatus;
}

extern "C"
void U2_gate(double lambda, double phi, unsigned int target_qubit_index, void *psi, ITYPE DIM) {
	GTYPE *psi_gpu;
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(psi);
	cudaError cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, DIM * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, state_cpu, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));

	U2_gate_host(lambda, phi, target_qubit_index, psi_gpu, DIM);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, psi_gpu, DIM * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	psi = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(psi_gpu);
	//free(state_cpu);
}

__host__ cudaError U3_gate_host(double lambda, double phi, double theta, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE DIM){
	GTYPE *U_gpu;
	GTYPE U_gate[4];
	cudaError cudaStatus;
	double sqrt2_inv = 1.0 / sqrt(2.0);
	GTYPE exp_val1 = make_cuDoubleComplex(cos(phi), sin(phi));
	GTYPE exp_val2 = make_cuDoubleComplex(cos(lambda), sin(lambda));
	double cos_val = cos(theta / 2);
	double sin_val = sin(theta / 2);

	U_gate[0] = make_cuDoubleComplex(cos_val, 0.0);
	U_gate[1] = make_cuDoubleComplex(-cos(lambda)*sin_val, -sin(lambda)*sin_val);
	U_gate[2] = make_cuDoubleComplex(cos(phi)*sin_val, sin(phi)*sin_val);
	U_gate[3] = cuCmul(exp_val1, exp_val2);
	U_gate[3] = make_cuDoubleComplex(U_gate[3].x*cos_val, U_gate[3].y*cos_val);

	checkCudaErrors( cudaMalloc((void**)&U_gpu, 4 * sizeof(GTYPE)));
	checkCudaErrors(cudaMemcpy(U_gpu, U_gate, 4 * sizeof(GTYPE), cudaMemcpyHostToDevice));

	ITYPE half_dim = DIM >> 1;
	int block = half_dim <= 1024 ? half_dim : 1024;
	int grid = DIM / block;
	single_qubit_dense_matrix_gate_gpu << <grid, block >> >(target_qubit_index, U_gpu, psi_gpu, DIM);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "U3_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	cudaFree(U_gpu);
	return cudaStatus;
}

extern "C"
void U3_gate(double lambda, double phi, double theta, unsigned int target_qubit_index, void *psi, ITYPE DIM) {
	GTYPE *psi_gpu;
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(psi);
	cudaError cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, DIM * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, state_cpu, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));

	U3_gate_host(lambda, phi, theta, target_qubit_index, psi_gpu, DIM);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, psi_gpu, DIM * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	psi = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(psi_gpu);
	//free(state_cpu);
}

extern "C"
void CNOT_gate(unsigned int control_qubit_indexr, unsigned int target_qubit_index, void *psi, ITYPE DIM)
{
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(psi);
	GTYPE *psi_gpu;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, DIM * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, state_cpu, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));
	
	ITYPE quad_dim = DIM >> 2;
	int block= quad_dim<=1024 ? quad_dim : 1024;
	int grid = DIM / block;
	CNOT_gate_gpu << <grid, block >> >(control_qubit_indexr, target_qubit_index, psi_gpu, DIM);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CNOT_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, psi_gpu, DIM * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	psi = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(psi_gpu);
	//free(state_cpu);
}

extern "C"
void CZ_gate(unsigned int control_qubit_indexr, unsigned int target_qubit_index, void *psi, ITYPE DIM)
{
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(psi);
	GTYPE *psi_gpu;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, DIM * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, state_cpu, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));
	
	CZ_gate_host(control_qubit_indexr, target_qubit_index, psi_gpu, DIM);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CZ_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, psi_gpu, DIM * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	psi = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(psi_gpu);
	//free(state_cpu);
}

extern "C"
void SWAP_gate(unsigned int target_qubit_index_0, unsigned int target_qubit_index_1, void *psi, ITYPE DIM)
{
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(psi);
	GTYPE *psi_gpu;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, DIM * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, state_cpu, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));
	
	ITYPE quad_dim = DIM >> 2;
	int block = quad_dim <= 1024 ? quad_dim : 1024;
	int grid = DIM / block;
	SWAP_gate_gpu << <grid, block >> >(target_qubit_index_0, target_qubit_index_1, psi_gpu, DIM);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "SWAP_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, psi_gpu, DIM * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	psi = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(psi_gpu);
	//free(state_cpu);
}


extern "C"
void single_qubit_Pauli_rotation_gate(unsigned int target_qubit_index, unsigned int op_idx, double angle, void *psi, ITYPE DIM)
{
	GTYPE *psi_gpu;
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(psi);
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, DIM * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, psi, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));
	
	single_qubit_Pauli_rotation_gate_host(target_qubit_index, op_idx, angle, psi_gpu, DIM);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "single_qubit_Pauli_rotation_gate launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(psi, psi_gpu, DIM * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	psi = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(psi_gpu);
	//free(state_cpu);
}

__global__ void inner_product_gpu(GTYPE *ret, GTYPE *psi, GTYPE *phi, ITYPE DIM){
	GTYPE sum = make_cuDoubleComplex(0.0, 0.0);
	for (ITYPE i = blockIdx.x * blockDim.x + threadIdx.x; i < DIM; i += blockDim.x * gridDim.x) {
		sum = cuCadd(sum, cuCmul(cuConj(psi[i]), phi[i]));
	}
	sum.x = warpReduceSum_double(sum.x);
	sum.y = warpReduceSum_double(sum.y);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret[0].x), sum.x);
		atomicAdd_double(&(ret[0].y), sum.y);
	}
}

extern "C"
CTYPE inner_product(void *psi, void *phi, ITYPE DIM) {
	GTYPE *psi_gpu, *phi_gpu;
	CTYPE* state_psi_cpu = reinterpret_cast<CTYPE*>(psi);
	CTYPE* state_phi_cpu = reinterpret_cast<CTYPE*>(phi);
	cudaError_t cudaStatus;
	CTYPE ret=CTYPE(0.0,0.0);
	GTYPE *ret_gpu;
	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, DIM * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, state_psi_cpu, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&phi_gpu, DIM * sizeof(CTYPE)));

	// Copy input vectors from host memory to GPU buffers.
	checkCudaErrors(cudaMemcpy(phi_gpu, state_phi_cpu, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&ret_gpu, sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(ret_gpu, &ret, sizeof(CTYPE), cudaMemcpyHostToDevice));

	int block = DIM <= 1024 ? DIM : 1024;
	int grid = DIM / block;
	inner_product_gpu << <grid, block >> >(ret_gpu, psi_gpu, phi_gpu, DIM);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "inner_product launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(&ret, ret_gpu, sizeof(CTYPE), cudaMemcpyDeviceToHost));

Error:
	cudaFree(ret_gpu);
	cudaFree(psi_gpu);
	cudaFree(phi_gpu);
	//free(state_cpu);
	return ret;
}

__global__ void expectation_value_single_qubit_Pauli_operator_gpu(
	GTYPE *ret, GTYPE U[4], GTYPE *psi, unsigned int target_qubit_index, ITYPE DIM
	){
	GTYPE sum = make_cuDoubleComplex(0.0, 0.0);
	GTYPE tmp;
	int j=0;
	for (ITYPE state = blockIdx.x * blockDim.x + threadIdx.x; state < DIM; state += blockDim.x * gridDim.x) {
		tmp = psi[state];
		j = (state >> target_qubit_index) & 1;
		if (j){
			tmp = cuCadd(cuCmul(U[2], psi[state^(1<<target_qubit_index)]), cuCmul(U[3], psi[state]));
		}
		else{
			tmp = cuCadd(cuCmul(U[0], psi[state]), cuCmul(U[1], psi[state^(1<<target_qubit_index)]));
		}
		sum = cuCadd(sum, cuCmul(cuConj(psi[state]), tmp));
	}
	sum.x = warpReduceSum_double(sum.x);
	sum.y = warpReduceSum_double(sum.y);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret[0].x), sum.x);
		atomicAdd_double(&(ret[0].y), sum.y);
	}
}

// calculate expectation value for single-qubit pauli operator
__host__ double expectation_value_single_qubit_Pauli_operator_host(unsigned int operator_index, unsigned int targetQubitIndex,
	GTYPE *psi_gpu, ITYPE DIM) {
	CTYPE ret = CTYPE(0.0, 0.0);
	GTYPE *ret_gpu;
	cuDoubleComplex PAULI_MATRIX[4][4] = {
		{ make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0) },
		{ make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0) },
		{ make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, -1), make_cuDoubleComplex(0, 1), make_cuDoubleComplex(0, 0) },
		{ make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(-1, 0) }
	};
	cudaError cudaStatus;
	cuDoubleComplex *PAULI_MATRIX_gpu;
	// Allocate GPU buffers for three vectors (two input, one output)    .
	checkCudaErrors(cudaMalloc((void**)&PAULI_MATRIX_gpu, 4 * sizeof(GTYPE)));
	checkCudaErrors(cudaMemcpy(PAULI_MATRIX_gpu, PAULI_MATRIX[operator_index], 4 * sizeof(GTYPE), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&ret_gpu, sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(ret_gpu, &ret, sizeof(CTYPE), cudaMemcpyHostToDevice));

	int block = DIM <= 1024 ? DIM : 1024;
	int grid = DIM / block;

	expectation_value_single_qubit_Pauli_operator_gpu << <grid, block >> >(ret_gpu, PAULI_MATRIX_gpu, psi_gpu, targetQubitIndex, DIM);

	// Copy output vector from GPU buffer to host memory.
	checkCudaErrors(cudaMemcpy(&ret, ret_gpu, sizeof(CTYPE), cudaMemcpyDeviceToHost));
	
Error:
	cudaFree(PAULI_MATRIX_gpu);
	cudaFree(ret_gpu);
	return ret.real();
}

extern "C"
double expectation_value_single_qubit_Pauli_operator(unsigned int operator_index, unsigned int targetQubitIndex,
	void *psi, ITYPE DIM){
	GTYPE *psi_gpu;
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(psi);
	double ret;
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, DIM * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, psi, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));

	ret = expectation_value_single_qubit_Pauli_operator_host(operator_index, targetQubitIndex, psi_gpu, DIM);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "expectation_value_single_qubit_Pauli_operator launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	checkCudaErrors(cudaDeviceSynchronize());
	psi = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(psi_gpu);
	//free(state_cpu);
	return ret;
}

__device__ void multi_Z_gate_device(ITYPE bit_mask, ITYPE DIM, GTYPE *psi_gpu)
{
	ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int minus_cnt = 0;
	if (idx < DIM){
		minus_cnt = popcount64(idx&bit_mask);
		if (minus_cnt & 1) psi_gpu[idx] = make_cuDoubleComplex(-psi_gpu[idx].x, -psi_gpu[idx].y);
	}
}

__global__ void multi_Z_gate_gpu(ITYPE bit_mask, ITYPE DIM, GTYPE *psi_gpu)
{
	multi_Z_gate_device(bit_mask, DIM, psi_gpu);
}

__host__ cudaError multi_Z_gate_host(int* gates, GTYPE *psi_gpu, ITYPE DIM, int n_qubits){
	ITYPE bit_mask=0;
	for (int i = 0; i < n_qubits; ++i){
		if (gates[i]==3) bit_mask ^= (1 << i);
	}
	cudaError cudaStatus;
	unsigned int block = DIM <= 1024 ? DIM : 1024;
	unsigned int grid = DIM / block;
	multi_Z_gate_gpu << <grid, block >> >(bit_mask, DIM, psi_gpu);
	cudaStatus = cudaGetLastError();
	return cudaStatus;
}

extern "C"
void multi_Z_gate(int* gates, void *psi, ITYPE DIM, int n_qubits){
	GTYPE *psi_gpu;
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(psi);
	cudaError_t cudaStatus;
	ITYPE bit_mask = 0;
	for (int i = 0; i < n_qubits; ++i){
		if (gates[i] == 3) bit_mask ^= (1 << i);
	}
	// printf("bit_mask: %d\n", bit_mask);
	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, DIM * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, psi, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));
	
	cudaStatus = multi_Z_gate_host(gates, psi_gpu, DIM, n_qubits);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multi_Z_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(psi, psi_gpu, DIM * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	psi = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(psi_gpu);
	//free(state_cpu);
}

__global__ void multi_Pauli_gate_gpu(
	int* gates, ITYPE bit_mask_XY, int* num_pauli_op, ITYPE DIM, GTYPE *psi_gpu, int n_qubits
	){
	ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
	ITYPE IZ_state, XY_state, prev_state;
	int target_qubit_index, IZ_itr, XY_itr;
	GTYPE tmp_psi, tmp_prev_state_psi, tmp_state_psi;
	ITYPE state = 0;
	int num_y1 = 0;
	int num_z1 = 0;
	int i_cnt = 0;
	int minus_cnt = 0;
	if (idx < (DIM >> 1)){
		IZ_state = idx & ((1 << (num_pauli_op[0] + num_pauli_op[3])) - 1);
		XY_state = idx >> (num_pauli_op[0] + num_pauli_op[3]);
		IZ_itr = (num_pauli_op[0] + num_pauli_op[3]) - 1;
		XY_itr = (num_pauli_op[1] + num_pauli_op[2]) - 1;
		for (int i = 0; i < n_qubits; ++i){
			target_qubit_index = n_qubits - 1 - i;
			switch (gates[target_qubit_index]){
			case 0:
				if ((IZ_state >> IZ_itr) & 1) state += (1LL << target_qubit_index);
				--IZ_itr;
				break;
			case 1:
				if ((XY_state >> XY_itr) & 1) state += (1LL << target_qubit_index);
				--XY_itr;
				break;
			case 2:
				if ((XY_state >> XY_itr) & 1){
					++minus_cnt;
					++num_y1;
					state += (1LL << target_qubit_index);
				}
				--XY_itr;
				++i_cnt;
				break;
			case 3:
				if ((IZ_state >> IZ_itr) & 1){
					++minus_cnt;
					++num_z1;
					state += (1LL << target_qubit_index);
				}
				--IZ_itr;
				break;
			}
		}
		prev_state = state;
		state = state^bit_mask_XY;
		tmp_prev_state_psi = psi_gpu[prev_state];
		tmp_state_psi = psi_gpu[state];
		tmp_psi = psi_gpu[state];
		psi_gpu[state] = psi_gpu[prev_state];
		psi_gpu[prev_state] = tmp_psi;
		if (minus_cnt & 1) psi_gpu[state] = make_cuDoubleComplex(-psi_gpu[state].x, -psi_gpu[state].y);
		if (i_cnt & 1){
			psi_gpu[prev_state] = make_cuDoubleComplex(psi_gpu[prev_state].y, psi_gpu[prev_state].x);
			psi_gpu[state] = make_cuDoubleComplex(psi_gpu[state].y, psi_gpu[state].x);
		}
		if ((i_cnt >> 1) & 1){
			psi_gpu[state] = make_cuDoubleComplex(-psi_gpu[state].x, -psi_gpu[state].y);
			psi_gpu[prev_state] = make_cuDoubleComplex(-psi_gpu[prev_state].x, -psi_gpu[prev_state].y);
		}
		minus_cnt = (num_pauli_op[2] - num_y1) + num_z1;
		if (minus_cnt & 1) psi_gpu[prev_state] = make_cuDoubleComplex(-psi_gpu[prev_state].x, -psi_gpu[prev_state].y);
	}
}


__host__ cudaError multi_Pauli_gate_host(int* gates, GTYPE *psi_gpu, ITYPE DIM, int n_qubits){
	cudaError cudaStatus;
	int num_pauli_op[4] = { 0, 0, 0, 0 };
	for (int i = 0; i < n_qubits; ++i) ++num_pauli_op[gates[i]];
	ITYPE bit_mask_Z = 0;
	for (int i = 0; i < n_qubits; ++i){
		if (gates[i] == 3) bit_mask_Z ^= (1 << i);
	}

	if (num_pauli_op[1] == 0 && num_pauli_op[2]==0){
		unsigned int block = DIM <= 1024 ? DIM : 1024;
		unsigned int grid = DIM / block;
		multi_Z_gate_gpu << <grid, block>> >(bit_mask_Z, DIM, psi_gpu);
		cudaStatus = cudaGetLastError();
		//cudaStatus = call_multi_Z_gate_gpu(gates, psi_gpu, DIM, n_qubits);
		return cudaStatus;
	}
	ITYPE bit_mask_XY = 0;
	for (int i = 0; i < n_qubits; ++i){
		if (gates[i] == 1 || gates[i]==2) bit_mask_XY ^= (1 << i);
	}

	int *gates_gpu, *num_pauli_op_gpu;
	checkCudaErrors(cudaMalloc((void**)&gates_gpu, 4 * sizeof(int)));
	checkCudaErrors(cudaMemcpy(gates_gpu, gates, 4 * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&num_pauli_op_gpu, 4 * sizeof(int)));
	checkCudaErrors(cudaMemcpy(num_pauli_op_gpu, num_pauli_op, 4 * sizeof(int), cudaMemcpyHostToDevice));
	
	ITYPE half_dim = DIM >> 1;
	unsigned int block = half_dim <= 1024 ? half_dim : 1024;
	unsigned int grid = DIM / block;
	multi_Pauli_gate_gpu << <grid, block >> >(gates_gpu, bit_mask_XY, num_pauli_op_gpu, DIM, psi_gpu, n_qubits);
	//multi_Pauli_gate_gpu << <1, DIM >> >(gates_gpu, bit_mask_XY, num_pauli_op_gpu, DIM, psi_gpu, n_qubits);
	cudaStatus = cudaGetLastError();

Error:
	cudaFree(gates_gpu);
	cudaFree(num_pauli_op_gpu);
	return cudaStatus;
}

extern "C"
void multi_Pauli_gate(int* gates, void *psi, ITYPE DIM, int n_qubits){
	GTYPE *psi_gpu;
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(psi);
	cudaError_t cudaStatus;
	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, DIM * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, psi, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));
	
	cudaStatus = multi_Pauli_gate_host(gates, psi_gpu, DIM, n_qubits);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multi_Pauli_gate_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(psi, psi_gpu, DIM * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	psi = reinterpret_cast<void*>(state_cpu);

Error:
	cudaFree(psi_gpu);
	//free(state_cpu);
}

__device__ GTYPE multi_Z_get_expectation_value_device(ITYPE idx, ITYPE bit_mask, ITYPE DIM, GTYPE *psi_gpu)
{
	GTYPE ret=make_cuDoubleComplex(0.0,0.0);
	// ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int minus_cnt = 0;
	if (idx < DIM){
		GTYPE tmp_psi_gpu = psi_gpu[idx];
		minus_cnt = popcount64(idx&bit_mask);
		if (minus_cnt & 1) tmp_psi_gpu = make_cuDoubleComplex(-tmp_psi_gpu.x, -tmp_psi_gpu.y);
		ret = cuCmul(cuConj(psi_gpu[idx]), tmp_psi_gpu);
	}
	return ret;
}

__global__ void multi_Z_get_expectation_value_gpu(GTYPE *ret, ITYPE bit_mask, ITYPE DIM, GTYPE *psi_gpu)
{
	GTYPE sum = make_cuDoubleComplex(0.0, 0.0);
	GTYPE tmp;
	for (ITYPE state = blockIdx.x * blockDim.x + threadIdx.x; state < DIM; state += blockDim.x * gridDim.x) {
		tmp = multi_Z_get_expectation_value_device(state, bit_mask, DIM, psi_gpu);
		sum = cuCadd(sum, tmp);
	}
	sum.x = warpReduceSum_double(sum.x);
	sum.y = warpReduceSum_double(sum.y);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret[0].x), sum.x);
		atomicAdd_double(&(ret[0].y), sum.y);
	}
}

__device__ GTYPE multipauli_get_expectation_value_device(ITYPE idx,
	ITYPE* bit_mask_gpu, int* num_pauli_op, ITYPE DIM, GTYPE *psi_gpu, int n_qubits
	){
	GTYPE ret;
	GTYPE tmp_psi, tmp_prev_state_psi, tmp_state_psi;
	ITYPE prev_state, state;
	int num_y1 = 0;
	int num_z1 = 0;
	int i_cnt = 0;
	int minus_cnt = 0;
	if (idx < DIM){
		i_cnt = num_pauli_op[2];
		num_y1 = popcount64(idx&bit_mask_gpu[2]);
		num_z1 = popcount64(idx&bit_mask_gpu[3]);
		minus_cnt = num_y1 + num_z1;
		prev_state = idx;
		state = idx^(bit_mask_gpu[1]+bit_mask_gpu[2]);
		tmp_prev_state_psi = psi_gpu[prev_state];
		tmp_state_psi = psi_gpu[state];
		// swap
		tmp_psi = tmp_state_psi;
		tmp_state_psi = tmp_prev_state_psi;
		tmp_prev_state_psi = tmp_psi;
		if (minus_cnt & 1) tmp_state_psi = make_cuDoubleComplex(-tmp_state_psi.x, -tmp_state_psi.y);
		if (i_cnt & 1) tmp_state_psi = make_cuDoubleComplex(tmp_state_psi.y, tmp_state_psi.x);
		if ((i_cnt >> 1) & 1) tmp_state_psi = make_cuDoubleComplex(-tmp_state_psi.x, -tmp_state_psi.y);
		// tmp_state      -> state      : state*conj(tmp_state)
		// tmp_prev_state -> prev_state : prev_state*conj(tmp_prev_state)
		ret = cuCmul(tmp_state_psi, cuConj(psi_gpu[state]));
	}
	return ret;
}

__global__ void multipauli_get_expectation_value_gpu(GTYPE* ret,
	ITYPE* bit_mask_gpu, int* num_pauli_op_gpu, ITYPE DIM, GTYPE *psi_gpu, int n_qubits
	){
	GTYPE sum = make_cuDoubleComplex(0.0, 0.0);
	GTYPE tmp;
	int j = 0;
	for (ITYPE state = blockIdx.x * blockDim.x + threadIdx.x; state < DIM; state += blockDim.x * gridDim.x) {
		tmp = multipauli_get_expectation_value_device(state, bit_mask_gpu, num_pauli_op_gpu, DIM, psi_gpu, n_qubits);
		sum = cuCadd(sum, tmp);
	}
	sum.x = warpReduceSum_double(sum.x);
	sum.y = warpReduceSum_double(sum.y);
	if ((threadIdx.x & (warpSize - 1)) == 0){
		atomicAdd_double(&(ret[0].x), sum.x);
		atomicAdd_double(&(ret[0].y), sum.y);
	}
}

__host__ double multipauli_get_expectation_value_host(unsigned int* gates, GTYPE *psi_gpu, ITYPE DIM, int n_qubits){
	CTYPE ret[1];
	ret[0]=CTYPE(0,0);
	GTYPE *ret_gpu;
	cudaError cudaStatus;

	checkCudaErrors(cudaMalloc((void**)&ret_gpu, sizeof(GTYPE)));
	checkCudaErrors(cudaMemcpy(ret_gpu, ret, sizeof(GTYPE), cudaMemcpyHostToDevice));

	int num_pauli_op[4] = { 0, 0, 0, 0 };
	for (int i = 0; i < n_qubits; ++i) ++num_pauli_op[gates[i]];
	ITYPE bit_mask[4] = { 0, 0, 0, 0 };
	ITYPE *bit_mask_gpu;
	for (int i = 0; i < n_qubits; ++i){
		bit_mask[gates[i]] ^= (1 << i);
	}
	if (num_pauli_op[1] == 0 && num_pauli_op[2] == 0){
		unsigned int block = DIM <= 1024 ? DIM : 1024;
		unsigned int grid = DIM / block;
		multi_Z_get_expectation_value_gpu << <grid, block >> >(ret_gpu, bit_mask[3], DIM, psi_gpu);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "multipauli_get_expectation_value_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}
		checkCudaErrors(cudaMemcpy(ret, ret_gpu, sizeof(CTYPE), cudaMemcpyDeviceToHost));
		return ret[0].real();
	}
	
	int *num_pauli_op_gpu;
	checkCudaErrors(cudaMalloc((void**)&num_pauli_op_gpu, 4 * sizeof(int)));
	checkCudaErrors(cudaMemcpy(num_pauli_op_gpu, num_pauli_op, 4 * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&bit_mask_gpu, 4 * sizeof(ITYPE)));
	checkCudaErrors(cudaMemcpy(bit_mask_gpu, bit_mask, 4 * sizeof(ITYPE), cudaMemcpyHostToDevice));
	
	unsigned int block = DIM <= 1024 ? DIM : 1024;
	unsigned int grid = DIM / block;
	multipauli_get_expectation_value_gpu << <grid, block >> >(ret_gpu, bit_mask_gpu, num_pauli_op_gpu, DIM, psi_gpu, n_qubits);
	
	cudaStatus = cudaGetLastError();
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multipauli_get_expectation_value_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	checkCudaErrors(cudaMemcpy(ret, ret_gpu, sizeof(CTYPE), cudaMemcpyDeviceToHost));

Error:
	cudaFree(ret_gpu);
	cudaFree(num_pauli_op_gpu);
	cudaFree(bit_mask_gpu);
	return ret[0].real();
}

extern "C"
double multipauli_get_expectation_value(unsigned int* gates, void *psi, ITYPE DIM, int n_qubits){
	GTYPE *psi_gpu, *ret_gpu;
	CTYPE* state_cpu = reinterpret_cast<CTYPE*>(psi);
	cudaError_t cudaStatus;
	double ret;
	//GTYPE ret[1] = { make_cuDoubleComplex(0.0, 0.0) };

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&psi_gpu, DIM * sizeof(CTYPE)));
	checkCudaErrors(cudaMemcpy(psi_gpu, psi, DIM * sizeof(CTYPE), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMalloc((void**)&ret_gpu, sizeof(GTYPE)));
	//checkCudaErrors(cudaMemcpy(ret_gpu, ret, sizeof(GTYPE), cudaMemcpyHostToDevice));

	ret = multipauli_get_expectation_value_host(gates, psi_gpu, DIM, n_qubits);

	/*
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multipauli_get_expectation_value_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching multi_Pauli_gate!\n", cudaStatus);
		goto Error;
	}
	*/
	//checkCudaErrors(cudaMemcpy(ret, ret_gpu, sizeof(CTYPE), cudaMemcpyDeviceToHost));

Error:
	cudaFree(psi_gpu);
	//cudaFree(ret_gpu);
	//free(state_cpu);
	return ret;
}
