#ifndef _UPDATE_OPS_CU_H_
#define _UPDATE_OPS_CU_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
//#include "util.h"
#include "util_common.h"

// update_ops_names.cu
__global__ void H_gate_gpu(unsigned int target_qubit_index, GTYPE *psi, ITYPE dim);
extern "C" DllExport __host__ cudaError H_gate_host(unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE dim);
extern "C" __declspec(dllexport) void H_gate(unsigned int target_qubit_index, void *psi, ITYPE dim);
__global__ void X_gate_gpu(unsigned int target_qubit_index, GTYPE *psi, ITYPE dim);
extern "C" DllExport __host__ cudaError X_gate_host(unsigned int target_qubit_index, void *state_gpu, ITYPE dim);
extern "C" __declspec(dllexport) void X_gate(unsigned int target_qubit_index, void *psi, ITYPE dim);
__global__ void Y_gate_gpu(unsigned int target_qubit_index, GTYPE *psi, ITYPE dim);
extern "C" DllExport __host__ cudaError Y_gate_host(unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE dim);
extern "C" __declspec(dllexport) void Y_gate(unsigned int target_qubit_index, void *psi, ITYPE dim);
__global__ void Z_gate_gpu(unsigned int target_qubit_index, GTYPE *psi, ITYPE DIM);
extern "C" DllExport __host__ cudaError Z_gate_host(unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE dim);
extern "C" __declspec(dllexport) void Z_gate(unsigned int target_qubit_index, void *psi, ITYPE DIM);

extern "C" DllExport __host__ cudaError U1_gate_host(double lambda, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE DIM);
extern "C" DllExport void U1_gate(double lambda, unsigned int target_qubit_index, void *psi, ITYPE DIM);
extern "C" DllExport __host__ cudaError U2_gate_host(double lambda, double phi, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE DIM);
extern "C" DllExport void U2_gate(double lambda, double phi, unsigned int target_qubit_index, void *psi, ITYPE DIM);
extern "C" DllExport __host__ cudaError U3_gate_host(double lambda, double phi, double theta, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE DIM);
extern "C" DllExport void U3_gate(double lambda, double phi, double theta, unsigned int target_qubit_index, void *psi, ITYPE DIM);


__global__ void CZ_gate_gpu(unsigned int control_qubit_indexr, unsigned int target_qubit_index, GTYPE *psi, ITYPE DIM);
extern "C" DllExport __host__ cudaError CZ_gate_host(unsigned int control_qubit_indexr, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE dim);
extern "C" DllExport void CZ_gate(unsigned int control_qubit_indexr, unsigned int target_qubit_index, void *psi, ITYPE DIM);
__global__ void CNOT_gate_gpu(unsigned int control_qubit_indexr, unsigned int target_qubit_index, GTYPE *psi, ITYPE dim);
extern "C" DllExport __host__ cudaError CNOT_gate_host(unsigned int control_qubit_indexr, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE dim);
extern "C" DllExport void CNOT_gate(unsigned int control_qubit_indexr, unsigned int target_qubit_index, void *psi, ITYPE DIM);
__global__ void SWAP_gate_gpu(unsigned int i, unsigned int k, GTYPE *psi_gpu, ITYPE dim);
extern "C" DllExport __host__ cudaError SWAP_gate_host(unsigned int control_qubit_indexr, unsigned int target_qubit_index, GTYPE *psi_gpu, ITYPE dim);
extern "C" DllExport void SWAP_gate(unsigned int control_qubit_indexr, unsigned int target_qubit_index, void *psi, ITYPE DIM);


// update_ops_single
extern "C" DllExport __host__ cudaError RX_gate_host(UINT target_qubit_index, double angle, GTYPE* state_gpu, ITYPE dim);
extern "C" DllExport void RX_gate(UINT target_qubit_index, double angle, void* state, ITYPE dim);
extern "C" DllExport __host__ cudaError RY_gate_host(UINT target_qubit_index, double angle, GTYPE* state_gpu, ITYPE dim);
extern "C" DllExport void RY_gate(UINT target_qubit_index, double angle, void* state, ITYPE dim);
extern "C" DllExport __host__ cudaError RZ_gate_host(UINT target_qubit_index, double angle, GTYPE* state_gpu, ITYPE dim);
extern "C" DllExport void RZ_gate(UINT target_qubit_index, double angle, void* state, ITYPE dim);
extern "C" DllExport __host__ cudaError S_gate_host(UINT target_qubit_index, GTYPE* state, ITYPE dim);
extern "C" DllExport void S_gate(UINT target_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ cudaError Sdag_gate_host(UINT target_qubit_index, GTYPE* state_gpu, ITYPE dim);
extern "C" DllExport void Sdag_gate(UINT target_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ cudaError T_gate_host(UINT target_qubit_index, GTYPE* state_gpu, ITYPE dim);
extern "C" DllExport void T_gate(UINT target_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ cudaError Tdag_gate_host(UINT target_qubit_index, GTYPE* state_gpu, ITYPE dim);
extern "C" DllExport void Tdag_gate(UINT target_qubit_index, void* state, ITYPE dim);
extern "C" DllExport __host__ cudaError sqrtX_gate_host(UINT target_qubit_index, GTYPE* state_gpu, ITYPE dim);
extern "C" DllExport void sqrtX_gate(UINT target_qubit_index, void* state, ITYPE dim);
__host__ cudaError sqrtXdag_gate_host(UINT target_qubit_index, GTYPE* state_gpu, ITYPE dim);
extern "C" DllExport void sqrtXdag_gate(UINT target_qubit_index, void* state, ITYPE dim);
__host__ cudaError sqrtY_gate_host(UINT target_qubit_index, GTYPE* state_gpu, ITYPE dim);
extern "C" DllExport void sqrtY_gate(UINT target_qubit_index, void* state, ITYPE dim);
__host__ cudaError sqrtYdag_gate_host(UINT target_qubit_index, GTYPE* state_gpu, ITYPE dim);
extern "C" DllExport void sqrtYdag_gate(UINT target_qubit_index, void* state, ITYPE dim);

extern "C" DllExport void single_qubit_Pauli_gate(unsigned int target_qubit_index, unsigned int Pauli_operator_type, void *state, ITYPE dim);

extern "C" DllExport cudaError single_qubit_Pauli_rotation_gate_host(unsigned int target_qubit_index, unsigned int op_idx, double angle, void *state_gpu, ITYPE dim);
//extern "C" DllExport __host__ cudaError single_qubit_Pauli_rotation_gate_host(unsigned int target_qubit_index, unsigned int op_idx, double angle, GTYPE *psi_gpu, ITYPE dim);
extern "C" DllExport void single_qubit_Pauli_rotation_gate(unsigned int target_qubit_index, unsigned int op_idx, double angle, void *psi, ITYPE DIM);

__global__ void single_qubit_dense_matrix_gate_gpu(unsigned int target_qubit_index,GTYPE matrix[4], GTYPE *state_gpu, ITYPE dim);
__host__ cudaError single_qubit_dense_matrix_gate_host(unsigned int target_qubit_index, CTYPE matrix[4], GTYPE *state_gpu, ITYPE dim);
extern "C" DllExport void single_qubit_dense_matrix_gate(unsigned int target_qubit_index, const CTYPE matrix[4], void *state, ITYPE DIM);

__device__ void single_qubit_phase_gate_device(unsigned int target_qubit_index, GTYPE phase, GTYPE *state_gpu, ITYPE dim);
__global__ void single_qubit_phase_gate_gpu(unsigned int target_qubit_index, GTYPE phase, GTYPE *state_gpu, ITYPE dim);
__host__ cudaError single_qubit_phase_gate_host(unsigned int target_qubit_index, CTYPE phase, GTYPE *state_gpu, ITYPE dim);
extern "C" DllExport void single_qubit_phase_gate(unsigned int target_qubit_index, CTYPE phase, void *state, ITYPE dim);

__global__ void single_qubit_diagonal_matrix_gate_gpu(unsigned int target_qubit_index,GTYPE matrix[2], GTYPE *state_gpu, ITYPE dim);
__host__ cudaError single_qubit_diagonal_matrix_gate_host(unsigned int target_qubit_index, const CTYPE diagonal_matrix[2], GTYPE *state_gpu, ITYPE dim);
extern "C" DllExport void single_qubit_diagonal_matrix_gate(unsigned int target_qubit_index, const CTYPE diagonal_matrix[2], void *state, ITYPE dim);

__device__ void single_qubit_control_single_qubit_dense_matrix_gate_device(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, GTYPE matrix[4], GTYPE *state, ITYPE dim);
__global__ void single_qubit_control_single_qubit_dense_matrix_gate_gpu(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, GTYPE matrix_gpu[4], GTYPE *state_gpu, ITYPE dim);
__host__ cudaError single_qubit_control_single_qubit_dense_matrix_gate_host(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, CTYPE matrix[4], GTYPE *state_gpu, ITYPE dim);
extern "C" DllExport void single_qubit_control_single_qubit_dense_matrix_gate(unsigned int control_qubit_index, unsigned int control_value, unsigned int target_qubit_index, CTYPE matrix[4], void *state, ITYPE dim);

// multi qubit
extern "C" DllExport void multi_qubit_Pauli_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,UINT pivot_qubit_index, CTYPE* state, ITYPE dim);

__host__ cudaError multi_qubit_dense_matrix_gate_cublas(UINT* target_qubit_index_list, UINT target_qubit_index_count, CTYPE* matrix, GTYPE* state_gpu, ITYPE dim);
__host__ cudaError multi_qubit_dense_matrix_gate(UINT* target_qubit_index_list, UINT target_qubit_index_count, CTYPE* matrix, GTYPE* state_gpu, ITYPE dim);

#endif // #ifndef _UPDATE_OPS_CU_H_
