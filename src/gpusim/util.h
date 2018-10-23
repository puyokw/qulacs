#ifndef _UTIL_CU_H_
#define _UTIL_CU_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
//#include <cuda_runtime.h>
//#include <cuda.h>

#include <complex>
#include "util_common.h"

inline void checkCudaErrors(const cudaError error);
inline void __cudaSafeCall(cudaError err, const char *file, const int line);
inline __device__ int popcount64(ITYPE b);
inline __device__ int popcount32(unsigned int b);
inline __device__ double atomicAdd_double(double* address, double val);
inline __device__ double __shfl_down_double(double var, unsigned int srcLane, int width);
inline __device__ int warpReduceSum(int val);
inline __device__ double warpReduceSum_double(double val);
inline __global__ void deviceReduceWarpAtomicKernel(int *in, int* out, ITYPE N);
inline __global__ void deviceReduceWarpAtomicKernel(double *in, double* out, ITYPE N);
inline __global__ void deviceReduceWarpAtomicKernel(GTYPE *in, GTYPE* out, ITYPE N);
inline __device__ void deviceReduceWarpAtomicKernel_device(GTYPE *in, GTYPE* out, ITYPE N);
inline __global__ void init_qstate(GTYPE* state_gpu, ITYPE dim);

extern "C" DllExport void get_quantum_state(GTYPE* psi_gpu, void* psi_cpu_copy, ITYPE dim);
//extern "C" DllExport CTYPE* get_quantum_state(GTYPE* psi_gpu, ITYPE dim);
extern "C" DllExport GTYPE* allocate_quantum_state(ITYPE dim);
extern "C" DllExport void initialize_quantum_state(GTYPE *state_gpu, ITYPE dim);
inline void release_quantum_state(CTYPE* state);
inline void memcpy_quantum_state_HostToDevice(CTYPE* state_cpu, GTYPE* state_gpu, ITYPE dim);

inline __device__ ITYPE insert_zero_to_basis_index_device(ITYPE basis_index, unsigned int qubit_index);
void get_Pauli_masks_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, 
    ITYPE* bit_flip_mask, ITYPE* phase_flip_mask, UINT* global_phase_90rot_count, UINT* pivot_qubit_index);
void get_Pauli_masks_whole_list(const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, 
    ITYPE* bit_flip_mask, ITYPE* phase_flip_mask, UINT* global_phase_90rot_count, UINT* pivot_qubit_index);

int cublass_zgemm_wrapper(ITYPE n, CTYPE alpha, const CTYPE *h_A, const CTYPE *h_B, CTYPE beta, CTYPE *h_C);
int cublas_zgemv_wrapper(ITYPE n, CTYPE alpha, const CTYPE *h_A, const CTYPE *h_x, CTYPE beta, CTYPE *h_y);
int cublas_zgemv_wrapper(ITYPE n, const CTYPE *h_matrix, GTYPE *d_state);
UINT* create_sorted_ui_list(const UINT* array, size_t size);
ITYPE* create_matrix_mask_list(const UINT* qubit_index_list, UINT qubit_index_count);
ITYPE insert_zero_to_basis_index(ITYPE basis_index, unsigned int qubit_index);


#endif // #ifndef _QCUDASIM_UTIL_H_
