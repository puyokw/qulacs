#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include "util.h"
#include "util.cuh"
#include "util_common.h"
#include "update_ops_cuda.h"
#include <cublas_v2.h>
#include <stdio.h>

/**  vqcsim からの移植
 * perform multi_qubit_Pauli_gate with XZ mask.
 * 
 * This function assumes bit_flip_mask is not 0, i.e., at least one bit is flipped. If no bit is flipped, use multi_qubit_Pauli_gate_Z_mask.
 * This function update the quantum state with Pauli operation. 
 * bit_flip_mask, phase_flip_mask, global_phase_90rot_count, and pivot_qubit_index must be computed before calling this function.
 * See get_masks_from_*_list for the above four arguemnts.
 */
void multi_qubit_Pauli_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count,UINT pivot_qubit_index, CTYPE* state, ITYPE dim);
void multi_qubit_Pauli_rotation_gate_XZ_mask(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, CTYPE* state, ITYPE dim);
void multi_qubit_Pauli_gate_Z_mask(ITYPE phase_flip_mask, CTYPE* state, ITYPE dim);
void multi_qubit_Pauli_rotation_gate_Z_mask(ITYPE phase_flip_mask, double angle, CTYPE* state, ITYPE dim);

// multi_qubit_PauliZ_gate
__device__ void multi_qubit_Pauli_gate_Z_mask_device(ITYPE phase_flip_mask, GTYPE* state_gpu, ITYPE dim){
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	// loop varaibles
	const ITYPE loop_dim = dim;
	if(state_index<loop_dim){
		// determine parity
		UINT bit1_num = popcount64(state_index & phase_flip_mask);
		// set values
		if(bit1_num&1) state_gpu[state_index] = make_cuDoubleComplex(-1*cuCreal(state_gpu[state_index]), -1*cuCimag(state_gpu[state_index]));
	}
}

__global__ void multi_qubit_Pauli_gate_Z_mask_gpu(ITYPE phase_flip_mask, GTYPE* state_gpu, ITYPE dim){
	multi_qubit_Pauli_gate_Z_mask_device(phase_flip_mask, state_gpu, dim);
}

__host__ cudaError multi_qubit_Pauli_gate_Z_mask_host(ITYPE phase_flip_mask, GTYPE* state_gpu, ITYPE dim){
	cudaError cudaStatus;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	multi_qubit_Pauli_gate_Z_mask_gpu << <grid, block >> >(phase_flip_mask, state_gpu, dim);		
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multi_qubit_Pauli_gate_Z_mask_host launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

__device__ void multi_qubit_Pauli_gate_XZ_mask_device(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, GTYPE* state_gpu, ITYPE dim){
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	// pivot mask
	const ITYPE pivot_mask = 1ULL << pivot_qubit_index;
	// loop varaibles
	const ITYPE loop_dim = dim>>1;
	GTYPE PHASE_M90ROT[4] = { make_cuDoubleComplex(1.0,0.0), make_cuDoubleComplex(0.0,-1), make_cuDoubleComplex(-1,0.0), make_cuDoubleComplex(0.0,1)};

	if(state_index<loop_dim){
		// create base index
		ITYPE basis_0 = insert_zero_to_basis_index_device(state_index, pivot_qubit_index);

		// gather index
		ITYPE basis_1 = basis_0 ^ bit_flip_mask;

		// determine sign
		unsigned int sign_0 = popcount64(basis_0 & phase_flip_mask)&1;
		unsigned int sign_1 = popcount64(basis_1 & phase_flip_mask)&1;
		 
		// fetch values
		GTYPE cval_0 = state_gpu[basis_0];
		GTYPE cval_1 = state_gpu[basis_1];

		// set values
		state_gpu[basis_0] = cuCmul(cval_1, PHASE_M90ROT[(global_phase_90rot_count + sign_0*2)&3]); // a % 4 = a & (4-1)
		state_gpu[basis_1] = cuCmul(cval_0, PHASE_M90ROT[(global_phase_90rot_count + sign_1*2)&3]); // a % 4 = a & (4-1)
	}
}

__global__ void multi_qubit_Pauli_gate_XZ_mask_gpu(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, GTYPE* state_gpu, ITYPE dim){
	multi_qubit_Pauli_gate_XZ_mask_device(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state_gpu, dim);
}

__host__ cudaError multi_qubit_Pauli_gate_XZ_mask_host(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, GTYPE* state_gpu, ITYPE dim){
	cudaError cudaStatus;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	// multi_qubit_Pauli_gate_Z_mask_gpu << <grid, block >> >(phase_flip_mask, state_gpu, dim);		
	multi_qubit_Pauli_gate_XZ_mask_gpu<< <grid, block >> >(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state_gpu, dim);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multi_qubit_Pauli_gate_XZ_mask_host launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

/*
const CTYPE PHASE_90ROT[4] = {1., 1.i, -1, -1.i};
const CTYPE PHASE_M90ROT[4] = { 1., -1.i, -1, 1.i };
*/

__device__ void multi_qubit_Pauli_rotation_gate_XZ_mask_device(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, GTYPE* state_gpu, ITYPE dim){
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	// pivot mask
	ITYPE pivot_mask = 1ULL << pivot_qubit_index;
	// loop varaibles
	ITYPE loop_dim = dim>>1;

	// coefs
	double cosval = cos(angle);
	double sinval = sin(angle);
	//GTYPE PHASE_90ROT[4] = {make_cuDoubleComplex(1.0,0.0), make_cuDoubleComplex(0.0,1.0), make_cuDoubleComplex(-1.0,0.0), make_cuDoubleComplex(0.0,-1.0)};
	GTYPE PHASE_M90ROT[4] = { make_cuDoubleComplex(1.0,0.0), make_cuDoubleComplex(0.0,-1), make_cuDoubleComplex(-1,0.0), make_cuDoubleComplex(0.0,1)};
	if(state_index<loop_dim){
		// create base index
		ITYPE basis_0 = insert_zero_to_basis_index_device(state_index, pivot_qubit_index);
		// gather index
		ITYPE basis_1 = basis_0 ^ bit_flip_mask;
		// determine parity
		unsigned int bit_parity_0 = popcount64(basis_0 & phase_flip_mask)&1;
		unsigned int bit_parity_1 = popcount64(basis_1 & phase_flip_mask)&1;
		
		// fetch values        
		GTYPE cval_0 = state_gpu[basis_0];
		GTYPE cval_1 = state_gpu[basis_1];
		
		// set values
		GTYPE tmp =  cuCmul(make_cuDoubleComplex(sinval*cuCreal(cval_1), sinval*cuCimag(cval_1)), PHASE_M90ROT[ (global_phase_90rot_count + bit_parity_0*2)&3 ]);
		//state[basis_0] = cuCmul(cosval, cval_0) + 1.i * sinval * cval_1 * PHASE_M90ROT[ (global_phase_90rot_count + bit_parity_0*2)&3 ]; // % 4
		state_gpu[basis_0] = cuCadd(make_cuDoubleComplex(cosval*cuCreal(cval_0), cosval*cuCimag(cval_0)), cuCmul(tmp, make_cuDoubleComplex(0.0,1.0)));
		
		//state[basis_1] = cosval * cval_1 + 1.i * sinval * cval_0 * PHASE_M90ROT[ (global_phase_90rot_count + bit_parity_1*2)&3 ]; // % 4
		tmp =  cuCmul(make_cuDoubleComplex(sinval*cuCreal(cval_0), sinval*cuCimag(cval_0)), PHASE_M90ROT[(global_phase_90rot_count + bit_parity_1*2)&3 ]);
		state_gpu[basis_1] = cuCadd(make_cuDoubleComplex(cosval*cuCreal(cval_1), cosval*cuCimag(cval_1)), cuCmul(tmp, make_cuDoubleComplex(0.0, 1.0)) ); // % 4
	}
}

__global__ void multi_qubit_Pauli_rotation_gate_XZ_mask_gpu(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, GTYPE* state_gpu, ITYPE dim){
	multi_qubit_Pauli_rotation_gate_XZ_mask_device(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, angle, state_gpu, dim);
}

__host__ cudaError multi_qubit_Pauli_rotation_gate_XZ_mask_host(ITYPE bit_flip_mask, ITYPE phase_flip_mask, UINT global_phase_90rot_count, UINT pivot_qubit_index, double angle, GTYPE* state_gpu, ITYPE dim){
	cudaError cudaStatus;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	// multi_qubit_Pauli_gate_Z_mask_gpu << <grid, block >> >(phase_flip_mask, state_gpu, dim);	
	multi_qubit_Pauli_rotation_gate_XZ_mask_gpu<< <grid, block >> >(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, angle, state_gpu, dim);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multi_qubit_Pauli_rotation_gate_XZ_mask_host launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

__device__ void multi_qubit_Pauli_rotation_gate_Z_mask_device(ITYPE phase_flip_mask, double angle, GTYPE* state_gpu, ITYPE dim){
	// loop variables
	const ITYPE loop_dim = dim;
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	// coefs
	const double cosval = cos(angle);
	const double sinval = sin(angle);

	if(state_index<loop_dim){
		// determine sign
		UINT bit_parity = popcount64(state_index & phase_flip_mask)&1;
		int sign = 1 - 2*bit_parity;
		
		// set value
		state_gpu[state_index] = cuCmul(state_gpu[state_index], make_cuDoubleComplex(cosval, sign * sinval));
	}
}

__global__ void multi_qubit_Pauli_rotation_gate_Z_mask_gpu(ITYPE phase_flip_mask, double angle, GTYPE* state_gpu, ITYPE dim){
	multi_qubit_Pauli_rotation_gate_Z_mask_device(phase_flip_mask, angle, state_gpu, dim);
}

__host__ cudaError multi_qubit_Pauli_rotation_gate_Z_mask_host(ITYPE phase_flip_mask, double angle, GTYPE* state_gpu, ITYPE dim){
	cudaError cudaStatus;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	
	// multi_qubit_Pauli_gate_Z_mask_gpu << <grid, block >> >(phase_flip_mask, state_gpu, dim);	
	multi_qubit_Pauli_rotation_gate_Z_mask_gpu<< <grid, block >> >(phase_flip_mask, angle, state_gpu, dim);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multi_qubit_Pauli_rotation_gate_Z_mask_host launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}

void multi_qubit_Pauli_gate_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, GTYPE* state_gpu, ITYPE dim){
	// create pauli mask and call function
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_partial_list(target_qubit_index_list, Pauli_operator_type_list, target_qubit_index_count,
		&bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	if(bit_flip_mask == 0){
		multi_qubit_Pauli_gate_Z_mask_host(phase_flip_mask, state_gpu, dim);
	}else{
		multi_qubit_Pauli_gate_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state_gpu, dim);
	}
}

void multi_qubit_Pauli_gate_whole_list(const UINT* Pauli_operator_type_list, UINT qubit_count, GTYPE* state_gpu, ITYPE dim){
	 // create pauli mask and call function
	 ITYPE bit_flip_mask = 0;
	 ITYPE phase_flip_mask = 0;
	 UINT global_phase_90rot_count = 0;
	 UINT pivot_qubit_index = 0;
	 get_Pauli_masks_whole_list(Pauli_operator_type_list, qubit_count,
		 &bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	 if(bit_flip_mask == 0){
		 multi_qubit_Pauli_gate_Z_mask_host(phase_flip_mask, state_gpu, dim);
	 }else{
		 multi_qubit_Pauli_gate_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, state_gpu, dim);
	 }
 } 

void multi_qubit_Pauli_rotation_gate_partial_list(const UINT* target_qubit_index_list, const UINT* Pauli_operator_type_list, UINT target_qubit_index_count, double angle, GTYPE* state_gpu, ITYPE dim){
	 // create pauli mask and call function
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_partial_list(target_qubit_index_list, Pauli_operator_type_list, target_qubit_index_count,
		&bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	if(bit_flip_mask == 0){
		multi_qubit_Pauli_rotation_gate_Z_mask_host(phase_flip_mask, angle, state_gpu, dim);
	}else{
		multi_qubit_Pauli_rotation_gate_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index,angle, state_gpu, dim);
	}
}
 
void multi_qubit_Pauli_rotation_gate_whole_list(const UINT* Pauli_operator_type_list, UINT qubit_count, double angle, GTYPE* state_gpu, ITYPE dim){
	// create pauli mask and call function
	ITYPE bit_flip_mask = 0;
	ITYPE phase_flip_mask = 0;
	UINT global_phase_90rot_count = 0;
	UINT pivot_qubit_index = 0;
	get_Pauli_masks_whole_list(Pauli_operator_type_list, qubit_count,
		&bit_flip_mask, &phase_flip_mask, &global_phase_90rot_count, &pivot_qubit_index);
	if(bit_flip_mask == 0){
		multi_qubit_Pauli_rotation_gate_Z_mask_host(phase_flip_mask, angle, state_gpu, dim);
	}else{
		multi_qubit_Pauli_rotation_gate_XZ_mask_host(bit_flip_mask, phase_flip_mask, global_phase_90rot_count, pivot_qubit_index, angle, state_gpu, dim);
	}
}

__device__ void multi_qubit_dense_matrix_gate_device(UINT* target_qubit_index_list, UINT target_qubit_index_count, GTYPE* matrix_gpu, 
	GTYPE* d_buffer, ITYPE* matrix_mask_list, UINT* sorted_insert_index_list, GTYPE* state_gpu, ITYPE dim){
	ITYPE state_index = blockIdx.x * blockDim.x + threadIdx.x;
	ITYPE x, y;
	// matrix dim, mask, buffer
    const ITYPE matrix_dim = 1ULL << target_qubit_index_count;
    
    // loop variables
	const ITYPE loop_dim = dim >> target_qubit_index_count;
	if(state_index<loop_dim){
		// create base index
        ITYPE basis_0 = state_index;
        for(UINT cursor=0; cursor < target_qubit_index_count ; cursor++){
			UINT insert_index = sorted_insert_index_list[cursor];
			basis_0 = insert_zero_to_basis_index_device(basis_0, insert_index );
		}
		// compute matrix-vector multiply
		for(y = 0 ; y < matrix_dim ; ++y ){
			d_buffer[basis_0 ^ matrix_mask_list[y]]=make_cuDoubleComplex(0.0,0.0);
			for(x = 0 ; x < matrix_dim ; ++x){
				d_buffer[basis_0 ^ matrix_mask_list[y]] = cuCadd(d_buffer[basis_0 ^ matrix_mask_list[y]], 
					cuCmul(matrix_gpu[y*matrix_dim + x], state_gpu[ basis_0 ^ matrix_mask_list[x] ]));
			}
		}
		// set result
        for(y = 0 ; y < matrix_dim ; ++y){
			state_gpu[basis_0 ^ matrix_mask_list[y]] = d_buffer[basis_0 ^ matrix_mask_list[y]];
        }
    }
}
__global__ void multi_qubit_dense_matrix_gate_gpu(UINT* target_qubit_index_list, UINT target_qubit_index_count, GTYPE* matrix_gpu, 
	GTYPE* d_buffer, ITYPE* matrix_mask_list, UINT* sorted_insert_index_list, GTYPE* state_gpu, ITYPE dim){
		multi_qubit_dense_matrix_gate_device(target_qubit_index_list, target_qubit_index_count, matrix_gpu, 
			d_buffer, matrix_mask_list, sorted_insert_index_list, state_gpu, dim);
}

__host__ cudaError multi_qubit_dense_matrix_gate(UINT* target_qubit_index_list, UINT target_qubit_index_count, CTYPE* matrix, GTYPE* state_gpu, ITYPE dim){
	cudaError cudaStatus;
	UINT* d_target_qubit_index_list;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_target_qubit_index_list), target_qubit_index_count * sizeof(UINT) ));
	checkCudaErrors(cudaMemcpy(d_target_qubit_index_list, target_qubit_index_list, target_qubit_index_count * sizeof(UINT), cudaMemcpyHostToDevice));

	// matrix dim, mask, buffer
    ITYPE matrix_dim = 1ULL << target_qubit_index_count;
	ITYPE* h_matrix_mask_list = create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);
	ITYPE* d_matrix_mask_list;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_matrix_mask_list), matrix_dim * sizeof(ITYPE) ));
	checkCudaErrors(cudaMemcpy(d_matrix_mask_list, h_matrix_mask_list, matrix_dim * sizeof(ITYPE), cudaMemcpyHostToDevice));

    // insert index
    UINT* h_sorted_insert_index_list = create_sorted_ui_list(target_qubit_index_list, target_qubit_index_count);
	UINT* d_sorted_insert_index_list;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_sorted_insert_index_list), matrix_dim * sizeof(ITYPE) ));
	checkCudaErrors(cudaMemcpy(d_sorted_insert_index_list, h_sorted_insert_index_list, matrix_dim * sizeof(ITYPE), cudaMemcpyHostToDevice));

    // loop variables
	ITYPE loop_dim = dim >> target_qubit_index_count;
	
	GTYPE* d_buffer;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_buffer), matrix_dim * matrix_dim * sizeof(GTYPE) ));
	cudaMemset(d_buffer, 0, matrix_dim * matrix_dim * sizeof(GTYPE));

	GTYPE* matrix_gpu;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&matrix_gpu), matrix_dim *matrix_dim * sizeof(GTYPE) ));
	checkCudaErrors(cudaMemcpy(matrix_gpu, matrix, matrix_dim *matrix_dim * sizeof(GTYPE), cudaMemcpyHostToDevice));
	
	unsigned int block = loop_dim <= 1024 ? loop_dim : 1024;
	unsigned int grid = loop_dim / block;
	multi_qubit_dense_matrix_gate_gpu << <grid, block >> >(d_target_qubit_index_list, target_qubit_index_count,
		matrix_gpu, d_buffer, d_matrix_mask_list, d_sorted_insert_index_list, state_gpu, dim);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "multi_qubit_dense_matrix_gate launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	cudaFree(d_target_qubit_index_list);
	cudaFree(d_buffer);
	cudaFree(d_sorted_insert_index_list);
	cudaFree(d_matrix_mask_list);
	cudaFree(matrix_gpu);
	free((UINT*)h_sorted_insert_index_list);
	free((ITYPE*)h_matrix_mask_list);
	return cudaStatus;
}

// len(state_new) < len(state_original)
__global__ void copy_partial_state_short(ITYPE* matrix_mask_list, ITYPE matrix_dim, ITYPE basis_0, GTYPE* state_new, GTYPE* state_original){
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < matrix_dim){
		state_new[j] = state_original[basis_0 ^ matrix_mask_list[j]];
	}
}

// len(state_original) < len(state_new)
__global__ void copy_partial_state_long(ITYPE* matrix_mask_list, UINT matrix_dim, ITYPE basis_0, GTYPE* state_new, GTYPE* state_original){
	ITYPE j = blockIdx.x * blockDim.x + threadIdx.x;
	if(j < matrix_dim){
		state_new[basis_0 ^ matrix_mask_list[j]] = state_original[j];
	}
}

__host__ cudaError multi_qubit_dense_matrix_gate_cublas(UINT* target_qubit_index_list, UINT target_qubit_index_count, CTYPE* matrix, GTYPE* state_gpu, ITYPE dim){
	cudaError cudaStatus;
	// matrix dim(=len), mask
    ITYPE matrix_dim = 1ULL << target_qubit_index_count;
	ITYPE* h_matrix_mask_list = create_matrix_mask_list(target_qubit_index_list, target_qubit_index_count);
	ITYPE* d_matrix_mask_list;

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_matrix_mask_list), matrix_dim * sizeof(ITYPE) ));
	checkCudaErrors(cudaMemcpy(d_matrix_mask_list, h_matrix_mask_list, matrix_dim * sizeof(ITYPE), cudaMemcpyHostToDevice));

    // insert index
    UINT* sorted_insert_index_list = create_sorted_ui_list(target_qubit_index_list, target_qubit_index_count);

    // loop variables
    ITYPE loop_dim = dim >> target_qubit_index_count;
	ITYPE basis_0 = 0;
	for(UINT cursor=0; cursor < target_qubit_index_count ; cursor++){
		UINT insert_index = sorted_insert_index_list[cursor];
		basis_0 = insert_zero_to_basis_index(basis_0, insert_index );
	}
	
	GTYPE* tmp_state_gpu;
	checkCudaErrors(cudaMalloc((void**)&tmp_state_gpu, matrix_dim * sizeof(GTYPE)));
	// copy memory state_gpu to tmp_state_gpu at device
	int block = matrix_dim <= 1024 ? matrix_dim : 1024;
	int grid = matrix_dim / block;

	copy_partial_state_short<< <grid, block >> >(d_matrix_mask_list, matrix_dim, basis_0, tmp_state_gpu, state_gpu);
	cublas_zgemv_wrapper(matrix_dim, matrix, tmp_state_gpu);
	copy_partial_state_long<< <grid, block >> >(d_matrix_mask_list, matrix_dim, basis_0, state_gpu, tmp_state_gpu);

Error:
	cudaFree(tmp_state_gpu);
	cudaFree(d_matrix_mask_list);
	free((UINT*)sorted_insert_index_list);
	free((ITYPE*)h_matrix_mask_list);
	return cudaStatus;
}
