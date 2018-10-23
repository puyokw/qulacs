
#include "update_ops_cuda.h"

int main(){
	GTYPE* state = allocate_quantum_state_host(1024);
	X_gate_host(0,state);
	return 0;
}