

nvcc QCsim.cu update_ops_*.cu -shared -o libgpusim.so
g++ test.cpp libgpusim.so -o a.out
./a.out

