all: cpu_serial.out cuda_parallel.out openmp.out cuda_serial.out openblas.out

cpu_serial.out: src/cpu_serial.cpp
	nvcc -x cu -I include src/cpu_serial.cpp -o cpu_serial.out

cuda_parallel.out: src/cuda_parallel.cpp
	nvcc -x cu -I include src/cuda_parallel.cpp -o cuda_parallel.out

cuda_serial.out: src/cuda_serial.cpp
	nvcc -x cu -I include src/cuda_serial.cpp -o cuda_serial.out

openmp.out: src/openmp.cpp
	g++ -fopenmp -I include src/openmp.cpp -o openmp.out

openblas.out : src/openblas.cpp
	export LD_LIBRARY_PATH=/opt/OpenBLAS/lib/
	g++ -std=c++11 -I include src/openblas.cpp -o openblas.out -I /opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib -lopenblas  -lpthread

clean:
	rm *.out
