all: cpu_serial.out cuda_parallel.out openmp.out cuda_serial.out openblas.out

cpu_serial.out: src/cpu_serial.cpp include/cpu_serial/*
	nvcc -x cu -I include src/cpu_serial.cpp -o cpu_serial.out

cuda_parallel.out: src/cuda_parallel.cpp include/cuda_parallel/*
	nvcc -x cu -I include src/cuda_parallel.cpp -o cuda_parallel.out

openmp.out: src/openmp.cpp include/openmp/*
	g++ -fopenmp -I include src/openmp.cpp -o openmp.out

openblas.out : src/openblas.cpp include/openblas/*
	g++ -std=c++11 -I include -I /opt/OpenBLAS/include src/openblas.cpp -o openblas.out -L/opt/OpenBLAS/lib -lopenblas -lpthread -static

cuda_serial.out: src/cuda_serial.cpp include/cuda_serial/*
	nvcc -x cu -I include src/cuda_serial.cpp -o cuda_serial.out

clean:
	rm *.out
