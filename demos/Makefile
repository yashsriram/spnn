all: mnist_serial.out mnist_parallel.out mnist_openmp.out mnist_gpuserial.out mnist_openblas.out

mnist_serial.out: mnist_serial.cpp
	nvcc -x cu mnist_serial.cpp -o mnist_serial.out

mnist_parallel.out: mnist_parallel.cpp
	nvcc -x cu mnist_parallel.cpp -o mnist_parallel.out

mnist_gpuserial.out: mnist_gpuserial.cpp
	nvcc -x cu mnist_gpuserial.cpp -o mnist_gpuserial.out

mnist_openmp.out: mnist_openmp.cpp
	g++ -fopenmp mnist_openmp.cpp -o mnist_openmp.out

mnist_openblas.out : mnist_openblas.cpp
	g++ -std=c++11 mnist_openblas.cpp -o mnist_openblas.out -I /opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib -lopenblas  -lpthread

clean:
	rm -f mnist_serial.out mnist_parallel.out mnist_openmp.out
