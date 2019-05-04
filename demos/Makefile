all: mnist_serial.out mnist_parallel.out mnist_openmp.out

mnist_serial.out: mnist_serial.cpp
	nvcc -x cu mnist_serial.cpp -o mnist_serial.out

mnist_parallel.out: mnist_parallel.cpp
	nvcc -x cu mnist_parallel.cpp -o mnist_parallel.out

mnist_openmp.out: mnist_openmp.cpp
	g++ -fopenmp mnist_openmp.cpp -o mnist_openmp.out
