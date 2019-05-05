# Points
* Things to consider during analysis
  * correctness (> 10% accuracy)
  * repeatablity (nothing fancy)
  * memory check (no mem leaks or other bad stuff using valgrind --tool=memcheck)
  * time profile (nvprof)
* Initialization done uniformly in -1 to 1
* Layers are numbered from 0 i.e. first hidden layer is layer 1
* Versions
  * CPU - Serial
  * GPU - Serial
  * GPU - kernels
  * openMP

# Todo
- [x] Control size of name field
- [x] Impl loss function
- [x] Remove memleaks from step_train
- [x] Batch gradient descent: fix loss decrement and check backprop
- [x] Normalize
- [x] Get MNIST data
- [x] Profile
