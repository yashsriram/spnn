#include <stdlib.h>
#include <iostream>
#include "../matrix.hpp"

int main() {
  /* set random seed; this controls randomness */
  srand(42);

  Matrix A = Matrix(3, 2);
  A.setUniform(-1, 1);
  std::cout << A << std::endl;

  /* set random seed; this controls randomness */
  srand(42);

  Matrix B = Matrix(3, 2);
  B.setUniform(-1, 1);
  std::cout << B << std::endl;

  return 0;
}
