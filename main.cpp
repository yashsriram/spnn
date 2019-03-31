#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <iostream>
#include "matrix.hpp"
#include "nn.hpp"

int main() {
  srand(42);
  spdlog::set_level(spdlog::level::debug);
  spdlog::set_pattern("[%^%L%$][%t][%H:%M:%S.%f] %v");

  try {
    auto fnn = FullyConnectedNetwork();
    fnn.addLayer(728);
    fnn.addLayer(30);
    fnn.addLayer(10);
    fnn.compile();

    Matrix in(1, 728, "input");
    in.setOnes();
    std::cout << in << std::endl;
    Matrix out = fnn.forwardPass(in);
    std::cout << out << std::endl;

  } catch (std::string e) {
    spdlog::error(e);
  }

  return 0;
}
