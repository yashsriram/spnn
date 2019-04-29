#include <spdlog/spdlog.h>
#include <stdlib.h>
#include <iostream>
#include "../matrix.hpp"
#include "../nn.hpp"

int main() {
  srand(42);
  spdlog::set_level(spdlog::level::info);
  spdlog::set_pattern("[%^%L%$][%t][%H:%M:%S.%f] %v");

  try {
    auto fnn = FullyConnectedNetwork();
    fnn.addLayer(728);
    fnn.addLayer(30);
    fnn.addLayer(10);
    fnn.compile();

    Matrix in(728, 1, "input");
    in.setOnes();
    /* std::cout << in << std::endl; */
    Matrix out = fnn.predict(in);
    out.name = "unnormalized log probabilities";
    std::cout << out << std::endl;

    Matrix target(10, 1, "target");
    target.setZeros();
    target.set(4, 0, 1);
    std::cout << target << std::endl;

    Matrix probabilities = out.softmax();
    std::cout << probabilities << std::endl;
    float loss = fnn.crossEntropyLoss(probabilities, target);
    spdlog::info("Cross entropy loss: {}", loss);

  } catch (std::string e) {
    spdlog::error(e);
  }

  return 0;
}
