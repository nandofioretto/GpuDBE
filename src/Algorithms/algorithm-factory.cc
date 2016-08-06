#include "Algorithms/algorithm-factory.hh"
#include "Algorithms/algorithm.hh"
#include "Algorithms/DPOP/dpop.hh"
#include "Kernel/agent.hh"

#include <string>
#include <vector>

Algorithm* AlgorithmFactory::create
(Agent& a, std::string type, std::vector<std::string> params)
{
  if (type.compare("DPOP") == 0 or type.compare("dpop") == 0) {
    return new DPOP(a);
  }
  return nullptr;
}
