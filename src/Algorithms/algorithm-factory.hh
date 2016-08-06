#ifndef ULYSSES_KERNEL__ALGORITHMS__ALGORITHM_FACTORY_H_
#define ULYSSES_KERNEL__ALGORITHMS__ALGORITHM_FACTORY_H_

#include "Kernel/globals.hh"

#include <string>
#include <vector>

class Algorithm;
class Agent;

class AlgorithmFactory
{
public:
  static Algorithm* create(Agent& a, std::string type = "DPOP",
			   std::vector<std::string> params = std::vector<std::string>());  

private:
  AlgorithmFactory() { }
  DISALLOW_COPY_AND_ASSIGN(AlgorithmFactory);
};

#endif // ULYSSES_KERNEL__ALGORITHMS__ALGORITHM_FACTORY_H_