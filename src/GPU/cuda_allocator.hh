#ifndef CUDA_DATA_ALLOCATOR_HH
#define CUDA_DATA_ALLOCATOR_HH

class Agent;
class TableConstraint;

namespace CUDA
{
  class Allocator
  {
  public:
    // Allocate global data and constraints
    // GPU-TODO: to-be changed to Constant or Texture Memory 
    static void allocate_data();

    // Allocate agent a on Global Memroy
    // GPU-TODO: to-be changed to Constant or Texture Memory 
    static void allocate_agent(Agent& a);

    // Allocate constraint c on Device Global Memory
    // GPU-TODO: to-be changed to Constant or Texture Memory 
    static void allocate_constraint(TableConstraint& c);

  };
}

#endif
