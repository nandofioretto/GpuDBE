#ifndef ULYSSES_ALGORITHMS__DPOP__DPOP_H_
#define ULYSSES_ALGORITHMS__DPOP__DPOP_H_

#include <memory>

#include "Algorithms/algorithm.hh"

class PseudoTreeConstruction;
class UtilPropagation;
class ValuePropagation;

namespace CUDA {class DPOPstate;}

// The DPOP Algorithm.
class DPOP : public Algorithm 
{
public:
  
  DPOP(Agent& owner);

  virtual ~DPOP();

  // It initializes the algorithm.
  virtual void initialize();

 // It initializes the algorithm.
  virtual void finalize();

  // It returns true if the algorithm can be executed in this agent.
  virtual bool canRun()
  { 
    return true; 
  }

  // It executes the algorithm.
  virtual void run();

  // It stops the algorithm saving the current results  and states if provided
  // by the algorithm itself.
  virtual void stop();

  // It returns whether the algorithm has terminated.
  virtual bool terminated()
  { 
    return false;
  }


private:
  
  // It holds true if the algorithm is terminated for this agent
  bool p_terminated;

  // wheather the agent has been allocated on device.
  bool p_on_device;
  
  // The Distributed Pseudo Tree construction Phase
  std::unique_ptr<PseudoTreeConstruction> p_pt_construction_phase;

  // The DPOP Util Propagation Phase
  std::unique_ptr<UtilPropagation> p_util_propagation_phase;

  // The DPOP Value Propagation Phase
  std::unique_ptr<ValuePropagation> p_value_propagation_phase;

  // The agent state, where it stores the UTIL message
  std::shared_ptr<CUDA::DPOPstate> p_state;

};

#endif // ULYSSES_ALGORITHMS__DPOP__DPOP_H_
