#ifndef ULYSSES_ALGORITHMS__ALGORITHM_STATE_H_
#define ULYSSES_ALGORITHMS__ALGORITHM_STATE_H_

#include <memory>

class Agent;

// The abstract class for an algorithm state
class AlgorithmState
{
public:
  typedef std::unique_ptr<AlgorithmState> uptr;
  typedef std::shared_ptr<AlgorithmState> sptr;
  
  AlgorithmState() { }
  
  // It initializes its private members.
  virtual void intialize(Agent& agent) = 0;
  
  // It resets the state to its standard state.
  virtual void reset(Agent& agent) = 0;
  
  // It performs the updated action on this state.
  virtual void updateState() = 0;  
};

#endif // ULYSSES_ALGORITHMS__ALGORITHM_STATE_H_
