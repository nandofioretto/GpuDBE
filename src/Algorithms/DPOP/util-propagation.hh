#ifndef ULYSSES_ALGORITHMS__DPOP__UTIL_PROPAGATION_H_
#define ULYSSES_ALGORITHMS__DPOP__UTIL_PROPAGATION_H_

#include "Algorithms/algorithm.hh"
#include "Algorithms/Orderings/pseudo-tree-ordering.hh"

#include <memory>

class Agent;
class UtilMsgHandler;
class Codec;
class ValuePropagation;
namespace CUDA {class DPOPstate; }

typedef PseudoTreeOrdering PseudoNode;

// It implements the DPOP UTIL propagation phase for a given agent.
// It joins the UTIL tables received from its children in the UTIL messages
// and computes the UTIL table to send to its parent. 
class UtilPropagation : public Algorithm
{
public:
  typedef std::unique_ptr<UtilPropagation> uptr;
  typedef std::shared_ptr<UtilPropagation> sptr;
  
  typedef size_t code_t;
  typedef std::pair<code_t, std::vector<cost_t> > utilpair_t;
  friend class ValuePropagation;

  // Initializes the constraints which involve the variables in the sparator
  // of the agent running this algorithms.
  UtilPropagation(Agent& owner, std::shared_ptr<CUDA::DPOPstate> state);

  virtual ~UtilPropagation();
  
  // It initializes the algorithm: it registers the UTIL message handler in the
  // agent inbox, and it construct the local space as comibnation of all the 
  // possible value assignments to the boundary variables, optimizing on the 
  // private variables. It will be used in the UTIL message construction.
  virtual void initialize();

 // It initializes the algorithm.
  virtual void finalize();

  // It returns true if the algorithm can be executed in this agent.
  virtual bool canRun();

  // It executes the algorithm. It construct the UTIL message and merges the
  // UTIL messages received from its children optimizing over the possible 
  // combinations of the boundary values.
  virtual void run();
  
  // It stops the algorithm saving the current results  and states if provided
  // by the algorithm itself.
  virtual void stop()
  { }

  // It returns whether the algorithm has terminated.
  virtual bool terminated()
  {
    return p_terminated;
  }

  // It returns true if the handler has downloaded all UTIL messages
  // received from each of the children of the pseudo-node associated with the
  // agent running this algorithm.
  virtual bool recvAllMessages();

protected:
  // The message handler associated to this algorithm
  std::shared_ptr<UtilMsgHandler> p_msg_handler;
  
  // It signal wether the algorithm execution has terminated; 
  bool p_terminated;

  // The agent state, where it stores the UTIL message
  std::shared_ptr<CUDA::DPOPstate> p_state;
};

#endif // ULYSSES_ALGORITHMS__DPOP__UTIL_PROPAGATION_H_

