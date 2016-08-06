#ifndef ULYSSES_ALGORITHMS__DPOP__VALUE_PROPAGATION_H_
#define ULYSSES_ALGORITHMS__DPOP__VALUE_PROPAGATION_H_

#include "Algorithms/algorithm.hh"
#include "Algorithms/DPOP/value-msg-handler.hh"
#include <memory>

class Agent;
class ValueMsgHandler;
namespace CUDA {class DPOPstate; }


// It implements the DPOP UTIL propagation phase for a given agent.
// It joins the UTIL tables received from its children in the UTIL messages
// and computes the UTIL table to send to its parent. 
class ValuePropagation : public Algorithm
{
public:
  typedef std::unique_ptr<ValuePropagation> uptr;
  typedef std::shared_ptr<ValuePropagation> sptr;

  ValuePropagation(Agent& owner, std::shared_ptr<CUDA::DPOPstate> state);

  virtual ~ValuePropagation();
  
  virtual void initialize();

 // It finalizes the algorithm.
  virtual void finalize();

  // It returns true if the algorithm can be executed in this agent.
  // It *must* be callsed ONLY if the util propagation phase of the agent 
  // running this algorithm, has terminated.
  virtual bool canRun();

  // It executes the algorithm. It construct the VALUE message and computes
  // the vest values for the agent's variables.
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

  // It returns true if the handler has downloaded all the VALUE messages
  // received from each of the pseudo-parents and the parent of the pseudo-node
  //associated with the agent running this algorithm.
  bool recvAllMessages();


private:
  // The message handler associated to this algorithm
  ValueMsgHandler::sptr p_msg_handler;
  
  // The agent state, where it stores the UTIL message
  std::shared_ptr<CUDA::DPOPstate> p_state;
  
  // It signal wether the algorithm execution has terminated; 
  bool p_terminated;

};

#endif // ULYSSES_ALGORITHMS__DPOP__UTIL_PROPAGATION_H_

