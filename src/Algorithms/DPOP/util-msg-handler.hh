#ifndef ULYSSES_ALGORITHMS__DPOP_UTIL_MSG_HANDLER_H_
#define ULYSSES_ALGORITHMS__DPOP_UTIL_MSG_HANDLER_H_

#include <memory>
#include <vector>

#include "Algorithms/DPOP/util-msg.hh"
#include "Algorithms/Orderings/pseudo-tree-ordering.hh"
#include "Communication/message-handler.hh"
#include "Kernel/globals.hh"

class Agent;

namespace CUDA{ class DPOPstate; }


typedef PseudoTreeOrdering PseudoNode;

// The message handler associated to messages of type: UTIL
class UtilMsgHandler : public MessageHandler
{
public:
  typedef std::unique_ptr<UtilMsgHandler> uptr;
  typedef std::shared_ptr<UtilMsgHandler> sptr;
  
  UtilMsgHandler(Agent& a);

  virtual ~UtilMsgHandler();
  
  // It downloads all the messages of type UTIL from the inbox of the
  // associated agent.
  virtual void processIncoming();

  // It prepare the outgoing message, based on the information 
  // collected from the received messages and the agent search.
  virtual void prepareOutgoing();

  // It sends the outgoing message to the agent with id the one
  // given as a parameter.
  virtual void send(oid_t dest_id);

  // It returns the outgoing message
  UtilMsg& outgoing() const
  {
    ASSERT(p_outgoing, "Outgoing message was not allocated");
    return *p_outgoing;
  }

  // It sets the variables for the outgoing message as the list of 
  // those one given as parameter.
  // It also initializes the auxiliary data structures used to optimze
  // over the messages received.
  // The vectors b_vars and util_vars, *must* be order lexicographicly
  void initialize(/*util_table_t* table_ptr, Codec::sptr rows_code*/);

  void setMsgContent(CUDA::DPOPstate& state);


  // It returns the next incoming message of type "UTIL"
  std::vector<UtilMsg::sptr >& received()
  {
    return p_received;
  }

  bool recvAllMessages();
  
  // Sums the costs of the tuple of values given as parameter, from 
  // each of the UTIL message received. 
  //
  // @return The cost of the utilities assicated to the input tuple, if it is 
  //         finite, or +/- Constants::infinity otherwise.
  cost_t msgCosts(std::vector<int> b_e_values);


private:
  
  // The messages received, and saved here as a store.
  std::vector<UtilMsg::sptr> p_received;

  // The outgoing message
  UtilMsg::uptr p_outgoing;

  // map[ m ][ v_m ] = idx
  // where m = the m-th message in 'p_received'
  //       v_m = the v_m-th variable in the variables of the message m
  //       idx = the index of the list of values V which is used to build 
  //       during the UTIL optimization phase. 
  std::vector<std::vector<int> > p_msgvars2validx;

  // The i-th p_query corresponds to the p_query for the i-th UTIL message
  // received
  std::vector<std::vector<int> > p_query;
};

#endif // ULYSSES_ALGORITHMS__DPOP_UTIL_MSG_HANDLER_H_
