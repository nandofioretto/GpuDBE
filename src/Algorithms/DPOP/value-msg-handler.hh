#ifndef ULYSSES_ALGORITHMS__DPOP__VALUE_MSG_HANDLER_H_
#define ULYSSES_ALGORITHMS__DPOP__VALUE_MSG_HANDLER_H_

#include "Kernel/globals.hh"
#include "Communication/message-handler.hh"
#include "Algorithms/Orderings/pseudo-tree-ordering.hh"
#include "Algorithms/DPOP/value-msg.hh"

#include <memory>
#include <vector>

class Agent;
//class DPOPstate;
namespace CUDA {class DPOPstate; }

// The message handler associated to messages of type: UTIL
class ValueMsgHandler : public MessageHandler
{
public:
  typedef std::unique_ptr<ValueMsgHandler> uptr;
  typedef std::shared_ptr<ValueMsgHandler> sptr;

  ValueMsgHandler(Agent& a);

  virtual ~ValueMsgHandler();
  
  // It Creates the (empty) outgoing messages, one for each of the agent's
  // children.
  void initialize(/*std::shared_ptr<DPOPstate> p_state*/);

  // It downloads all the messages of type VALUE from the inbox of the 
  // associated agent.
  virtual void processIncoming() {}
  void processIncoming(std::shared_ptr<CUDA::DPOPstate> p_dpop_state);

  // It prepare the outgoing messages using the information collected from the 
  // received messages and the agent search.
  // The i-th outgoing message will contain all the boundary which this agent 
  // shares with the destination agent of the i-th outgoing message.
  virtual void prepareOutgoing() {}
  void prepareOutgoing(std::shared_ptr<CUDA::DPOPstate> p_dpop_state);

  // It sends the outgoing message to the agent with id the one given as a
  // parameter. It the dest_id has the default value, then all messages 
  // will be sent.
  virtual void send(oid_t dest_id=Constants::nullid);

  // It returns the i-th outgoing message
  ValueMsg& outgoing(int i) const
  {
    ASSERT( p_outgoing[ i ], "Outgoing message was not allocated");
    return *p_outgoing[ i ];
  }
  
  // The number of outgoing messages.
  size_t nbOutgoing() const
  {
    return p_outgoing.size();
  }

  // It returns the VALUE messages recevied from its parent
  ValueMsg::sptr& received()
  {
    return p_received;
  }

  // Returns true if the agent has received all messages form its parent and its
  // pseudo-parents
  bool recvAllMessages();
  
  // It returns true if the object has been initialized 
  bool initialized() const
  {
    return p_initialized;
  }

private:
  // The messages received, and saved here as a store.
  ValueMsg::sptr p_received;
  
  // The outgoing messages (one per each children and pseudo-children)
  std::vector<ValueMsg::uptr > p_outgoing;

  // std::shared_ptr<DPOPstate> p_dpop_state;

  // It marks whether the initalization has been performed
  bool p_initialized;

};

#endif // ULYSSES_ALGORITHMS__DPOP_UTIL_MSG_HANDLER_H_
