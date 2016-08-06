#ifndef ULYSSES_COMMUNICATION__MESSAGE_HANDLER_H_
#define ULYSSES_COMMUNICATION__MESSAGE_HANDLER_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "Communication/message.hh"
#include "Kernel/globals.hh"

class Agent;

class MessageHandler
{
public:
  typedef std::unique_ptr<MessageHandler> uptr;
  typedef std::shared_ptr<MessageHandler> sptr;
  enum status_t{k_none, k_active, k_terminated, k_wait};
  
  MessageHandler(Agent& a, std::string msgtype="")
    : p_owner(&a), p_status(k_none), p_signature(msgtype)
  { }

  virtual ~MessageHandler() 
  { }

  // It reads the stauts that it needs (if it does) and prepare the execution
  // of the algorithm accordingly.
  virtual void prcessStatus(std::unordered_map<std::string,status_t>& map) //= 0;
  { }
  
  // It processes the incoming message, which must be of type recognized by 
  // the message handler.
  virtual void processIncoming(Message::sptr msg) //= 0;
  { }
  
  // It decodes an incoming message into a message of type specific to that of
  // the derivred class.
  virtual Message* decodeMessage(Message::sptr msg) //= 0; 
  { return nullptr; }
  
  // It stores the message just received for further processing, and returns the
  // index of the array where the message has been stored (if appliable).
  virtual int storeReceivedMessage(Message::sptr msg) //= 0;
  { return 0; }
    
  virtual void uponActivation() { }
  
  virtual void uponTermination() { }
  
  
  // It returns the message handler signature
  std::string signature() const
  {
    return p_signature;
  }
  
  // It returns the current status of the message handler.
  status_t status() const
  { 
    return p_status; 
  }
  
  // It returns the agent owning this message handler.
  Agent& owner() const 
  {
    return *p_owner;
  }

  //////////////////////////////////// DEPRECATED
  // It prepare the outgoing message, based on the information 
  // collected from the last received message.
  virtual void prepareOutgoing() = 0;

  // It process the message(s) read from the inbox. It saves the messages in 
  // a local storage to be process at needed.
  virtual void processIncoming() = 0;
  
  // It sends the outgoing message to the agent with id given as a parameter.
  virtual void send(oid_t dest_id) = 0;
  /////////////////////////////////////
  

protected:
  // The agent owning this message handler.
  Agent* p_owner;
  
  // The current message handler status.
  status_t p_status;

  // The message handler signature, which must correspond to the type of the
  // message handled by this message handler.
  std::string p_signature;
};


#endif // ULYSSES_COMMUNICATION__MESSAGE_HANDLER_H_
