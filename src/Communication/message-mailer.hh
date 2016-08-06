#ifndef ULYSSES_COMMUNICATION__MESSAGE_MAILER_H_
#define ULYSSES_COMMUNICATION__MESSAGE_MAILER_H_

#include <memory>
#include <string>
#include <unordered_map>

class Agent;

class MessageMailer
{
public:
  typedef std::unique_ptr<MessageMailer> uptr;
  typedef std::shared_ptr<MessageMailer> sptr;
  enum status_t{k_none, k_active, k_sent, k_wait};
  
  MessageMailer(Agent& a, std::string msgtype="")
    : p_owner(&a), p_status(k_none), p_signature(msgtype)
  { }

  virtual ~MessageMailer() 
  { }

  // It reads the stauts that it needs (if it does) and prepare the execution
  // of the algorithm accordingly.
  virtual void prcessStatus(std::unordered_map<std::string,status_t>& map) = 0;
  
  // It creates a new outgoing message.
  virtual void processOutgoing() = 0;
  
  // It sends the outgoing message to the agent with id given as a parameter.
  virtual void send(size_t dest_id) = 0;
  
  // It decodes an incoming message into a message of type specific to that of
  // the derivred class.
  // virtual Message* encodeMessage() //= 0;
  // { return nullptr; }
      
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
  
protected:
  // The agent owning this mailer.
  Agent* p_owner;
  
  // The current mailer status.
  status_t p_status;

  // The mailer signature, which must correspond to the type of the message
  // handled by this mailer.
  std::string p_signature;
};


#endif // ULYSSES_COMMUNICATION__MESSAGE_MAILER_H_
