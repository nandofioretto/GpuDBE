#ifndef ULYSSES_ALGORITHMS__ALGORITHM_H_
#define ULYSSES_ALGORITHMS__ALGORITHM_H_

#include <memory>
#include <unordered_map>

#include "Kernel/globals.hh"
#include "Kernel/types.hh"

class MessageHandler;
class MessageMailer;
class Agent;

// The abstract class for any algorithm implementation
class Algorithm
{
public:
  typedef std::unique_ptr<Algorithm> uptr;
  typedef std::shared_ptr<Algorithm> sptr;
  enum status_t{k_none, k_init, k_running, k_terminated, k_stopped};
  
  Algorithm(Agent& owner);
  
  virtual ~Algorithm() { }
  
  // It initializes the algorithm.
  virtual void initialize() = 0;

  // It finalizes the algorithm.
  virtual void finalize() = 0;

  // It returns true if the algorithm can be executed in this agent.
  virtual bool canRun() = 0;

  // It executes the algorithm.
  virtual void run() = 0;

  // It stops the algorithm saving the current results  and states if provided
  // by the algorithm itself.
  virtual void stop() = 0;

  // It returns whether the algorithm has terminated.
  virtual bool terminated() = 0;

  // It waits untill the condition is satisified 
  // void wait(bool condition);

  // It attaches the message handler and the Message mailer given as parameter 
  // as well as register the mail it handles, in the owner's mailbox.
  void attachMailSystem(std::string type, std::shared_ptr<MessageHandler> handler,
                        std::shared_ptr<MessageMailer> mailer=nullptr);

  // It detaches the message handler given as parameter as well as deregister
  // the mail it handles, in the owner's mailbox.
  void detachMailSystem(msg_t type);

  // It returns the message handler associated to the message type given as a 
  // parameter.
  MessageHandler& handler(msg_t msg_type)
  {
    ASSERT( p_msg_handler.find( msg_type ) != p_msg_handler.end(),
	    "No Message Handler associated to message type: " << msg_type << " was found!"); 
    return *p_msg_handler[ msg_type ];
  }
  
  // It returns the mailer associated to the message type given as a parameter.
  MessageMailer& mailer(msg_t msg_type)
  {
    ASSERT( p_msg_mailer.find( msg_type ) != p_msg_mailer.end(),
	    "No mailer associated to message type: " << msg_type << " was found!"); 
    return *p_msg_mailer[ msg_type ];
  }
    
  Agent& owner() const
  {
    return *p_owner;
  }

protected:
  // The agent executing this algorithm.
  Agent* p_owner;
  
  // The current algorithm status.
  status_t p_status;
    
  // The set of message handlers held by this algorithm
  std::unordered_map<msg_t, std::shared_ptr<MessageHandler> > p_msg_handler;
  
  // The set of mailers held by this algorithm
  std::unordered_map<msg_t, std::shared_ptr<MessageMailer> > p_msg_mailer;
  
}; 

#endif // ULYSSES_ALGORITHMS__ALGORITHM_H_
