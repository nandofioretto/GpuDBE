#ifndef ULYSSES_COMMUNICATION__MAILBOX_H_
#define ULYSSES_COMMUNICATION__MAILBOX_H_

#include <deque>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

#include "Kernel/types.hh"

class Message;
class MessageHandler;

// The mailbox associated to each agent.
// It has different queues of incoming and outgoing messages distinguished by 
// the message type.
class Mailbox
{
public:
  typedef std::string msg_t;

  Mailbox() { }

  ~Mailbox() { }
    
  // It registers a new mail associated with type specified as a parameter.
  void attachMail(std::string type);

  // It deregisters the mail type given as a parameter. It enforces the removal 
  // of the message handler associated to the mail type (calling its destructor).
  void detachMail(msg_t type);
  
  // It resturns the 'older' message present in the inbox, removing such message
  // from the inbox.
  std::shared_ptr<Message> readNext();

  // It resturns the 'older' message of type given as a parameter present in the
  // inbox, removing it from the inbox. 
  std::shared_ptr<Message> readNext(msg_t type);

  // It returns all the messages of type given as a parameter, present in the 
  // inbox. It removes such messages from the inbox.
  std::vector<std::shared_ptr<Message> > readAll(msg_t type);
  
  // It saves the new message into the inbox.
  void receive(std::shared_ptr<Message> msg);

  // It sends the message given as a parameter, by recoding it into the 
  // incoming message queue of the destination agent.
  void send(std::shared_ptr<Message> msg);

  // It returns the number of messages of type given as a parameter, which 
  // have been received and not yet read.
  size_t size(std::string msg_type)
  {
    if( map_inbox_size_.find( msg_type ) == map_inbox_size_.end() )
      return 0;
    else
      return std::max(0, map_inbox_size_[ msg_type ]);
  }

  size_t size() const
  {
    return inbox_.size();
  }

  // It checks whether the mailbox is empty. 
  bool empty()
  { 
    return inbox_.size() == 0; 
  }

  // It checks whether the mailbox contains no messages of type given as a
  // parameter.
  bool isEmpty(msg_t msg_type)
  {
    return (map_inbox_size_[ msg_type ] == 0);
  }

  // Clear the mailbox by removing all its elements
  void clear()
  {
    while (!inbox_.empty() ) 
      inbox_.pop_front();
  }


protected:
  // The queue of messages which where received by the current agent
  // but not yet processed.
  std::deque< std::shared_ptr<Message> > inbox_;

  // Count the number of messages of a certain type that are present
  // in the mailbox at a given moment.
  std::unordered_map<msg_t, int> map_inbox_size_;

};


#endif // ULYSSES_COMMUNICATION__MAILBOX_H_
