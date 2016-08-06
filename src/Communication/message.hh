#ifndef ULYSSES_COMMUNICATION__MESSAGE_H_
#define ULYSSES_COMMUNICATION__MESSAGE_H_

#include <string>

#include "Kernel/globals.hh"
#include "Utilities/Statistics/message-statistics.hh"

class LocalStatistics;

// The Message abstract class. 
// Each Message should implement this class. 
class Message
{
public:
  typedef std::unique_ptr<Message> uptr;
  typedef std::shared_ptr<Message> sptr;

  Message()
  : p_source_id(Constants::nullid), 
    p_destination_id(Constants::nullid),
    p_stamp(0)
  { }
    
  virtual ~Message()
  { }
  
  // It clones the message object.
  virtual Message* clone() = 0;

  // It returns the message type.
  virtual std::string type() const = 0;

  // It resets the message content (without affeting the message header).
  virtual void reset() = 0;

  // It returns a message description.
  virtual std::string dump() const;

  // It returns the source Agent.
  oid_t source() const // DEPRECATED sobstiutited by senderID
    { return p_source_id; }
  
  // It returns the source Agent.
  oid_t senderID() const
    { return p_source_id; }

  // It returns the destination Agent.
  oid_t destination() const // DEPRECATED sobstiutited by receiverID
    { return p_destination_id; }

  oid_t receiverID() const 
    { return p_destination_id; }

  // It returns the stamp.
  size_t stamp() const
    { return p_stamp; }

  // It sets the Agent source.
  void setSource (oid_t a)
    { p_source_id = a; }

  // It sets the Agent source.
  void setSenderID(oid_t a)
    { p_source_id = a; }
  
  // It sets the Agent destination.
  void setDestination (oid_t a)
    { p_destination_id = a; }

  void setReceiverID (oid_t a)
    { p_destination_id = a; }

  // Sets the stamp to the one given as a parameter.
  void setStamp(size_t stamp)
    { p_stamp = stamp; }

  // Updates the message statistics, given the local (agent) statistics.
  void updateStatistics(LocalStatistics& localstats);

  // It returns the message statistics.
  MessageStatistics& statistics()
    { return p_stats; }

protected:
  // Note: this is a protected copy constructor - it can only called within the
  // object, or its derivates. It copies parameters carried by this class, and
  // used by the clone functions.
  Message(const Message& other)
  {
    p_source_id  = other.p_source_id;
    p_destination_id = other.p_destination_id;
    p_stamp = other.p_stamp;
    p_stats = other.p_stats;
  }

private:
  // Agent source
  oid_t p_source_id;
 
  // Agent destination
  oid_t p_destination_id;
  
  // The counters carried within the current message
  MessageStatistics p_stats;

  // Ordered time stamp. It is assigned by the source agent and it 
  // represents the n-th message sent by that agent. 
  size_t p_stamp;
};

#endif // ULYSSES_COMMUNICATION__MESSAGE_H_
