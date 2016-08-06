#include "Communication/mailbox.hh"
#include "Communication/message.hh"
#include "Communication/message-handler.hh"
#include "Kernel/agent.hh"
#include "Kernel/agent-factory.hh"

#include "Problem/dcop-instance.hh"

void Mailbox::attachMail(std::string type)
{
  map_inbox_size_[ type ] = 0;
}


void Mailbox::detachMail(std::string type)
{
  map_inbox_size_.erase( type );
}


Message::sptr Mailbox::readNext()
{
  ASSERT( !inbox_.empty(), "Trying to access to an empty inbox");
  
  Message::sptr result = std::move(inbox_.front());
  map_inbox_size_[ result->type() ]--;
  inbox_.pop_front();
  return result;
}

// Inefficient! In future use the readNext() and process each message 
// accordingly.
Message::sptr Mailbox::readNext(std::string type)
{
  ASSERT( map_inbox_size_[ type ] > 0, "Trying to access to an empty inbox");

  std::deque<Message::sptr>::iterator msg = inbox_.end();
  
  while( (*(--msg))->type().compare( type ) != 0 ) 
    ;

  Message::sptr result = std::move( (*msg) );//->clone();
  inbox_.erase( msg );
  map_inbox_size_[ result->type() ]--;
  return result;
}


std::vector<Message::sptr> Mailbox::readAll(std::string type)
{
  if (map_inbox_size_.find(type) == map_inbox_size_.end())
    return std::vector<Message::sptr>();
  
  if( map_inbox_size_[ type ] == 0 )
    return std::vector<Message::sptr>();

  std::deque<Message::sptr>::iterator msg = inbox_.begin();
  std::vector<Message::sptr> res;
  std::vector<std::deque<Message::sptr>::iterator> to_delete;
 
  for (; msg != inbox_.end(); ++msg) 
  {
    if( (*msg)->type().compare( type ) == 0 )
    {
      res.push_back( std::move(*msg) );
      to_delete.push_back( msg );
    }
  }

  for(int i=0; i<to_delete.size(); i++)
    inbox_.erase( to_delete[ i ] );

  map_inbox_size_[ type ] = 0;
  return res;
}


void Mailbox::receive(Message::sptr msg)
{
  inbox_.push_back( msg );
  map_inbox_size_[ msg->type() ]++;
  
  Agent& owner = g_dcop->agent(msg->destination());
  // Updates agent owner statistics
  owner.updateStatistics( msg->statistics() );

  // std::cout << owner.name() << " recvs: " << msg->dump() << std::endl;
  // getchar();
}
 

void Mailbox::send(Message::sptr msg)
{
  // Updates message statistics:
  Agent& owner = g_dcop->agent(msg->source());  
  msg->updateStatistics(owner.statistics());
  
  // std::cout << "Sending: " << msg->dump();
  // getchar();

  Agent& a = g_dcop->agent(msg->destination());
  a.openMailbox().receive( msg );
}
