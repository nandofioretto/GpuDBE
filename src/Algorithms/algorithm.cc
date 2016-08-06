#include "Algorithms/algorithm.hh"
#include "Kernel/types.hh"
#include "Kernel/agent.hh"
#include "Communication/message-handler.hh"
#include "Communication/message-mailer.hh"
#include "Communication/mailbox.hh"

Algorithm::Algorithm(Agent& owner)
    : p_owner(&owner)
{ 
  // Statistics::registerTimer("initialization", owner.id());
}


void Algorithm::attachMailSystem(std::string type, MessageHandler::sptr handler,
  MessageMailer::sptr mailer)
{
  p_owner->openMailbox().detachMail(type);
  p_msg_handler[ type ] = MessageHandler::sptr(handler);
  p_msg_mailer[ type ] = MessageMailer::sptr(mailer);
}


void Algorithm::detachMailSystem(msg_t type)
{
  p_owner->openMailbox().detachMail(type);
  p_msg_handler[ type ].reset();
  p_msg_handler.erase( type );
  p_msg_mailer.erase( type );
}
