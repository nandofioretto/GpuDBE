#include "Kernel/globals.hh"
#include "Utilities/utils.hh"
#include "Kernel/agent.hh"
#include "Communication/mailbox.hh"
#include "Communication/scheduler.hh"
#include "Algorithms/DPOP/value-propagation.hh"
#include "Algorithms/DPOP/value-msg-handler.hh"
#include "GPU/cuda_dpop_state.hh"

#include <vector>
#include <memory>
#include <algorithm>

using namespace std;

ValuePropagation::ValuePropagation(Agent& owner, std::shared_ptr<CUDA::DPOPstate> state)
  : Algorithm(owner), p_terminated(false)
{
  p_msg_handler = ValueMsgHandler::sptr(new ValueMsgHandler(owner));
  p_state = state;
}


ValuePropagation::~ValuePropagation()
{ }


void ValuePropagation::initialize()
{
  Mailbox& MB = owner().openMailbox();
  attachMailSystem("VALUE", p_msg_handler);
}

void ValuePropagation::finalize()
{
  // detachMailSystem("VALUE");
  // Reschedule the agent running this algorithm to let the 
  // calling routine to continue.
  Scheduler::FIFOinsert(owner().id());
}


bool ValuePropagation::canRun()
{
  // We cannot initialize the handler at the begin because the pseudo-tree
  // construction may not have terminated.
  if (not p_msg_handler->initialized())
    p_msg_handler->initialize(/*p_state*/);

  return (! terminated() and ( (owner().ptNode().isRoot()) or 
			       (!owner().ptNode().isRoot() and recvAllMessages())));
}


bool ValuePropagation::recvAllMessages()
{
  if (!owner().openMailbox().isEmpty("VALUE")){
    p_msg_handler->processIncoming(p_state);
  }
  
  // Transparent rescheduling for sequential Hack:
  // Reschedule the agent running this algorithm to let the 
  // calling routine to continue.
  if(!p_msg_handler->recvAllMessages())
    Scheduler::FIFOinsert(owner().id());
  return (p_msg_handler->recvAllMessages());
}


void ValuePropagation::run()
{
  ValueMsgHandler &handler = *p_msg_handler;

  // If DCOP instance has no solution cannot retrieve values.
  // if(!Constants::isFinite(g_dcop->cost())) {
  //   p_terminated = true;
  //   handler.send();
  //   return;
  // }
  
  // std::cout << "value propagation for agent " << owner().name() << std::endl;
  // If Root copies the values from global memory to dpop_state
  p_state->compute_best_value(owner());

  handler.prepareOutgoing();
  handler.send();
   
  g_dcop->set_var_solution( p_state->get_var_id(), p_state->get_xi_best_value());
  
  p_terminated = true;
}
