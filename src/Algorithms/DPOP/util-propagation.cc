#include <algorithm>
#include <vector>
#include <utility>

#include "Utilities/utils.hh"
#include "Kernel/globals.hh"
#include "Kernel/agent.hh"
#include "Algorithms/DPOP/util-propagation.hh"
#include "Algorithms/DPOP/util-msg-handler.hh"
//#include "Algorithms/DPOP/dpop-state.hh"
#include "GPU/cuda_dpop_state.hh"

#include "Communication/mailbox.hh"
#include "Communication/scheduler.hh"

#include "Problem/dcop-instance.hh"
#include "preferences.hh"

using namespace std;

UtilPropagation::UtilPropagation(Agent& owner,
		std::shared_ptr<CUDA::DPOPstate> state) :
		Algorithm(owner), p_terminated(false), p_state(state), p_msg_handler(
				UtilMsgHandler::sptr(new UtilMsgHandler(owner)))

{
	p_msg_handler = UtilMsgHandler::sptr(new UtilMsgHandler(owner));
	p_state = state;
}

UtilPropagation::~UtilPropagation() {
}

void UtilPropagation::initialize() {
	Mailbox& MB = owner().openMailbox();
	attachMailSystem("UTIL", p_msg_handler);
}

void UtilPropagation::finalize() {
	Scheduler::FIFOinsert(owner().id());
}

bool UtilPropagation::canRun() {
	return (!terminated() and (owner().ptNode().isLeaf() or recvAllMessages()));
}

bool UtilPropagation::recvAllMessages() {
	UtilMsgHandler &handler = *p_msg_handler;

	if (!owner().openMailbox().isEmpty("UTIL")) {
		handler.processIncoming();
	}

	// Transparent rescheduling for sequential Hack:
	// Reschedule the agent running this algorithm to let the
	// calling routine to continue.
	if (!handler.recvAllMessages())
		Scheduler::FIFOinsert(owner().id());

	return (handler.recvAllMessages());
}

void UtilPropagation::run() {
	if (preferences::verbose) {
		std::cout << "UTIL propagation running for agent " << owner().name() << std::endl;
	}
	for (auto msg : p_msg_handler->received()) {
		// Todo: Decode messages here.
		p_state->setChildUtilTableInfo(msg->source(), msg->getUtilTablePtr(),
				msg->getUtilTableRows());
	}

	p_state->compute_util_table(owner());

	if (!owner().ptNode().isRoot()) {
		p_msg_handler->prepareOutgoing();
		p_msg_handler->setMsgContent(*p_state);
		p_msg_handler->send(owner().ptNode().parent());
	}

	p_terminated = true;
}
