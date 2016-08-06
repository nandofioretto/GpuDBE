#include "Algorithms/DPOP/value-msg-handler.hh"
#include "Algorithms/DPOP/value-msg.hh"
#include "Communication/scheduler.hh"
#include "Kernel/globals.hh"
#include "Kernel/agent.hh"
#include "Utilities/statistics.hh"
#include "Utilities/utils.hh"
#include "GPU/cuda_dpop_state.hh"

using namespace std;

ValueMsgHandler::ValueMsgHandler(Agent& a) :
		MessageHandler(a), p_initialized(false) {
}

ValueMsgHandler::~ValueMsgHandler() {
}

void ValueMsgHandler::initialize() {
	for (oid_t cid : owner().ptNode().children()) {
		ValueMsg::uptr msg(new ValueMsg);
		msg->setSource(owner().id());
		msg->setDestination(cid);

		for (oid_t vid : Utils::concat(owner().ptNode().content(),
				owner().ptNode().separator()))
			msg->set_sep_var_id(vid);

		p_outgoing.push_back(std::move(msg));
	}
	// p_dpop_state = state;
	p_initialized = true;
}

void ValueMsgHandler::processIncoming(
		std::shared_ptr<CUDA::DPOPstate> p_dpop_state) {
	if (owner().openMailbox().isEmpty("VALUE"))
		return;

	p_received = dynamic_pointer_cast<ValueMsg>(
			owner().openMailbox().readNext("VALUE"));

	//---------------
	// NOT HANDLED ON GPU
	//---------------
	// Store values of the variables of the received message
	for (oid_t vid : p_received->get_sep_variables()) {
		if (!Utils::find(vid, owner().ptNode().separator()))
			continue;
		int val = p_received->get_value(vid);
		p_dpop_state->set_sep_value(val, vid);
	}

}

void ValueMsgHandler::prepareOutgoing(
		std::shared_ptr<CUDA::DPOPstate> p_dpop_state) {
	//---------------
	// HANDLED ON GPU
	//---------------

	oid_t xi = owner().ptNode().content();
	std::vector<oid_t> sep_vars = Utils::concat(xi,
			owner().ptNode().separator());

	// Take all variables in the separator set:
	for (int i = 0; i < p_outgoing.size(); i++) {
		for (oid_t vid : sep_vars) {
			int val =
					(vid == xi) ?
							p_dpop_state->get_xi_best_value() :
							p_dpop_state->get_sep_value(vid);

			p_outgoing[i]->set_value(val, vid);
		}
	}
}

void ValueMsgHandler::send(oid_t dest_id) {
	if (dest_id == Constants::nullid) {
		// sends all outgoing messages
		for (int i = 0; i < p_outgoing.size(); ++i) {
			ValueMsg::sptr to_send(p_outgoing[i]->clone());
			owner().openMailbox().send(to_send);
			Scheduler::FIFOinsert(to_send->destination());
		}
	}
}

bool ValueMsgHandler::recvAllMessages() {
	return (owner().ptNode().isRoot() or p_received);
}
