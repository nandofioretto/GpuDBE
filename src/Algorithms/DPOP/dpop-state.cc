#include "Algorithms/DPOP/dpop-state.hh"
#include "Algorithms/Orderings/pseudo-tree-ordering.hh"
#include "Utilities/utils.hh"
#include "Utilities/constraint-utils.hh"

#include "Algorithms/DPOP/util-msg-handler.hh"
#include "Kernel/globals.hh"
#include "Kernel/agent.hh"
#include "Kernel/table-constraint.hh"

#include <vector>
#include <cmath>

using namespace std;
using namespace Utils;

void DPOPstate::initialize(Agent& agent, UtilMsgHandler::sptr handler) {
	// Set the message handler ptr
	p_util_msg_handler = handler;

	ai = &agent;
	xi = &agent.boundaryVariableAt(0);

	//---------------------------------------------------------------------------//
	// Retrieve the separator set of this agent and create the UTIL table
	PseudoTreeOrdering& tn = agent.ptNode();
	p_util_table_rows = Codec::sptr(new Codec(tn.separator()));

	//---------------------------------------------------------------------------//
	// Allocate memory for the util_table and for the best values of the local var
	// associated to each row of the util table.
	nbRowsUT = p_util_table_rows->size() > 0 ? p_util_table_rows->size() : 1;
	hostUtilTable = new int[nbRowsUT];

	//---------------------------------------------------------------------------//
	// Solve the unary conatraints involving the local variable.
	std::vector<oid_t> c_unary = ConstraintUtils::involvingExclusively(
			agent.boundaryVariableIDs(), -1);
	solveUnaryConstraint(agent, c_unary);

	//---------------------------------------------------------------------------//
	// Select the constraints to be checked during the util computation
	// _PPP is the set of parents and pseudo-parents of this agent.
	if (!tn.root()) {
		std::vector<oid_t> ppp = tn.pseudoParents();
		ppp.push_back(tn.parent());
		ppp.push_back(tn.content());

		std::vector<oid_t> cons = ConstraintUtils::involvingExclusively(ppp,
				ai->id());
		Utils::exclude_emplace(c_unary, cons);
		for (oid_t cid : cons)
			p_constraints.push_back(
					static_cast<TableConstraint*>(&g_dcop->constraint(cid)));
	}

}

int DPOPstate::get_xi_best_value() {
	// Order sep values in the same order of variables in codec
	std::vector<int> values;
	for (oid_t vid : p_util_table_rows->variables()) {
		values.push_back(p_sep_values[vid]);
	}

	size_t row = p_util_table_rows->encode(values);
	//////////// On GPU we might recompute this one /////////
	return p_xi_assignment[row];
}

void DPOPstate::solveUnaryConstraint(Agent& agent, std::vector<oid_t> c_unary) {
	IntVariable& x = agent.boundaryVariableAt(0);
	p_unary.resize(x.size(), 0);

	// solve unary constraints
	for (oid_t cid : c_unary) {
		TableConstraint& c = static_cast<TableConstraint&>(g_dcop->constraint(cid));
		for (int d = x.min(); d <= x.max(); d++) {
			p_unary[d] += c.getCost(std::vector<int> { d });
		}
	}
}

void DPOPstate::updateState() {
	// All Util messages have been received at this point. Thus, it
	// initializes the message handler which creates some auxiliary data
	// structures.
	p_util_msg_handler->initialize( /*&p_util_table, p_util_table_rows*/);

	std::vector<oid_t> vsep = p_util_table_rows->variables();
	std::map<oid_t, int> vsep_idx;
	vsep_idx[xi->id()] = -1;
	for (int i = 0; i < vsep.size(); i++)
		vsep_idx[vsep[i]] = i;

	std::vector<int> vsep_val(vsep.size());

	std::vector<int> b_vsep_val(vsep.size() + 1); // xi union vsep

	std::vector<int> to_evaluate(2);

	///std::cout << "world: " << wid << std::endl;
	for (int r = 0; r < p_util_table_rows->size(); r++) // ROWS of UTIL table
	{
		cost_t util = Constants::worstvalue;
		int xi_assignment = -1;
		vsep_val = p_util_table_rows->decode(r);

		for (int i = 1; i < b_vsep_val.size(); i++)
			b_vsep_val[i] = vsep_val[i - 1];

		// Projection across domain elements
		for (int d = xi->min(); d <= xi->max(); d++) // d \in D_xi
		{
			b_vsep_val[0] = d;
			cost_t util_x = 0;
			//////////////// Evaluate Unary constraints ///////////////
			util_x = Utils::sum(util_x, p_unary[d]);
			//////////////// Evaluate constraints //////////////////
			for (TableConstraint* con : p_constraints) {
				// Popolate constraint values to check
				for (int k = 0; k < con->arity(); k++) {
					int idx = vsep_idx[con->variableIdAt(k)];
					to_evaluate[k] = idx == -1 ? d : vsep_val[idx];
				}

				util_x = Utils::sum(util_x, con->getCost(to_evaluate));
			}
			//////////////// Aggregate UTIL messages //////////////////
			util_x = Utils::sum(util_x, p_util_msg_handler->msgCosts(b_vsep_val));
			/////////////////////////////////////////////////////
			if (Utils::isBetter(util_x, util)) {
				util = util_x;
				xi_assignment = d;
			}
		}  // for all d \in D_i (projection)

		p_util_table[r] = util;
		p_xi_assignment[r] = xi_assignment;

		///std::cout << " best util: " << util << std::endl;

	} // for all combo of values in sep vars

}

void DPOPstate::updateRootState() {
	// All Util messages have been received at this point. Thus, it
	// initializes the message handler which creates some auxiliary data
	// structures.
	p_util_msg_handler->initialize( /*&p_util_table, p_util_table_rows*/);

	std::vector<int> b_vsep_val(1); // xi

	cost_t util = Constants::worstvalue;
	int xi_assignment = -1;

	for (int d = xi->min(); d <= xi->max(); d++) // d \in D_xi
	{
		b_vsep_val[0] = d;
		cost_t util_x = 0;
		//////////////// Evaluate Unary constraints ///////////////
		util_x = Utils::sum(util_x, p_unary[d]);

		//////////////// Aggregate UTIL messages //////////////////
		util_x = Utils::sum(util_x, p_util_msg_handler->msgCosts(b_vsep_val));
		if (Utils::isBetter(util_x, util)) {
			util = util_x;
			xi_assignment = d;
		}
	}

	p_util_table[0] = util;
	p_xi_assignment[0] = xi_assignment;

}
