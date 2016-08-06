#include <vector>
#include <cmath>

#include "GPU/cuda_allocator.hh"
#include "GPU/gpu_data_allocator.hh"
#include "GPU/cuda_utils.hh"
#include "Algorithms/Orderings/pseudo-tree-ordering.hh"
#include "Utilities/utils.hh"
#include "Utilities/constraint-utils.hh"
#include "Kernel/globals.hh"
#include "Kernel/agent.hh"
#include "Kernel/codec.hh"
#include "Kernel/table-constraint.hh"
#include "Problem/dcop-instance.hh"

#include "Kernel/types.hh"

using namespace CUDA;
using namespace std;

void Allocator::allocate_data()
{
	int nb_agents = g_dcop->nbAgents();
	int nb_variables = g_dcop->nbVariables();
	int nb_constraints = g_dcop->nbConstraints();
	int dom_size = g_dcop->variable(0).size();

	//---------------------------------------------------------------------------//

	GPU_allocator::allocate_data(nb_agents, nb_variables, nb_constraints, dom_size);

	for (Constraint* c : g_dcop->constraints()) {
		TableConstraint& pc = static_cast<TableConstraint&>(*c);
		if (pc.arity() > 1);
			allocate_constraint(pc);
	}
}


void Allocator::allocate_agent(Agent& ai)
{
	int cuda_ai_id = ai.id();
	int cuda_xi_id = ai.variableAt(0).id();
	int cuda_dom_min = ai.variableAt(0).min();
	int cuda_dom_max = ai.variableAt(0).max();
	size_t cuda_util_table_rows = 0;
	int cuda_sep_size = 0;

	//---------------------------------------------------------------------------//
	// Retrieve the separator set of this agent and create the UTIL table
	// TODO: In case, table pre-processing goes here
	//
	std::vector<int> cuda_sep = ai.ptNode().separator();
	Codec __codec(cuda_sep);
	cuda_util_table_rows = __codec.size() > 0 ? __codec.size() : 1;
	cuda_sep_size = cuda_sep.size();

	//---------------------------------------------------------------------------//
	// Retrieve the set of children of ai
	std::vector<int> cuda_children;
	int max_ch_sep_size = 0;
	for (int cid : ai.ptNode().children()) {
		cuda_children.push_back(cid); // agent id
		std::vector<int> sep = g_dcop->agent(cid).ptNode().separator();
		if (sep.size() > max_ch_sep_size)
			max_ch_sep_size = sep.size();
		cuda_children.push_back(sep.size()); // sep size
		for (int vid : sep) {
			// relative idx pos wrt separator array of this agent
			cuda_children.push_back(Utils::findIdx(cuda_sep, vid));
		}
	}

    //---------------------------------------------------------------------------//
    // Solve Unary constraints
	IntVariable& x = ai.boundaryVariableAt(0);
	std::vector<int> __c_unary;
	int dom_size = x.size();
	std::vector<util_t> cuda_unary(dom_size, 0);

	for (Constraint* c : x.constraints()) {
		if (c->arity() == 1) {
			__c_unary.push_back(c->id());
			TableConstraint* pc = static_cast<TableConstraint*>(c);
			for (int d = x.min(); d <= x.max(); d++) {
				cuda_unary[d] += pc->getCost(std::vector<int> { d });
			}
		}
	}

	//---------------------------------------------------------------------------//
	// Select the constraints to be checked during the util computation
	std::vector<int> cuda_binary_constraints_id;
	if (!ai.ptNode().root()) {
		std::vector<int> __ppp = ai.ptNode().pseudoParents();
		__ppp.push_back(ai.ptNode().parent());
		__ppp.push_back(ai.ptNode().content());

		cuda_binary_constraints_id = ConstraintUtils::involvingExclusively(__ppp, ai.id());
		Utils::exclude_emplace(__c_unary, cuda_binary_constraints_id);
	}

	std::vector<int> cuda_binary_constraints(cuda_binary_constraints_id.size() * 3);
	for (int i = 0; i < cuda_binary_constraints_id.size(); i++) {
		Constraint& c = g_dcop->constraint(cuda_binary_constraints_id[i]);
		cuda_binary_constraints[3 * i] = cuda_binary_constraints_id[i];
		cuda_binary_constraints[3 * i + 1] = Utils::findIdx(cuda_sep, c.scopeIds()[0]);
		cuda_binary_constraints[3 * i + 2] = Utils::findIdx(cuda_sep, c.scopeIds()[1]);
	}

	ai.set_cuda_data(cuda_binary_constraints_id.size(), cuda_children.size(),
					 max_ch_sep_size, __c_unary.size());

	// todo: Transfer the following function here (inline)
	GPU_allocator::init_agent(cuda_ai_id, cuda_xi_id, cuda_dom_min,
			cuda_dom_max, cuda_unary, cuda_binary_constraints,
			cuda_util_table_rows, cuda_sep, cuda_children, max_ch_sep_size);

	//CUDAutils::dump_agent_info(cuda_ai_id);
}


void Allocator::allocate_constraint(TableConstraint& c)
{
	int cuda_id = c.id();
	int cuda_arity = c.arity();
	std::vector<oid_t> cuda_scope = c.scopeIds();
	Codec __codec(c.scope());

	std::vector<util_t> cuda_utils(__codec.size());
	for (int i = 0; i < __codec.size(); i++) {
		cuda_utils[i] = c.getCost(__codec.decode(i));
	}

	// todo: Transfer the following function here (inline)
	GPU_allocator::init_constraint(cuda_id, cuda_scope, cuda_utils);

	// CUDAutils::dump_constraint(cuda_id);
}
