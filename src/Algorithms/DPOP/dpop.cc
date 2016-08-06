#include <chrono>
#include <thread>
#include <memory>

#include "preferences.hh"
#include "Algorithms/DPOP/dpop.hh"
#include "Algorithms/DPOP/pseudo-tree-phase.hh"
#include "Algorithms/DPOP/util-propagation.hh"
#include "Algorithms/DPOP/value-propagation.hh"

#include "Kernel/globals.hh"
#include "Kernel/agent.hh"
#include "Utilities/statistics.hh"
#include "Utilities/utils.hh"
#include "Communication/scheduler.hh"

#include "GPU/cuda_allocator.hh"
#include "GPU/cuda_dpop_state.hh"

using namespace std;
using namespace CUDA;

DPOP::DPOP(Agent& owner)
: Algorithm(owner), p_terminated(false), p_on_device(false)
{
	p_state = shared_ptr<DPOPstate>(new DPOPstate());

	p_pt_construction_phase = PseudoTreeConstruction::uptr(new PseudoTreeConstruction(owner));

	p_util_propagation_phase = unique_ptr<UtilPropagation>(new UtilPropagation(owner, p_state));

	p_value_propagation_phase = unique_ptr<ValuePropagation>(new ValuePropagation(owner, p_state));
}

DPOP::~DPOP()
{ }


void DPOP::initialize()
{ 
	p_pt_construction_phase->initialize();
	p_pt_construction_phase->set_elected_root(g_dcop->get_elected_root());
	p_pt_construction_phase->set_heuristic(g_dcop->get_heuristic());
	p_util_propagation_phase->initialize();
	p_value_propagation_phase->initialize();

	owner().statistics().setStartTimer();
	owner().statistics().setSimulatedTime();
}


void DPOP::finalize()
{  
	if( !p_terminated ) {
		p_terminated = true;
		// std::cout << owner().ordering().dump() << std::endl;
	}
}


void DPOP::run()
{
	// -------------------------------------------------------------------------------------------//
	// Pseudo-Tree
	// -------------------------------------------------------------------------------------------//
	l_PseudoTreeConstructionPhase:
	if (p_pt_construction_phase->canRun()) {
		p_pt_construction_phase->run();
	}

	if (p_pt_construction_phase->terminated()) {
		if( !p_on_device ) {
			if(preferences::verboseDevInit) {
				std::cout << "[GPU] Allocating Agent " << owner().id() << " data on device..." << std::flush;
			}
			CUDA::Allocator::allocate_agent( owner() );
			p_state->initialize(owner());
			if(preferences::verboseDevInit) {
				std::cout << "done\n";
			}
			p_on_device = true;
		}
		goto l_UtilPhase;
	}
	Scheduler::FIFOinsert(owner().id());
	return;	  // pass the control to next agents in the scheduler


	// -------------------------------------------------------------------------------------------//
	// UTIL Propagation
	// -------------------------------------------------------------------------------------------//
	l_UtilPhase:
	if (p_util_propagation_phase->canRun()) {
		p_util_propagation_phase->run();
	}
	if (p_util_propagation_phase->terminated()) {
		goto l_ValuePhase;
	}
	Scheduler::FIFOinsert(owner().id());
	return;	  // pass the control to next agents in the scheduler


	// -------------------------------------------------------------------------------------------//
	// VALUE Propagation
	// -------------------------------------------------------------------------------------------//
	l_ValuePhase:
	if (p_value_propagation_phase->canRun()) {
		p_value_propagation_phase->run();
	}
	if (p_value_propagation_phase->terminated()) {
		finalize();
	}
	else
		Scheduler::FIFOinsert(owner().id());
	return;
}

void DPOP::stop()
{ }
