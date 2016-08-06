#include "GPU/cuda_dpop_state.hh"
#include "GPU/cuda_utils.hh"
#include "GPU/gpu_dpop_util_phase.hh"
#include "GPU/gpu_dpop_value_phase.hh"
#include "GPU/gpu_data_allocator.hh"

#include "preferences.hh"
#include "Algorithms/Orderings/pseudo-tree-ordering.hh"
#include "Algorithms/DPOP/util-msg.hh"
#include "Kernel/agent.hh"
#include "Kernel/codec.hh"
#include "Kernel/globals.hh"
#include "Kernel/table-constraint.hh"
#include "Utilities/utils.hh"
#include "Problem/dcop-instance.hh"
#include "Utilities/constraint-utils.hh"

#include <vector>
#include <memory>
#include <map>
#include <cuda_runtime.h>


using namespace CUDA;


DPOPstate::DPOPstate() :
  pHostUtilTable(NULL), p_sep_value_child(NULL), p_sep_values(NULL), pHostConstraintSelectedValues(NULL) {

	cudaCheck(cudaEventCreate(&startEvent));
	cudaCheck(cudaEventCreate(&stopEvent));
	time_ms = 0;
	alloc_time_ms = 0;
}


DPOPstate::~DPOPstate() {
	cudaCheck(cudaEventRecord(startEvent, 0));

	if (pHostUtilTable != NULL) {
	    Statistics::startTimer("gpu-alloc");
		if (preferences::usePinnedMemory) {
			cudaFreeHost(pHostUtilTable);
		} else {
			delete[] pHostUtilTable;
		}
	    Statistics::stopwatch("gpu-alloc");
	}
	cudaCheck(cudaEventRecord(stopEvent, 0));
	cudaCheck(cudaEventSynchronize(stopEvent));
	cudaCheck(cudaEventElapsedTime(&time_ms, startEvent, stopEvent));
	alloc_time_ms += time_ms;

	if (preferences::verboseDevInit) {
		printf("[CPU] agt %d host-free-time elapsed %f ms\n", p_ai_id, time_ms);
	}

	if (p_sep_value_child != NULL)
		delete[] p_sep_value_child;
	if (pSeparatorSetId.size() > 0)
		delete[] p_sep_values;
	if (pHostConstraintSelectedValues != NULL)
		delete[] pHostConstraintSelectedValues;


	cudaCheck(cudaEventDestroy(startEvent));
	cudaCheck(cudaEventDestroy(stopEvent));
}


void DPOPstate::initialize(Agent& a) {
	PseudoTreeOrdering& tn = a.ptNode();

	p_ai_id = a.id();
	p_xi_id = a.variableAt(0).id();
	p_dom_size = a.variableAt(0).size();
	p_is_root = tn.isRoot();
	p_is_leaf = tn.isLeaf();
	p_nb_binary = a.get_cuda_nb_binary();
	p_cinfo_size = a.get_cuda_children_info_size();
	p_max_cinfo_size = a.get_max_ch_set_size();
	p_has_unary = a.get_cuda_has_unary();
	p_nb_agents = g_dcop->nbAgents();

	pSeparatorSetId = tn.separator();
	Codec __codec(pSeparatorSetId);
	pUtilTableRowsAfterProj = __codec.size() > 0 ? __codec.size() : 1;
	pUtilTableRows = pUtilTableRowsAfterProj * p_dom_size;

	//---------------------------------------------------------------------------//
	// Allocate memory for the util_table and for the best values of the local var
	// associated to each row of the util table.
	if (pUtilTableRows * sizeof(int) > preferences::maxHostMemory) {  // 12 GB mem limit on host
		std::cout << "separator set size: " << pSeparatorSetId.size() <<  " Exceeding Host Memory Limit. Aborting...\n";
		exit(-1);
	}

	//-----------------------------------------------
	// Allocate Hoste UTIL Table (this should be done in a pre-processing step)!
	//-----------------------------------------------
    Statistics::startTimer("gpu-alloc");
	cudaCheck(cudaEventRecord(startEvent, 0));
	if (preferences::usePinnedMemory) {
		cudaMallocHost((void**)&pHostUtilTable, pUtilTableRows * sizeof(int));      // host pinned
	} else {
		pHostUtilTable = new int[pUtilTableRows];    // host paged
	}
	cudaCheck(cudaEventRecord(stopEvent, 0));
	cudaCheck(cudaEventSynchronize(stopEvent));
	cudaCheck(cudaEventElapsedTime(&time_ms, startEvent, stopEvent));
    Statistics::stopwatch("gpu-alloc");

	alloc_time_ms += time_ms;
	if (preferences::verboseDevInit) {
		printf("[CPU] agt %d host-allocation-time elapsed %f ms\n", p_ai_id, time_ms);
	}

	if (preferences::singleAgent || preferences::usePinnedMemory) {
		a.statistics().setStartTimer();
		a.statistics().setSimulatedTime();
	}

	//---------------------------------------------------------------------------//
	// Host part
	//---------------------------------------------------------------------------//
	pXiAssignment = -1;
	pXiSubtreeUtil = Constants::unsat;

	//---------------------------------------------------------------------------//
	// Retrieve the separator set of this agent and create the UTIL table
	//---------------------------------------------------------------------------//
	for (oid_t vid : pSeparatorSetId) {
		pSepValues[vid] = Constants::NaN;
	}

	//---------------------------------------------------------------------------//
	// Select the constraints to be checked during the util computation
	// _PPP is the set of parents and pseudo-parents of this agent.
	if (!tn.root()) {
		std::vector<oid_t> ppp = tn.pseudoParents();
		ppp.push_back(tn.parent());
		ppp.push_back(tn.content());

		std::vector<oid_t> cons = ConstraintUtils::involvingExclusively(ppp,
				p_ai_id);
		// Utils::exclude_emplace(c_unary, cons);
		for (oid_t cid : cons)
			pConstraints.push_back(
					static_cast<TableConstraint*>(&g_dcop->constraint(cid)));

		pHostConstraintSelectedValues = new int[cons.size()];
	}

	// ------------------------------------------------ //
	// This is for integrating children tables on CPU
	// NOT EFFICIENT
	// ------------------------------------------------ //
	pChildIdToUtilTablePtr.resize(g_dcop->nbAgents(), NULL);
	pChildIdToUtilTableRows.resize(g_dcop->nbAgents(), 0);

	// c_info
	if (p_cinfo_size > 0)
		p_sep_value_child = new int[p_cinfo_size]; // TODO: this should be new [ max_child_sep_size ]

	p_sep_values = new int[pSeparatorSetId.size()];
	for (int i = 0; i < pSeparatorSetId.size(); i++)
		p_sep_values[i] = 0;

	if (pSeparatorSetId.size() > 0) {
		p_sep_values[pSeparatorSetId.size() - 1] = -1;
		_curr_idx = pSeparatorSetId.size() - 1;
	}

	// Retrieve the set of children of ai
	int max_ch_sep_size = 0;
	for (int cid : tn.children()) {
		pChildrenId.push_back(cid);

		p_children_info.push_back(cid); // agent id
		std::vector<int> cIdSep = g_dcop->agent(cid).ptNode().separator();

		if (cIdSep.size() > max_ch_sep_size)
			max_ch_sep_size = cIdSep.size();

		p_children_info.push_back(cIdSep.size()); // sep size

		for (int vid : cIdSep) {
			// relative idx pos wrt separator array of this agent
			p_children_info.push_back(Utils::findIdx(pSeparatorSetId, vid));
		}
	}

	//std::cout << "Children of Agent " << p_ai_id << ": " << Utils::dump(pChildrenId) << "\n";
}


void DPOPstate::setChildUtilTableInfo(int chId, int* tablePtr,
		size_t tableRows) {
	pChildIdToUtilTablePtr[chId] = tablePtr;
	pChildIdToUtilTableRows[chId] = tableRows;
}


int DPOPstate::utilTableCudaVersion() {
	// Leaf = version 0
	if (p_is_leaf)
		return 0;

	// If can store all children in Global Memory + >= 1/10 of its util table = version 1
	size_t chRequiredMemory = 0;
	for (int cId : pChildrenId) {
		chRequiredMemory += getChildTableRows(cId) * sizeof(int);
	}
	size_t thisRequiredMemory = pUtilTableRowsAfterProj * sizeof(int);
	size_t cudaFreeMem = CUDAutils::get_nb_bytes_free_global_memory();
	if (chRequiredMemory > cudaFreeMem)
		return 2;
	if (cudaFreeMem - chRequiredMemory >= (0.1 * thisRequiredMemory))
		return 1;
	return 2;
}


void DPOPstate::compute_util_table(Agent& a) {
	int version = utilTableCudaVersion();

	GPU_DPOPutilPhase phase;
	phase.compute_util_table(*this, version);
	if (preferences::singleAgent) {
		// Remove transfers time
		a.statistics().addSimulatedTime(-phase.getDataTransferTimeMs()*1000);
		Statistics::addTimer("gpu-alloc", phase.getDataTransferTimeMs());
	}

	if (version == 2 || version == 3) {
		aggregate_children_and_project();
	}

	// Uncomment this lines if you want to account for exclusively GPU computing time
	// (i.e., no data transfers - whose allocations can be done in a pre-processing step)
	// a.statistics().setSimulatedTime( phase.getKernelTimeMs() );
	// a.statistics().setStartTimer(); // disregard previously set start timer
}

void DPOPstate::compute_best_value(Agent& a) {
	size_t gpu_time = 0;

	if (a.ptNode().isRoot()) {
		GPU_DPOPvaluePhase::get_best_values(*this);
		g_dcop->set_util(get_xi_best_util());
	} else {
		get_xi_best_value();
		saveConstraintValue();

		// Now Constraints selected values are also initialized -> copy them on gpu
		GPU_DPOPvaluePhase::copy_constraint_sel_values(*this, gpu_time);
	}

	// If enabled counts only GPU process time
	// a.statistics().setStartTimer();
	// a.statistics().setSimulatedTime( gpu_time );
}



void l_get_dPow(int* dPow, int dPow_size, int d) {
  if (dPow_size == 0)
    return;
  dPow[dPow_size - 1] = 1;
  
  for (int i = dPow_size - 2; i >= 0; i--) {
    dPow[i] = dPow[i + 1] * d;
  }
}


/**
 * The projection can definatevly be handled on GPU
 */
void DPOPstate::aggregate_children_and_project() {
	std::cout << "[CPU time] UTIL_ " << p_ai_id
		  << " Large Table processing: " << pUtilTableRowsAfterProj << " rows: " << std::flush; 
	//Handles aggregation of children and UTIL projection on CPU...\n";
	reset_decode();
	
	Statistics::registerTimer("cpu-compute", p_ai_id);
	Statistics::startTimer("cpu-compute", p_ai_id);

	util_t util_di = 0;
	util_t util_wi = 0;
	util_t util_tmp = -1;

	int _i = 0, _j = 0, _di = 0, _ch_id = 0, _x1 = 0, _ch_row = 0;
	int _ch_sep_size = 0;  //, _c_code;
	int nb_var_sep = pSeparatorSetId.size();
	int d_size = p_dom_size;

	size_t row_bp = 0;
	for (size_t _row = 0; _row < pUtilTableRowsAfterProj; _row++) {
	  if (!p_is_root) // ok (not a problem in terms of time)
	    fast_decode_next(p_sep_values, nb_var_sep, d_size);

	  // ---------------------------------------------------------------------- //
	  // Compute Util Table Entry Value
	  util_wi = Constants::unsat;
	  for (_di = 0; _di < d_size; _di++) // Reticulate across al domain elements
	    {
	      util_di = 0;
	      row_bp = _row * d_size + _di;
	      util_di = pHostUtilTable[row_bp];
	      
	      if (util_di == Constants::unsat)
		continue;
	      
	      //--------------------------------------------------------------------- //
	      // Messages from Children
	      //--------------------------------------------------------------------- //
	      _i = 0;
	      while (_i < p_children_info.size()) {
		_ch_id = p_children_info[_i];
		_i++;
		_ch_sep_size = p_children_info[_i];
		_i++;
		
		if (_ch_sep_size <= 0)
		  continue;

		for (_j = 0; _j < _ch_sep_size; _j++) {
		  _x1 = p_children_info[_i];
		  _i++; // index of sep_values (-1 if current agent)
		  p_sep_value_child[_j] =
		    (_x1 == -1) ? _di : p_sep_values[_x1];
		}

		_ch_row = fast_encode(p_sep_value_child, _ch_sep_size, d_size);
		util_tmp = pChildIdToUtilTablePtr[_ch_id][_ch_row];

		if (util_tmp == Constants::unsat)
		  break;		
		util_di += util_tmp;
	      }
	      if (util_wi == Constants::unsat || util_di > util_wi) {
		util_wi = util_di;
	      }
	    }
	  
	  // Update Util Table
	  pHostUtilTable[_row] = util_wi;
	}
	
	Statistics::stopwatch("cpu-compute", p_ai_id);
	std::cout << Statistics::getTimer("cpu-compute", p_ai_id) << " ms \n";;
}

int DPOPstate::get_xi_best_value() {
	if (pXiAssignment != -1)
		return pXiAssignment;

	util_t util_di = 0;
	util_t best_util = 0;
	util_t util_tmp = 0;
	int best_val = 0;

	int _i = 0, _j = 0, _di = 0, _ch_id = 0, _idx = 0, _ch_row = 0;
	int _ch_sep_size = 0;

	int nb_var_sep = pSeparatorSetId.size();
	int d_size = p_dom_size;

	std::vector<int> to_evaluate(2);

	for (int i = 0; i < pSeparatorSetId.size(); i++) {
		p_sep_values[i] = pSepValues[pSeparatorSetId[i]];
	}

	std::map<oid_t, int> vsep_idx;
	vsep_idx[p_xi_id] = -1;
	for (int i = 0; i < pSeparatorSetId.size(); i++)
		vsep_idx[pSeparatorSetId[i]] = i;

	// Order sep values in the same order of variables in codec
	best_util = Constants::unsat;

	for (_di = 0; _di < p_dom_size; _di++) {
		util_di = 0;
		//--------------------------------------------------------------------- //
		// Binary constraints
		//--------------------------------------------------------------------- //
		for (TableConstraint* con : pConstraints) {
			// Popolate constraint values to check
			for (_j = 0; _j < con->arity(); _j++) {
				_idx = vsep_idx[con->variableIdAt(_j)];
				to_evaluate[_j] = _idx == -1 ? _di : p_sep_values[_idx];
			}
			util_tmp = con->getCost(to_evaluate);

			if (util_tmp == Constants::unsat) {
				util_di = Constants::unsat;
				break;
			}

			util_di += util_tmp;
		}

		if (util_di == Constants::unsat)
			continue;

		//--------------------------------------------------------------------- //
		// Messages from Children
		//--------------------------------------------------------------------- //
		_i = 0;
		while (_i < p_children_info.size()) {
			_ch_id = p_children_info[_i];
			_i++;
			_ch_sep_size = p_children_info[_i];
			_i++;

			if (_ch_sep_size <= 0)
				continue;

			for (_j = 0; _j < _ch_sep_size; _j++) {
				_idx = p_children_info[_i];
				_i++; // index of sep_values (-1 if current agent)
				p_sep_value_child[_j] = (_idx == -1) ? _di : p_sep_values[_idx];
			}

			_ch_row = fast_encode(p_sep_value_child, _ch_sep_size, d_size);
			util_tmp = pChildIdToUtilTablePtr[_ch_id][_ch_row];

			if (util_tmp == Constants::unsat) {
				util_di = Constants::unsat;
				break;
			}

			util_di += util_tmp;
			// std::cout << " Aggr Util = " << util_di << "\n";
		}

		if (util_di == Constants::unsat)
			continue;

		if (best_util == Constants::unsat || util_di > best_util) {
			best_util = util_di;
			best_val = _di;
		}
	}

	set_xi_value(best_val);

	return best_val;
}

void DPOPstate::saveConstraintValue() {

	// assert(pXiAssignment[ wid ] != -1);

	std::vector<int> to_evaluate(2);
	for (int i = 0; i < pSeparatorSetId.size(); i++) {
		p_sep_values[i] = pSepValues[pSeparatorSetId[i]];
	}

	std::map<oid_t, int> vsep_idx;
	vsep_idx[p_xi_id] = -1;
	for (int i = 0; i < pSeparatorSetId.size(); i++) {
		vsep_idx[pSeparatorSetId[i]] = i;
	}
	int idx = 0;
	for (int i = 0; i < pConstraints.size(); i++) {
		TableConstraint* con = pConstraints[i];
		// Popolate constraint values to check
		for (int j = 0; j < con->arity(); j++) {
			idx = vsep_idx[con->variableIdAt(j)];
			to_evaluate[j] = idx == -1 ? pXiAssignment : p_sep_values[idx];
		}
		// std::cout  << "Agent " << p_ai_id << ": Saving Costr_" << con->id() << " cost: "
		// 		 << con->getCost100(to_evaluate, wid) << "\n";
		pHostConstraintSelectedValues[i] = con->getCost(to_evaluate);
	}
}

////////// MISC /////////////
void DPOPstate::print_host_util_table() {
	printf("Host Util Table agent_%d [%d]\n", p_ai_id, pUtilTableRowsAfterProj);
	size_t limit = pUtilTableRowsAfterProj < 50 ? pUtilTableRowsAfterProj : 10;

	for (size_t r = 0; r < limit; r++) {
		printf("%zu [", r);
		printf("%d ", pHostUtilTable[r]);
		printf("] \n");
	}
}

void DPOPstate::print_host_util_table_unproj() {
	printf("Host Util Table agent_%d [%d]\n", p_ai_id, pUtilTableRows);
	size_t limit = pUtilTableRows < 50 ? pUtilTableRows : 10;

	for (size_t r = 0; r < limit; r++) {
		printf("%zu [", r);
		printf("%d ", pHostUtilTable[r]);
		printf("] \n");
	}
}
