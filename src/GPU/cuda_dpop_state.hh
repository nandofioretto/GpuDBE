#ifndef CUDA_DPOP_STATE_H
#define CUDA_DPOP_STATE_H

#include <iostream>
#include <vector>
#include <memory>
#include <utility>      // std::pair, std::make_pair
#include <map>

#include "GPU/gpu_globals.hh"

class Agent;
class TableConstraint;
class Codec;

namespace CUDA {
class DPOPstate {
public:
	DPOPstate();
	~DPOPstate();

	void initialize(Agent& a);
	void compute_util_table(Agent& a);
	void compute_best_value(Agent & a);
	void compute_regret_table(Agent& a);
	int utilTableCudaVersion();
	int regretTableCudaVersion();

	// Phase 1
	int get_util(int row) {
		return pHostUtilTable[row];
	}

	// Phase 2
	void set_sep_value(int val, int vid) {
		pSepValues[vid] = val;
	}

	void set_xi_value(int val) {
		pXiAssignment = val;
	}

	void set_xi_util(int util) {
		pXiSubtreeUtil = util;
	}

	int get_sep_value(int vid) {
		return pSepValues[vid];
	}

	// Get the best value for this variable and a given world
	int get_xi_best_value();

	int get_xi_best_util() {
		return pXiSubtreeUtil;
	}

	int* getUtilTablePtr() {
		return pHostUtilTable;
	}

	// @ deprecated
	int* getUtilTableRow(size_t row) {
		return &pHostUtilTable[row];
	}

	size_t getUtilTableRows() {
		return pUtilTableRows;
	}

	size_t getUtilTableRowsAfterProj() {
		return pUtilTableRowsAfterProj;
	}

	void setChildUtilTableInfo(int chId, int* tablePtr, size_t tableSize);

	// Phase 3
	int* getConstraintSelectedValues() {
		return pHostConstraintSelectedValues;
	}

	// Phase 4
	int* getRegretTablePtr() {
		return pHostUtilTable;
	}
	int* getRegretTableRow(size_t row) {
		return &pHostUtilTable[row];
	}
	size_t getRegretTableRows() {
		return pUtilTableRows;
	}

	size_t getRegretTableRowsAfterProj() {
		return pUtilTableRowsAfterProj;
	}

	size_t getRegretTableCols() {
		return 1;
	}

	void setChildRegretTableInfo(int chId, int* tablePtr, size_t tableSize);

	// Misc
	int get_agent_id() const {
		return p_ai_id;
	}

	int get_nb_dcop_agents() const {
		return p_nb_agents;
	}

	int get_var_id() {
		return p_xi_id;
	}

	int get_dom_size() {
		return p_dom_size;
	}

	std::vector<int>& get_separator() {
		return pSeparatorSetId;
	}

	bool is_root() {
		return p_is_root;
	}

	bool isLeaf() {
		return p_is_leaf;
	}

	int get_nb_binary_constraints() {
		return p_nb_binary;
	}

	int get_children_info_size() {
		return p_cinfo_size;
	}

	int get_max_children_info_size() {
		return p_max_cinfo_size;
	}

	bool has_unary_constraints() {
		return p_has_unary;
	}

	size_t getChildTableRows(int chId) {
		return pChildIdToUtilTableRows[chId];
	}

	int* getChildTablePtr(int chId) {
		return pChildIdToUtilTablePtr[chId];
	}

	std::vector<int>& getChildrenId() {
		return pChildrenId;
	}

	void set_children_util_tables(std::vector<std::pair<int, int*> >& msgs);
	void aggregate_children_and_project();
	void saveConstraintValue();
	void print_host_util_table();
	void print_host_regret_table();
	void print_host_util_table_unproj();
	void print_host_regret_table_unproj();
	void aggregate_children_and_project_regret();

private:
	cudaEvent_t startEvent, stopEvent;
	float time_ms;
	float alloc_time_ms;  // time used by device to allocate / deallocate data (unnecessary on single agent)

	int _curr_idx;
	int p_ai_id;
	int p_xi_id;
	int p_dom_size;
	int p_nb_agents;
	std::vector<int> pSeparatorSetId;

	bool p_is_root;
	bool p_is_leaf;
	int p_nb_binary;
	int p_cinfo_size;
	int p_max_cinfo_size;
	bool p_has_unary;

	// UTIL Table without projection
	int* pHostUtilTable;
	size_t pUtilTableRows;
	size_t pUtilTableRowsAfterProj;

	//---------------------------------------------------------------------------//
	// INHERITED FROM PREVIOUS STATE  (cpu part)

	// Values retured after the first optimization (VALUE PHASE 1)
	// For each variable of the separator set we have a values
	std::map<int, int> pSepValues;

	std::vector<TableConstraint*> pConstraints;
	// The constraint value selected after DPOP
	int* pHostConstraintSelectedValues;

	// The best values returned after the UTIL propagation phase
	int pXiAssignment;
	int pXiSubtreeUtil;

	std::vector<int*> pChildIdToUtilTablePtr;
	std::vector<size_t> pChildIdToUtilTableRows;
	std::vector<int> pChildrenId;

	//---------------------------------------------------------------------------//
	// Children Tables integration on CPU
	std::vector<int> p_children_info;
	int* p_sep_value_child;
	int* p_sep_values;

private:

	int fast_encode(int* t, int t_size, int d) {
		int _d = d;
		int ofs = t[--t_size];
		while (t_size > 0) {
			ofs += t[--t_size] * _d;
			_d *= d;
		}
		return ofs;
	}

	void fast_decode(int code, int* t, int t_size, int d) {
		for (int i = t_size - 1; i >= 0; i--) {
			t[i] = code % d;
			code /= d;
		}
	}

	void fast_decode_next(int *t, int t_size, int d) {
		while (t[_curr_idx] == d - 1 && _curr_idx >= 0) {
			t[_curr_idx] = 0;
			_curr_idx--;
		}
		if (_curr_idx < 0)
			return;

		t[_curr_idx]++;
		_curr_idx = t_size - 1;
	}

	void reset_decode() {
		if (pSeparatorSetId.size() > 0) {
			for (int i = 0; i < pSeparatorSetId.size(); i++)
				p_sep_values[i] = 0;

			p_sep_values[pSeparatorSetId.size() - 1] = -1;
			_curr_idx = pSeparatorSetId.size() - 1;
		}
	}

};
}

#endif
