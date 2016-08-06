#ifndef ULYSSES_GPU_DATA_ALLOCATOR_H
#define ULYSSES_GPU_DATA_ALLOCATOR_H

#include <vector>
#include <utility>      // std::pair, std::make_pair
#include <iostream>
#include <cuda_runtime.h>

#include "Kernel/types.hh"

class DCOPinstance;

namespace CUDA {
class GPU_allocator {
public:

	static void allocate_data(int nb_agents, int nb_variables,
			int nb_constraints, int dom_size);

	static void init_agent(int cuda_ai_id, int cuda_xi_id, int cuda_dom_min,
			int cuda_dom_max, std::vector<util_t> unary,
			std::vector<int> binary_constraints, size_t cuda_util_table_rows,
			std::vector<int> sep, std::vector<int> children,
			int max_ch_sep_size);

	static void init_constraint(int cuda_id, std::vector<int> scope,
			std::vector<util_t> utils);

	static void dump_util_table(int agentID);

    //template<class T*>
    static int* allocate_PinnedHostArray(size_t bytes) {
    	int* pinnedHost;
    	cudaMallocHost((void**)&pinnedHost, bytes);      // host pinned
    	return pinnedHost;
    }

    template<class T>
    static void allocate_PinnedHostArray(T* array, size_t bytes) {
    	cudaMallocHost((void**)&array, bytes);      // host pinned
    }


};

};

#endif // ULYSSES_GPU_DATA_ALLOCATOR_H
