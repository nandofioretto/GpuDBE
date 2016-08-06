#ifndef GPU_CUDA_UTILS_H_
#define GPU_CUDA_UTILS_H_

#include <cuda_runtime.h>

namespace CUDA {
class CUDAutils {
public:
	static void initializeCUDA(int argc, char **argv);

	static void dump_agent_values(int id);
	static void dump_util_table(int id);
	static void dump_worlds_solution();
	static void dump_agent_info(int id);
	static void dump_constraint(int id);
	static void dump_used_memory();
	static void check_memory();
	static size_t get_nb_bytes_free_global_memory();
	static void set_max_global_memory(size_t mem);

	static void startTimer(cudaEvent_t& event, cudaStream_t stream = (cudaStream_t)0);
	static float stopTimer(cudaEvent_t& s_event, cudaEvent_t& e_event, cudaStream_t stream = (cudaStream_t)0);
};
}

#endif

