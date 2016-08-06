// CUDA and CUBLAS functions
// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <assert.h>
//#include <helper_string.h>  // helper for shared functions common to CUDA Samples
#include <helper_functions.h>
#include <helper_cuda.h>

#include <vector>
#include <string>
#include <iostream>

#include "GPU/cuda_utils.hh"
#include "GPU/gpu_globals.hh"
#include "preferences.hh"

using namespace CUDA;

__global__ void gpu_dump_agent_values(int ID) {
//  printf("Agent: %d\n", gdev_DPOP_Agents[ID].id);
//
//  printf("  values for each world :");
//  for(int i=0; i<gdev_DPOP_Agents[ID].util_table_cols; i++)
//    printf("%d ", gdev_DPOP_Agents[ ID ].best_value[i]);
//  printf("\n");
}

__global__ void gpu_dump_util_table(int ID) {
#ifdef FALSE
	int ut_pitch = gdev_DPOP_Agents[ID].util_table_pitch;

	printf("Agent: %d\n", gdev_DPOP_Agents[ID].id);

	int T_size = gdev_DPOP_Agents[ID].var_sep_size;
	int* T = new int[T_size];

	for(int row=0; row<gdev_DPOP_Agents[ ID ].util_table_rows; row++) {
		printf("  UTIL code: [%d] : <", row);
		gcuda_decode(row, T, T_size, gdev_dom_size);
		for( int t=0; t<T_size; t++ )
		printf("%d ", T[t]);
		printf("> : ");
		for(int wid=0; wid<gdev_nb_worlds; wid++) {
			printf("%d ", gdev_DPOP_Agents[ID].util_table[ (row * ut_pitch) + wid ]);
		}
		printf("\n");
	}

	printf("  var sep id:");
	for(int i=0; i<gdev_DPOP_Agents[ID].var_sep_size; i++)
	printf("%d ", gdev_DPOP_Agents[ ID ].var_sep_id[i]);
	printf("\n");
	delete[] T;
#endif
}

__global__ void gpu_dump_worlds_solution() {
//  printf("world\t");
//  for(int aid = 0; aid < gdev_nb_agents; aid++)
//    printf("a_%d\t", aid);
//  printf("\n");
//
//  for (int wid = 0; wid < gdev_nb_worlds; wid++)
//  {
//    printf("  w-%d\t",wid);
//    for(int aid = 0; aid < gdev_nb_agents; aid++) {
//      printf("%d\t", gdev_DPOP_Agents[aid].util_table[ wid ]);
//    }
//    printf("\n");
//  }

}


/**
 * Support function to print agent information.
 */
__global__ void gpu_dump_agent_info(int ID) {
	int dom_size = gdev_DPOP_Agents[ID].dom_max - gdev_DPOP_Agents[ID].dom_min + 1;

	printf("Agent: %d\n", gdev_DPOP_Agents[ID].id);
	printf("  nb cons: %d\n", gdev_DPOP_Agents[ ID ].nb_binary_con);
	printf("  binary cons: ");
	for(int i=0; i<gdev_DPOP_Agents[ ID ].nb_binary_con; i++) {
		int cid = gdev_DPOP_Agents[ ID ].binary_con[3*i];
		printf(" %d (%d %d )\n", cid,
				gdev_DPOP_Agents[ ID ].binary_con[ 3*i +1],
				gdev_DPOP_Agents[ ID ].binary_con[ 3*i + 2 ]);

		for (int x=0; x<dom_size; x++) {
			for (int y=0; y<dom_size; y++) {
				printf("  [%d, %d]: %d\n", x, y, gdev_Constraints[cid].utils[x*dom_size + y]);
			}
		}
	}
	printf("\n");

	printf("  util table rows (encode sep values) : %d\n", gdev_DPOP_Agents[ID].util_table_rows);

	printf("  var sep id:");
	for (int i = 0; i < gdev_DPOP_Agents[ID].var_sep_size; i++)
		printf("%d ", gdev_DPOP_Agents[ID].var_sep_id[i]);
	printf("\n");

	printf("  children info:");
	for (int i = 0; i < gdev_DPOP_Agents[ID].children_info_size; i++)
		printf("%d ", gdev_DPOP_Agents[ID].children_info[i]);
	printf("max ch sep size: %d", gdev_DPOP_Agents[ID].max_child_sep_size);
	printf("\n");

}

/**
 * Support function to print constraint information.
 */
__global__ void gpu_dump_constraint(int ID) {
	printf("constraint: %d [%d, %d]\n", gdev_Constraints[ID].id,
			gdev_Constraints[ID].scope_id[0], gdev_Constraints[ID].scope_id[1]);
	printf("  utils: \n");
	int N = gdev_dom_size;

	for (int x = 0; x < N; x++) {
		for (int y = 0; y < N; y++) {
			printf("  [%d, %d]: ", x, y);
			printf("%d ", gdev_Constraints[ID].utils[x * N + y]);
			printf("\n");
		}
	}

}

// fast truncation of double-precision to integers
#define CUMP_D2I_TRUNC (double)(3ll << 51)
// computes r = a + b subop c unsigned using extended precision
#define VADDx(r, a, b, c, subop) \
  asm volatile("vadd.u32.u32.u32." subop " %0, %1, %2, %3;" :  \
	       "=r"(r) : "r"(a) , "r"(b), "r"(c));

// computes a * b mod m; invk = (double)(1<<30) / m
__device__ __forceinline__
unsigned mul_m(unsigned a, unsigned b, volatile unsigned m,
		volatile double invk) {

	unsigned hi = __umulhi(a * 2, b * 2); // 3 flops
	// 2 double instructions
	double rf = __uint2double_rn(hi) * invk + CUMP_D2I_TRUNC;
	unsigned r = (unsigned) __double2loint(rf);
	r = a * b - r * m; // 2 flops

	// can also be replaced by: VADDx(r, r, m, r, "min") // == umin(r, r + m);
	if ((int) r < 0)
		r += m;
	return r;
}

void CUDAutils::dump_agent_values(int id) {
	printf("\n----------------------------------------------\n");
	gpu_dump_agent_values<<<1,1>>>(id);
	cudaCheck(cudaDeviceSynchronize());
	printf("\n----------------------------------------------\n");
}

void CUDAutils::dump_util_table(int id) {
	printf("\n----------------------------------------------\n");
	gpu_dump_util_table<<<1,1>>>(id);
	cudaCheck(cudaDeviceSynchronize());
	printf("\n----------------------------------------------\n");
}

void CUDAutils::dump_worlds_solution() {
	printf("\n----------------------------------------------\n");
	gpu_dump_worlds_solution<<<1,1>>>();
	cudaCheck(cudaDeviceSynchronize());
	printf("\n----------------------------------------------\n");
}

void CUDAutils::dump_agent_info(int id) {
	printf("\n----------------------------------------------\n");
	gpu_dump_agent_info<<<1,1>>>(id);
	cudaCheck(cudaDeviceSynchronize());
	printf("\n----------------------------------------------\n");
}

void CUDAutils::dump_constraint(int id) {
	printf("\n----------------------------------------------\n");
	gpu_dump_constraint<<<1,1>>>(id);
	cudaCheck(cudaDeviceSynchronize());
	printf("\n----------------------------------------------\n");
}

void CUDAutils::dump_used_memory() {
	std::cout << "Global Memory used: " << CUDA::used_memory::global << " / "
			<< CUDA::info::global_memory<< std::endl;
}

void CUDAutils::check_memory() {
	if (!CUDA::used_memory::check()) {
		std::cout << "Out of Memory: Insufficient GPU Global Memory\n";
		std::cout << "Global Memory used: " << CUDA::used_memory::global
				<< " / " << CUDA::info::global_memory
				<< std::endl;
		exit(-2);
	}
}

size_t CUDAutils::get_nb_bytes_free_global_memory() {
	// reserve extra 128 MB in CUDA memory
	return max(0, (int) (CUDA::info::global_memory - CUDA::used_memory::global)); // - (128 * (1024*1024));
}

void CUDAutils::set_max_global_memory(size_t sizeMB) {
	printf("Setting Max GPU global Memory to = %zu MB\n", sizeMB);
	CUDA::info::global_memory = sizeMB * 1024 * 1024;
}

void CUDAutils::startTimer(cudaEvent_t& event, cudaStream_t stream) {
	cudaCheck(cudaEventRecord(event, stream));
}

float CUDAutils::stopTimer(cudaEvent_t& s_event, cudaEvent_t& e_event, cudaStream_t stream) {
	float ms;
	cudaCheck(cudaEventRecord(e_event, stream));
	cudaCheck(cudaEventSynchronize(e_event));
	cudaCheck(cudaEventElapsedTime(&ms, s_event, e_event));
	return ms;
}

void CUDAutils::initializeCUDA(int argc, char **argv) {
	// By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
	int devID = 0;
	float memLimit = preferences::maxDevMemory;

	if (checkCmdLineFlag(argc, (const char **) argv, "device")) {
		devID = getCmdLineArgumentInt(argc, (const char **) argv, "device");
		cudaCheck(cudaSetDevice(devID));
	}
	if (checkCmdLineFlag(argc, (const char **) argv, "memlimitMB")) {
		memLimit = getCmdLineArgumentFloat(argc, (const char **) argv, "memlimitMB");
		CUDA::info::global_memory = memLimit * 1e+6;
	}
	if (checkCmdLineFlag(argc, (const char **) argv, "memlimitGB")) {
		memLimit = getCmdLineArgumentFloat(argc, (const char **) argv, "memlimitGB");
		CUDA::info::global_memory = memLimit * 1e+9;
	}

	// get number of SMs on this GPU

	cudaDeviceProp deviceProp;
	cudaCheck(cudaGetDeviceProperties(&deviceProp, devID));

	if (memLimit == 0) {
	  CUDA::info::global_memory = deviceProp.totalGlobalMem;
	} else {
	  if (CUDA::info::global_memory > deviceProp.totalGlobalMem) 
	    CUDA::info::global_memory = deviceProp.totalGlobalMem;
	}

	CUDA::info::shared_memory = deviceProp.sharedMemPerBlock;
	CUDA::info::max_dim_grid = deviceProp.maxGridSize[0];

	if (!preferences::silent) {
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\t"
				"Global Memory: %.2f gb shared memory: %zu bytes\n\n", devID,
				deviceProp.name, deviceProp.major, deviceProp.minor,
				CUDA::info::global_memory / 1000000000.00,
				CUDA::info::shared_memory);
	}
	// use a larger block size for Fermi and above
	int block_size = (deviceProp.major < 2) ? 16 : 32;
}
