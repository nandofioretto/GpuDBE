// Utilities and system includes
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#include <vector>
#include <string>
#include <iostream>
#include <limits>       // std::numeric_limits

#include "GPU/gpu_globals.hh"
#include "GPU/gpu_data_allocator.hh"
#include "GPU/cuda_utils.hh"
#include "Utilities/permutations.hh"
#include "Kernel/types.hh"

using namespace CUDA;
using namespace std;

//#define VERBOSE

/* To be called only once for all Agents */
__global__ void gpu_InitGlobals(int dom_size, long int infty) {
	gdev_dom_size = dom_size;
	gdev_infty = infty;
}

__global__ void gpu_allocAgents(dev_class_DPOP_Agent* agents, int nb_agents) {
	gdev_nb_agents = nb_agents;
	gdev_DPOP_Agents = agents;
}

__global__ void gpu_allocVariables(dev_class_Variable* variables, int nb_vars) {
	gdev_nb_variables = nb_vars;
	gdev_Variables = variables;
}

__global__ void gpu_allocConstraints(dev_class_Constraint* constraints,
		int nb_cons) {
	gdev_nb_constraints = nb_cons;
	gdev_Constraints = constraints;
}

__global__ void gpu_initAgent(int ID, int cuda_xi_id, int cuda_dom_min,
		int cuda_dom_max, util_t* cuda_unary, int cuda_nb_binary_constraints,
		int* cuda_binary_constraints, int* cuda_binary_constraints_sel_values,
		size_t cuda_util_table_rows, util_t* cuda_util_table, int cuda_sep_size,
		int* cuda_sep, int cuda_children_size, int* cuda_children,
		int max_sep_ch) {
	gdev_DPOP_Agents[ID].id = ID;
	gdev_DPOP_Agents[ID].xi_id = cuda_xi_id;
	gdev_DPOP_Agents[ID].dom_min = cuda_dom_min;
	gdev_DPOP_Agents[ID].dom_max = cuda_dom_max;
	gdev_DPOP_Agents[ID].unary = cuda_unary;
	gdev_DPOP_Agents[ID].binary_con = cuda_binary_constraints;
	gdev_DPOP_Agents[ID].binary_con_sel_values =
			cuda_binary_constraints_sel_values;
	gdev_DPOP_Agents[ID].nb_binary_con = cuda_nb_binary_constraints;
	gdev_DPOP_Agents[ID].util_table = cuda_util_table;
	gdev_DPOP_Agents[ID].util_table_rows = cuda_util_table_rows;
	gdev_DPOP_Agents[ID].var_sep_id = cuda_sep;
	gdev_DPOP_Agents[ID].var_sep_size = cuda_sep_size;
	gdev_DPOP_Agents[ID].children_info = cuda_children;
	gdev_DPOP_Agents[ID].children_info_size = cuda_children_size;
	gdev_DPOP_Agents[ID].max_child_sep_size = max_sep_ch;
}

__global__ void gpu_initConstraint(int ID, int cuda_arity, int* cuda_scope,
		util_t* cuda_utils) {
	gdev_Constraints[ID].id = ID;
	gdev_Constraints[ID].arity = cuda_arity;
	gdev_Constraints[ID].scope_id = cuda_scope;
	gdev_Constraints[ID].utils = cuda_utils;
}



/**
 * Allocate the memory for the gloal containers:
 * TODO: use Thrust.
 */
void GPU_allocator::allocate_data(int nb_agents, int nb_variables, int nb_constraints, int dom_size) {
	cudaError_t error;

	// Create streams
	agtStream = new cudaStream_t[nb_agents];
	for (int i = 0; i < nb_agents; i++)
		cudaCheck( cudaStreamCreate(&agtStream[i]) );


	gpu_InitGlobals<<< 1, 1 >>> (dom_size, -100); // Problem is Maximize
	cudaCheck(cudaDeviceSynchronize());

	dev_class_Variable* variables;
	unsigned int mem_size_vars = nb_variables * sizeof(dev_class_Variable);
	cudaCheck(cudaMalloc((void **) &variables, mem_size_vars));
	gpu_allocVariables<<<1,1>>>(variables, nb_variables);
	cudaCheck(cudaDeviceSynchronize());
	CUDA::used_memory::incr_global(mem_size_vars);
	CUDAutils::check_memory();

	dev_class_Constraint* constraint;
	unsigned int mem_size_cons = nb_constraints * sizeof(dev_class_Constraint);
	cudaCheck(cudaMalloc((void **) &constraint, mem_size_cons));
	gpu_allocConstraints<<<1,1>>>(constraint, nb_constraints);
	cudaCheck(cudaDeviceSynchronize());
	CUDA::used_memory::incr_global(mem_size_cons);
	CUDAutils::check_memory();

	dev_class_DPOP_Agent* agents;
	unsigned int mem_size_agents = nb_agents * sizeof(dev_class_DPOP_Agent);
	cudaCheck(cudaMalloc((void **) &agents, mem_size_agents));
	gpu_allocAgents<<<1,1>>>(agents, nb_agents);
	cudaCheck(cudaDeviceSynchronize());
	CUDA::used_memory::incr_global(mem_size_agents);
	CUDAutils::check_memory();
}


// NOTE This function needs to be allocated after the pseudo-tree construction.
void GPU_allocator::init_agent(int cuda_ai_id, int cuda_xi_id, int cuda_dom_min,
		int cuda_dom_max, vector<util_t> unary,
		vector<int> binary_constraints, size_t cuda_util_table_rows,
		vector<int> sep, vector<int> children, int max_ch_sep_size) {

	int cuda_dom_size = cuda_dom_max - cuda_dom_min;

	// --------------------------------------------------------------------//
	// Allocate memory for Separator set best value storage
	int cuda_sep_size = sep.size();
	int* __sep = new int[cuda_sep_size];
	for (int i = 0; i < cuda_sep_size; i++)
		__sep[i] = sep[i];

	int* cuda_sep;
	unsigned int size = cuda_sep_size * sizeof(int);
	cudaCheck(cudaMalloc((void **) &cuda_sep,  size));
	cudaCheck(cudaMemcpy(cuda_sep, __sep, size, cudaMemcpyHostToDevice));
	cudaCheck(cudaDeviceSynchronize());
	CUDA::used_memory::incr_global(cuda_sep_size * sizeof(int));
	CUDAutils::check_memory();

	// --------------------------------------------------------------------//
	// Allocate memory for Children
	int cuda_children_size = children.size();
	int* __children = new int[cuda_children_size];
	for (int i = 0; i < cuda_children_size; i++)
		__children[i] = children[i];

	int* cuda_children;
	size = cuda_children_size * sizeof(int);
	cudaCheck(cudaMalloc((void **) &cuda_children, size));
	cudaCheck(cudaMemcpy(cuda_children, __children, size, cudaMemcpyHostToDevice));
	cudaCheck(cudaDeviceSynchronize());
	CUDA::used_memory::incr_global(cuda_children_size * sizeof(int));
	CUDAutils::check_memory();

	// --------------------------------------------------------------------//
	// Allocate memory for util table
	// util_t* cuda_util_table;
	// size_t cuda_ut_pitch; // new number of cols (with padding)
	// cudaCheck( cudaMallocPitch( &cuda_util_table, &cuda_ut_pitch, cuda_util_table_cols*sizeof(util_t), cuda_util_table_rows ) );
	// cudaCheck( cudaMemset2D( cuda_util_table, cuda_ut_pitch, 0, cuda_util_table_cols*sizeof(util_t), cuda_util_table_rows ));
	// cudaCheck( cudaDeviceSynchronize() );
	// CUDA::used_memory::incr_global(cuda_util_table_rows*cuda_ut_pitch*sizeof(util_t));
	// CUDAutils::check_memory();

	// --------------------------------------------------------------------//
	// Allocate memory and copy for constraints
	util_t* __unary = new util_t[cuda_dom_size];
	for (int i = 0; i < cuda_dom_size; i++) {
		__unary[i] = unary[i];
	}

	util_t* cuda_unary;
	size = cuda_dom_size * sizeof(util_t);
	cudaCheck(cudaMalloc((void **) &cuda_unary, size));
	cudaCheck(cudaMemcpy(cuda_unary, __children, size, cudaMemcpyHostToDevice));
	cudaCheck(cudaDeviceSynchronize());
	CUDA::used_memory::incr_global(size);
	CUDAutils::check_memory();

	int cuda_nb_binary_constraints = binary_constraints.size();
	int* __binary = new int[cuda_nb_binary_constraints];
	for (int i = 0; i < cuda_nb_binary_constraints; i++)
		__binary[i] = binary_constraints[i];

	int* cuda_binary_constraints;
	size = cuda_nb_binary_constraints * sizeof(int);
	cudaCheck(cudaMalloc((void **) &cuda_binary_constraints, size));
	cudaCheck(cudaMemcpy(cuda_binary_constraints, __binary, size, cudaMemcpyHostToDevice));
	cudaCheck(cudaDeviceSynchronize());
	CUDA::used_memory::incr_global(size);
	CUDAutils::check_memory();

	int* cuda_binary_constraints_sel_values;
	int nb_constr = (int) (cuda_nb_binary_constraints / 3);
	size = nb_constr * sizeof(int);
	cudaCheck(cudaMalloc((void **) &cuda_binary_constraints_sel_values, size));
	cudaCheck(cudaDeviceSynchronize());
	CUDA::used_memory::incr_global(size);
	CUDAutils::check_memory();

	gpu_initAgent<<<1,1>>>(cuda_ai_id, cuda_xi_id, cuda_dom_min, cuda_dom_max,
			cuda_unary, nb_constr, cuda_binary_constraints,
			cuda_binary_constraints_sel_values,
			0, NULL,
			cuda_sep_size, cuda_sep,
			cuda_children_size, cuda_children, max_ch_sep_size);
	cudaCheck(cudaDeviceSynchronize());

	delete[] __sep;
	delete[] __children;
	delete[] __unary;
	delete[] __binary;

}


void  GPU_allocator::init_constraint(int cuda_id, std::vector<int> scope, std::vector<util_t> utils)
{
	int cuda_arity = scope.size();

	// --------------------------------------------------------------------//
	// Allocate memory for scope
	int* __scope = new int[cuda_arity];
	for (int i = 0; i < cuda_arity; i++)
		__scope[i] = scope[i];

	int* cuda_scope;
	unsigned int size = cuda_arity * sizeof(int);
	cudaCheck(cudaMalloc((void ** )&cuda_scope, size));
	cudaCheck(cudaMemcpy(cuda_scope, __scope, size, cudaMemcpyHostToDevice));
	cudaCheck(cudaDeviceSynchronize());
	CUDA::used_memory::incr_global(size);
	CUDAutils::check_memory();

	// --------------------------------------------------------------------//
	// Allocate memory for utils
	unsigned int nrows = utils.size();
	util_t* __utils = new util_t[nrows];
	for (int i = 0; i < nrows ; i++) {
		__utils[i] = utils[i];
	}

	util_t* cuda_utils;
	size = nrows * sizeof(util_t);
	cudaCheck(cudaMalloc((void ** )&cuda_utils, size));
	cudaCheck(cudaMemcpy(cuda_utils, __utils, size, cudaMemcpyHostToDevice));
	cudaCheck(cudaDeviceSynchronize());
	CUDA::used_memory::incr_global(size);
	CUDAutils::check_memory();

	gpu_initConstraint<<<1,1>>>(cuda_id, cuda_arity, cuda_scope, cuda_utils);
	cudaCheck(cudaDeviceSynchronize());

	delete[] __scope;
	delete[] __utils;
  
#ifdef VERBOSE
  gpu_dump_constraint<<<1,1>>>(cuda_id);
  cudaCheck( cudaDeviceSynchronize() );
#endif
}
