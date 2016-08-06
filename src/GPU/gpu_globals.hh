#ifndef ULYSSES_GPU_DEV_GLOBALS_H
#define ULYSSES_GPU_DEV_GLOBALS_H

/* Common dependencies */
#include <assert.h>
#include <stdlib.h> // for atoi
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include "Kernel/types.hh"
#include <cuda_runtime.h>

#define UNSAT          -123456

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

/*
 * Info for Current CUDA card.
 * TODO: This should go in cuda utils
 */
namespace CUDA {

  class info {
  public:
    static size_t global_memory; // in bytes
    static size_t shared_memory; // in bytes
    static size_t max_dim_grid;
  };
  
  class used_memory {
  public:
    static size_t global;
    static size_t shared;
    static size_t texture;
    static size_t constant;
    
    static bool check() {
      return (global < info::global_memory);
    }
    
    static void incr_global(size_t mem) {
      global += mem;
    }
  };

}

/*
 * The Variable class
 * @deprecated
 */
struct dev_class_Variable
{
  int id;
  int d_min; int d_max;
};

/**
 * The Constraint class
 * todo: This is a non-efficient storage. Store everything in one vector to make it more data
 * read friendly.
 */
struct dev_class_Constraint
{
  int id;
  int arity;	    // Fixed to 2
  int* scope_id;	// The position in the array gdev_Agent::variables_ID
  util_t* utils;   // The set of all utilities of the constraint
};

// Local Structures (Agent)
struct dev_class_DPOP_Agent
{
  int id;
  int xi_id;			// The variable ID of this agent
  int dom_min;
  int dom_max;

  // int** xi_assignment;  // (not used) Mirror of DPOPstate::p_xi_assignment
  int*  var_sep_id;     // variables in separator set of ai
  int   var_sep_size;
  
  // list pairs (agent id of children, size of their separator set)
  // list the following information:
  // [ch_id, ch_separator_size, i_1, i_2, ...]
  // where i_l are the index of the var_sep_id or -1
  int*  children_info;
  int   children_info_size;
  int   max_child_sep_size; 

  // unary: [d0 ... dn]
  util_t* unary;	// Unary constraints already solved for each d \in D_i

  int nb_binary_con; // if any the unary constraint is the first one in constraint_IDs

  // Here save [cid, xi_1, xi_2 ...]
  int* binary_con;  // Listed in same order as of gdev_Constraints;

  // where binary_con_sel_values[ i*nb_worlds ] correspond to cid of binary_con[ i*3 ];
  int* binary_con_sel_values;

  // UTIL PHASE ----------- 
  util_t*  util_table;     // Mirror of DPOPstate::p_util_table
  size_t   util_table_rows;
  // ----------------------

  // VALUE PHASE --------
  int best_value;
  util_t best_util;

};






///////////////////////////////////////////////////////////////////////////////

size_t g_hash_util( int d, int a, int T[] );
void g_hash_tuple(int* T, int d, int a, size_t idx );
__device__ int ipow(int base, int exp);
__device__ int cuda_hash_util( int d, int scope[], int a, int T[]);
__device__ int cuda_hash_util2( int d, int scope[], int a, int T[], int i_sobst, int i_val);
__device__ void cuda_hash_tuple(int* T, int d, int a, size_t idx );

inline __device__ size_t gcuda_encode(int* t, int t_size, int d) {
  size_t ofs=0;
  for(int i=0; i<t_size; i++) {
    ofs += t[i] * ipow(d, t_size-i-1);
  }
  return ofs;
}


inline __device__ void gcuda_decode(size_t code, int* t, int t_size, int d)
{
  for (int i = t_size-1; i >= 0; i--)
  {
    t[i] = code % d;
    code /= d;
  }
}


//struct cudaProblem {
//
//}

extern cudaStream_t* agtStream;


extern __device__ int gdev_nb_constraints;
extern __device__ int gdev_nb_variables;
extern __device__ int gdev_nb_agents;

extern __device__ struct dev_class_Constraint* gdev_Constraints;
extern __device__ struct dev_class_Variable*  gdev_Variables;
extern __device__ struct dev_class_DPOP_Agent*  gdev_DPOP_Agents;

// NOT USED
extern __device__ int gdev_dom_size;
extern __device__ int gdev_opt_type; // The optimization type (maximize / minimize [default])
extern __device__ int gdev_infty;    // The infinity cost



#endif // ULYSSES_GPU_DEV_GLOBALS_H
