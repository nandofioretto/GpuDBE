#include <vector>
#include <string>
#include <iostream>
#include <limits>       // std::numeric_limits
#include <cassert>
#include <cmath>       /* ceil */

// Utilities and system includes
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "GPU/gpu_dpop_value_phase.hh"
#include "GPU/gpu_globals.hh"
#include "GPU/gpu_data_allocator.hh"
#include "GPU/cuda_utils.hh"
#include "GPU/cuda_dpop_state.hh"

//#define VERBOSE

using namespace CUDA;

__global__ void compute_best_value_of_world(int aid, int d_size, int nb_binary, int nb_var_sep);


__global__ void gpu_copy_values(int* ret_val, int* ret_util, int aid) 
{
  int wid = threadIdx.x;
  ret_val[ wid ] = gdev_DPOP_Agents[aid].best_value;//[ wid ];
  ret_util[ wid ] = gdev_DPOP_Agents[aid].best_util;//[ wid ];
}

void GPU_DPOPvaluePhase::get_best_values(DPOPstate& state) 
{
  if(!state.is_root()) 
    return;

  int nb_worlds = 1;
  int* dev_values;
  int* dev_utils;
  int* host_values = new int[ nb_worlds ];
  int* host_utils  = new int[ nb_worlds ];
  cudaCheck( cudaMalloc(&dev_values, nb_worlds * sizeof(int)) );
  cudaCheck( cudaMalloc(&dev_utils,  nb_worlds * sizeof(int)) );
  
  // copy best values on host
  gpu_copy_values<<<1, nb_worlds>>>( dev_values, dev_utils, state.get_agent_id() );
  cudaCheck( cudaDeviceSynchronize() );
  
  cudaCheck( cudaMemcpy( host_values, dev_values, nb_worlds * sizeof(int), cudaMemcpyDeviceToHost) );
  cudaCheck( cudaMemcpy( host_utils, dev_utils, nb_worlds * sizeof(int), cudaMemcpyDeviceToHost) );
  
  for(int i=0; i<nb_worlds; i++) {
    state.set_xi_value( host_values[ i ]);
    state.set_xi_util ( host_utils[ i ] );
    printf("   Inserting value %d - util %d for world %d\n", host_values[ i ], host_utils[ i ],  i );
  }
  
  delete[] host_values;
  delete[] host_utils;
  cudaCheck( cudaFree( dev_values ) );
  cudaCheck( cudaFree( dev_utils ) );
}


__global__ void gpu_copy_constraints_sel_values( int a_id, int* values) {
  gdev_DPOP_Agents[ a_id ].binary_con_sel_values = values;
}


__global__ void gpu_print_constraints_sel_values( int a_id )
{
//  int nConstr = gdev_DPOP_Agents[ a_id ].nb_binary_con;
//
//  for( int i=0; i < nConstr; i++ ) {
//    printf("cid= %d ", gdev_DPOP_Agents[ a_id ].binary_con[ i*3 ]);
//    for (int wid = 0; wid < nWorlds; wid++) {
//    	printf("%d ", gdev_DPOP_Agents[ a_id ].binary_con_sel_values[ i * nWorlds + wid ] );
//    }
//    printf("\n");
//  }
}

void GPU_DPOPvaluePhase::copy_constraint_sel_values(DPOPstate& dpop_state, size_t& gpu_time_us) 
{
  int a_id = dpop_state.get_agent_id();
  int* dev_values;
  int nb_worlds = 1;// dpop_state.get_nb_worlds();
  int nb_constraints = dpop_state.get_nb_binary_constraints();
  cudaCheck( cudaMalloc(&dev_values, nb_constraints * nb_worlds * sizeof(int)) );
  cudaCheck( cudaMemcpy( dev_values, dpop_state.getConstraintSelectedValues(),
			nb_constraints * nb_worlds * sizeof(int), cudaMemcpyHostToDevice) );
  gpu_copy_constraints_sel_values<<<1, 1>>>( a_id, dev_values );
  cudaCheck( cudaDeviceSynchronize() );
  // gpu_print_constraints_sel_values<<<1,1>>>( a_id );
  // cudaCheck( cudaDeviceSynchronize() );
  
  // cudaCheck(cudaFree(dev_values));
}








/////////////////////////////////////////////////////////////////////////////////////////////////////////////
void GPU_DPOPvaluePhase::compute_best_value(DPOPstate& dpop_state, size_t& gpu_time_us)
{
  int a_id = dpop_state.get_agent_id();
  int d_size = dpop_state.get_dom_size();
  int nb_worlds = 1;//dpop_state.get_nb_worlds();
  int nb_binary = dpop_state.get_nb_binary_constraints();
  int nb_var_sep  = dpop_state.get_separator().size();
  bool has_unary  = dpop_state.has_unary_constraints();

  size_t nb_threads = nb_worlds;
  size_t nb_blocks  = 1;
  int shared_mem = nb_binary * 3 * sizeof(int)
    + nb_worlds * nb_var_sep * sizeof(int);
  if( has_unary )
    shared_mem += nb_worlds * d_size * sizeof(int);

  assert(shared_mem <= CUDA::info::shared_memory);
  
  compute_best_value_of_world<<<nb_blocks, nb_threads, shared_mem>>>
    (a_id, d_size, nb_binary, nb_var_sep);
  cudaCheck( cudaDeviceSynchronize() );

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
// C U D A   K E R N E L   ( VALUE message computation )
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void compute_best_value_of_world(int aid, int d_size, int nb_binary, int nb_var_sep)
{
#ifdef false
  // pitch depends from the number of worlds, thus it is equal for all constraints (and utils)
  int pitch     = gdev_Constraints[0].utils_pitch;     // (global mem)

  // ---------------------------------------------------------------------- //
  // Registers
  // ---------------------------------------------------------------------- //
  int wid = threadIdx.x;
  int util_di = 0;
  int util = 0;
  int util_tmp = 0;
  int best_di = 0; // only used by root.

  int _i = 0, _di = 0, _id = 0, _x1 = 0, _x2 = 0;
  int _scope_x1, _scope_x2;

  // ---------------------------------------------------------------------- //
  // Shared Memory Allocation
  // ---------------------------------------------------------------------- //
  // TODO: All this information needs to be stored in a contiguous array on global memory.
  // here we partition it in a smart way.
  extern __shared__ int __smem[];
  int* __constraint = __smem;
  _i = nb_binary * 3;
  
  int* __sep_values = &__smem[ _i + wid * nb_var_sep ];  // one per world

  if (nb_binary > 0 && threadIdx.x == 0) {
    memcpy(__constraint, gdev_DPOP_Agents[aid].binary_con, nb_binary*3*sizeof(int));
  }

  // Retrieve best Table row
  for (_i = 0; _i < nb_var_sep; _i++) {
    _id = gdev_DPOP_Agents[aid].var_sep_id[ _i ];
    __sep_values[ _i ] = gdev_DPOP_Agents[_id].best_value;// [ wid ]; // (global mem)
  }
  __syncthreads();

 
  // ---------------------------------------------------------------------- //
  // Compute Util Table Entry Value
  util = UNSAT;
  for (_di = 0; _di < d_size; _di++) // Reticulate across al domain elements
  {
    util_di = 0;   
    //--------------------------------------------------------------------- //
    // Binary constraints
    //--------------------------------------------------------------------- //
    for (_i = 0; _i < nb_binary; _i++)
    {
      _id = __constraint[ 3*_i ];
      _x1 = __constraint[3*_i + 1];
      _x2 = __constraint[3*_i + 2];
      _scope_x1 = _x1 == -1 ? _di : __sep_values[_x1];
      _scope_x2 = _x2 == -1 ? _di : __sep_values[_x2];
      
      util_tmp = gdev_Constraints[_id].utils[ (_scope_x1 * d_size + _scope_x2) * pitch + wid ]; // (global mem)

      // printf("[KERNEL] thread %d world %d : constr_%d [%d %d] UTIL= %d\n", threadIdx.x, wid, _id, _scope_x1, _scope_x2, util_tmp);
      
      if (util_tmp == UNSAT) {util_di = UNSAT; break; }
      util_di += util_tmp;
    }


    if (util_di == UNSAT) 
      continue;

    if (util == UNSAT || util_di > util) { 
      util = util_di;
      best_di = _di;
    }
  }

  gdev_DPOP_Agents[aid].best_value/*[ wid ]*/ = best_di; // (global mem)
  gdev_DPOP_Agents[aid].best_util/*[ wid ]*/ = util;     // (global mem)
#endif
}
