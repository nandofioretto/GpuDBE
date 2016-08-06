#ifndef GPu_DPOP_VALUE_PHASE_H_
#define GPu_DPOP_VALUE_PHASE_H_

namespace CUDA
{
  class DPOPstate; 

  class GPU_DPOPvaluePhase
  {
  public: 
    static void compute_best_value(DPOPstate& state, size_t& gpu_time_us);

    static void get_best_values(DPOPstate& state);

    static void copy_constraint_sel_values(DPOPstate& dpop_state, size_t& gpu_time_us);

  };
}

#endif
