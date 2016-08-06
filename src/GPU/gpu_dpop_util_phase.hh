#ifndef GPu_DPOP_UTIL_PHASE_H_
#define GPu_DPOP_UTIL_PHASE_H_

#include <iostream>

namespace CUDA
{
  class DPOPstate; 

  class GPU_DPOPutilPhase
  {
  public: 
    GPU_DPOPutilPhase();
    
    ~GPU_DPOPutilPhase();

    void compute_util_table(DPOPstate& state, int version);
    
    float getKernelTimeMs() {
    	return compute_time_ms;
    }

    float getDataTransferTimeMs() {
    	return copy_time_ms;
    }

  private:
    /**
     * Support function
     * Device setup based on the UTIL Table Size
     */
    void setup_kernel(int _version_, DPOPstate& dpop_state);
    
    /**
     *
     */
    void set_device_table_size(int _version_, DPOPstate& dpop_state);
    
    // Copy children Util Table into global memory.
    // TODO: Important: this need to be done as soon as you receive a table from a children.
    void memcpyHtoD_children_tables (DPOPstate& dpop_state);
    
    void execute_kernel(int version, DPOPstate& dpop_state, size_t nbBlocksShift,
			size_t rowsToCompute, size_t runningNbBlocks, int nbThreads, 
			size_t sharedMem, cudaStream_t streamID);
    
    size_t nbThreads;
    size_t nbBlocks;
    size_t nbGroups;		// groups of threads
    size_t devPitch;
    size_t sharedMem;
    
    
    // Memory Util Table to compute
    size_t host_nTableRowsAfterProj;
    size_t host_nTableRowsNoProj;
    
    size_t host_nBytes;
    size_t dev_nBytes;
    size_t dev_nTableRows;
    
    int **dev_chTables;
    int **host_chTablesMirror; // the pointers to be copied on the device;
    int *dev_table;
    int *host_table;
    
    // Statistics
    //cudaEvent_t startEvent, stopEvent;
    cudaEvent_t startEventCpy, stopEventCpy;
    cudaEvent_t startEventCmp, stopEventCmp;
    float tot_time_ms, compute_time_ms, copy_time_ms;

    cudaStream_t* computeStreams;
    size_t n_computeStreams;
  };

}

#endif
