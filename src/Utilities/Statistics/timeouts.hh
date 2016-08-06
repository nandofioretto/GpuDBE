#ifndef ULYSSES_UTILITIES__STATISTICS__TIMEOUTS_H_
#define ULYSSES_UTILITIES__STATISTICS__TIMEOUTS_H_

#include "Kernel/globals.hh"

class Timeouts
{
public:  
  Timeouts() 
    : p_NCCCs_limit(0),  p_memory_limit(0), p_simulated_timeout_us(0), p_wallclock_timeout_us(0)
  { }
  
  // It sets the simluated and wallclock timeouts
  void setTimeouts(size_t st_ms=0, size_t wct_ms=0, size_t ncccs=0, size_t mem=0)
  {
    p_simulated_timeout_us = (st_ms * 1000);
    p_wallclock_timeout_us = (wct_ms * 1000);
    p_NCCCs_limit = ncccs;
    p_memory_limit = mem;
  }
  
  // It returns the simluated time timeout
  size_t simulatedTimeout() const
  {
   return p_simulated_timeout_us; 
  }

  // It returns the wallclock time timeout
  size_t wallclockTimeout() const
  {
   return p_wallclock_timeout_us;
  }
  
  // It returns the maximum number of Non-Concurrent Constraint Checks.
  size_t NCCCsLimit() const
  {
    return p_NCCCs_limit;
  }
  
  // It returns the maximum memory allocable by one agent in bytes.
  size_t memoryLimit() const
  {
    return p_memory_limit;
  }
  
protected:  
  // Maximum Number of Non-Concurrent Constraint Checks
  size_t p_NCCCs_limit;
  
  // Maximum memory in bytes.
  size_t p_memory_limit;
    
  // The timout for the simulated running time in microseconds.
  size_t p_simulated_timeout_us;
  
  // The timout for the wall clock running time in microseconds.
  size_t p_wallclock_timeout_us;
};

#endif // ULYSSES_UTILITIES__STATISTICS__TIMEOUTS_H_