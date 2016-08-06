#ifndef ULYSSES_UTILITIES__STATISTICS__METRICS_H_
#define ULYSSES_UTILITIES__STATISTICS__METRICS_H_

#include "Kernel/globals.hh"

class Metrics
{
public:  
  Metrics()
    : p_NCCCs(0), p_simulated_us(0), p_wallclock_us(0),
    p_pseudo_NCCCs(0), p_pseudo_simulated_us(0)
  { }
  
  ~Metrics()
  { }

  // It resets the current statistics.
  void reset()
  {
    p_NCCCs = 0;
    p_simulated_us = 0;
    p_wallclock_us = 0;
    p_pseudo_NCCCs = 0;
    p_pseudo_simulated_us = 0;
  }
  
  // It sets the NCCC as the count given as a parameter.
  void setNCCC(size_t count)
  {
    p_NCCCs = count;
  }
  
  // It returns the NCCC.
  size_t NCCC() const 
  {
    return p_NCCCs;
  }

  // It sets the pseudo-NCCC as the count given as a parameter.
  void setPseudoNCCCs(size_t count)
  {
    p_pseudo_NCCCs = count;
  }
  
  // It returns the pseudo-NCCC.
  size_t pseudoNCCCs() const
  { 
    return p_pseudo_NCCCs; 
  }
   
  // It sets the simulated time as the time given as a parameter.
  void setSimulatedTime(size_t time=0)
  {
    p_simulated_us = time;
  }

  void addSimulatedTime(size_t time)
  {
    p_simulated_us += time;
  }
  
  // It returns the Simulated time elapsed.
  size_t simulatedTime() const
  {
    return p_simulated_us; 
  }

  // It sets the pseudo simulated time as the time given as a parameter.
  void setPseudoSimulatedTime(size_t time=0)
  {
    p_pseudo_simulated_us = time;
  }
      
  // It returns the pseudo simulated time elapsed.
  size_t pseudoSimulatedTime() const
  {
    return p_pseudo_simulated_us; 
  }

  // It sets the wallclock time as the time given as a parameter.
  void setWallclockTime(size_t time)
  {
    p_wallclock_us = time;
  }

  // It returns the Wall Clock time elapsed.
  size_t wallclockTime() const
  {
    return p_wallclock_us; 
  }

  void resetPseudoMeasures()
  {
    p_pseudo_NCCCs = 0;
    p_pseudo_simulated_us = 0;
  }
  
protected:  
  // Number of Non-Concurrent Constraint Checks
  size_t p_NCCCs;
  
  // The number of non-concurrent constraint checks within the pseudo-agents
  // It holds the last NCCCs count executed by some pseudo-agent.
  size_t p_pseudo_NCCCs;
    
  // The simulated running time in microseconds.
  size_t p_simulated_us;

  // The number of non-concurrent constraint checks within the pseudo-agents
  // It holds the last NCCCs count executed by some pseudo-agent.
  size_t p_pseudo_simulated_us;
  
  // The wall clock running time in microseconds.
  size_t p_wallclock_us;
};

#endif // ULYSSES_UTILITIES__STATISTICS__METRICS_H_
