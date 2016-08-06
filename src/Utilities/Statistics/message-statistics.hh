#ifndef ULYSSES_UTILITIES__STATISTICS__MESSAGE_STATISTICS_H_
#define ULYSSES_UTILITIES__STATISTICS__MESSAGE_STATISTICS_H_

#include "Kernel/globals.hh"
#include "Utilities/Statistics/metrics.hh"

class MessageStatistics : public Metrics
{
public:
  MessageStatistics() 
  {
    Metrics::reset(); 
  }
    
  MessageStatistics(const MessageStatistics& other)
  {
    p_NCCCs = other.p_NCCCs;
    p_pseudo_NCCCs = other.p_pseudo_NCCCs;
    p_simulated_us = other.p_simulated_us;
    p_pseudo_simulated_us = other.p_pseudo_simulated_us;
    p_wallclock_us = other.p_wallclock_us;
  }
  
  MessageStatistics& operator=(const MessageStatistics& other)
  {
    if (this != &other) {
      p_NCCCs = other.p_NCCCs;
      p_pseudo_NCCCs = other.p_pseudo_NCCCs;
      p_simulated_us = other.p_simulated_us;
      p_pseudo_simulated_us = other.p_pseudo_simulated_us;
      p_wallclock_us = other.p_wallclock_us;
    }
    return *this;
  }
    
private:  
};

#endif // ULYSSES_UTILITIES__STATISTICS__MESSAGE_STATISTICS_H_