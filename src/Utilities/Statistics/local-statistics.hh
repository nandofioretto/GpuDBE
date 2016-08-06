#ifndef ULYSSES_UTILITIES__STATISTICS__LOCAL_STATISTICS_H_
#define ULYSSES_UTILITIES__STATISTICS__LOCAL_STATISTICS_H_

#include <chrono>
#include <string>

#include "Kernel/globals.hh"
#include "Utilities/Statistics/metrics.hh"
#include "Utilities/Statistics/timeouts.hh"
#include "Utilities/Statistics/message-statistics.hh"

class MessageStatistics;

class LocalStatistics : public Metrics, public Timeouts
{
public:
  typedef std::chrono::time_point<std::chrono::steady_clock> timepoint_t;
  
  LocalStatistics()
    : p_nb_inner_msg_sent(0), p_nb_outer_msg_sent(0), 
      p_inner_msg_cost(0), p_outer_msg_cost(0), p_used_memory(0)
  {
    Metrics::reset();
    p_stopwatch_start = std::chrono::steady_clock::now();
  }

  ~LocalStatistics()
  { }
  
  LocalStatistics& operator=(const LocalStatistics& other)
  {
    if (this != &other)
    {
      p_NCCCs = other.p_NCCCs;
      p_pseudo_NCCCs = other.p_pseudo_NCCCs;
      p_simulated_us = other.p_simulated_us;
      p_pseudo_simulated_us = other.p_pseudo_simulated_us;      
      p_wallclock_us = other.p_wallclock_us;
      p_used_memory  = other.p_used_memory;
      
      setTimeouts(other.Timeouts::simulatedTimeout(), 
                  other.Timeouts::wallclockTimeout(), 
                  other.Timeouts::NCCCsLimit(),
                  other.Timeouts::memoryLimit());

      p_nb_inner_msg_sent = other.p_nb_inner_msg_sent;
      p_nb_outer_msg_sent = other.p_nb_outer_msg_sent;
      p_inner_msg_cost = other.p_inner_msg_cost;
      p_outer_msg_cost = other.p_outer_msg_cost;
    }
    return *this;    
  }

  // It updates the NCCCs if the ones received are bigger than the ones
  // stored.
  // This routine is called at receiving of a (any) new message.
  void update(MessageStatistics& msg_stats)
  {
    incrSimulatedTime(msg_stats.pseudoSimulatedTime());
    incrNCCCs(msg_stats.pseudoNCCCs());
  
    // Increments the local statistics with the units from the pseudo-agents.
    if(msg_stats.NCCC() > NCCC())
      setNCCC(msg_stats.NCCC());

    if(msg_stats.simulatedTime() > simulatedTime()) {
      setSimulatedTime(msg_stats.simulatedTime());
      // setStartTimer(); // NO
    }
  }

  void incrNCCCs(size_t count=1)
  {
    p_NCCCs += count;
    p_pseudo_NCCCs += count;
  }

  void incrSimulatedTime(size_t count=0)
  {
    p_simulated_us += count;
    p_pseudo_simulated_us += count;
  }
  
  // It sets the number of messages sent as the count given as a parameter.
  void setNbInnerMessageSent(size_t count)
  {
    p_nb_inner_msg_sent = count;
  }

  // It sets the number of messages sent as the count given as a parameter.
  void setNbOuterMessageSent(size_t count)
  {
    p_nb_outer_msg_sent = count;
  }

  // It increments the number of inner messages sent of a count given as a 
  // parameter.
  void incrNbInnerMessageSent(size_t count=1)
  {
    p_nb_inner_msg_sent += count;
  }

  // It increments the number of outer messages sent of a count given as a 
  // parameter.
  void incrNbOuterMessageSent(size_t count=1)
  {
    p_nb_outer_msg_sent += count;
  }
  
  // It increases the amount of used memory
  void incrUsedMemory(size_t mem)
  {
    p_used_memory += mem;
  }

  // It returns the number of messages sent to agents running on the same 
  // machine.
  size_t nbInnerMessageSent() const
  {
    return p_nb_inner_msg_sent;
  }

  // It returns the number of messages sent to agents running on a different 
  // machine.
  size_t nbOuterMessageSent() const
  {
    return p_nb_outer_msg_sent;
  }

  // It returns the number of messages sent.
  size_t nbMessageSent() const
  {
    return p_nb_inner_msg_sent + p_nb_outer_msg_sent;
  }
  
  // It sets the message costs.
  void setMessageCosts(size_t inner_cost, size_t outer_cost)
  {
    p_inner_msg_cost = inner_cost;
    p_outer_msg_cost = outer_cost;
  }
  
  // It returns the inner message cost in terms of NCCC.
  size_t innerMessageCost() const
  {
     return p_inner_msg_cost; 
  }

  // It returns the outer message cost in terms of NCCC.
  size_t outerMessageCost() const
  {
    return p_outer_msg_cost;
  }
  
  size_t usedMemory() const
  {
    return p_used_memory;
  }
  
  // It returns whether a timeout has been reached.
  bool timeouts()
  {
    if(simulatedTimeout()){
      std::cout << "Simulated Timeout reached : " 
                << Timeouts::simulatedTimeout() / 1000.00 << " ms\n";
      return true;
    }
    if(wallclockTimeout()){
      std::cout << "Wallclock Timeout reached : "
                << Timeouts::wallclockTimeout() / 1000.00 << " ms\n";
      return true;
    }
    if(memoryLimit()){
      std::cout << "Memory Limit reached : "
                << Timeouts::memoryLimit() << " bytes\n";
      return true;
    }
    return false;
  }
  
  // It returns wether the simulated timout has been reached.
  bool simulatedTimeout()
  {
    return (Timeouts::simulatedTimeout() != 0 && 
            stopwatch() > Timeouts::simulatedTimeout());
  }

  // It returns wether the wallclock timout has been reached.
  bool wallclockTimeout()
  {
    return (Timeouts::wallclockTimeout() != 0 && 
            stopwatch() > Timeouts::wallclockTimeout());
  }
  
  bool memoryLimit()
  {
    return (Timeouts::memoryLimit() != 0 &&
            usedMemory() > Timeouts::memoryLimit());
  }
  
  // Start the specific timer (simulated only) as this time instance. 
  void setStartTimer()
  {
    p_stopwatch_start = std::chrono::steady_clock::now();
  }
  
  // Stop the current clock and returns the total runtime the agent has
  // been recorded from the first start.
  size_t stopwatch()
  {
    timepoint_t now = std::chrono::steady_clock::now();
    std::chrono::microseconds elapsed =
      std::chrono::duration_cast<std::chrono::microseconds>
        (now - p_stopwatch_start);
    return (p_simulated_us + elapsed.count());
  }
  
  std::string dump() const;
  
private:    
  // The number of Messages exchanged among agents running on the same machine
  size_t p_nb_inner_msg_sent;

  // The number of Messages exchanged among agents running on different machines
  size_t p_nb_outer_msg_sent;
  
  // The cost (in terms of NCCCs of messages exchanged among agents running
  // on the same machine)
  size_t p_inner_msg_cost;

  // The cost (in terms of NCCCs of messages exchanged among agents running
  // on different machines)
  size_t p_outer_msg_cost;
  
  // The start time for a measurement.  
  timepoint_t p_stopwatch_start;
  
  // The maximum memory allocable by one agent (in bytes)
  size_t p_used_memory;
};

#endif // ULYSSES_UTILITIES__STATISTICS__LOCAL_STATISTICS_H_
