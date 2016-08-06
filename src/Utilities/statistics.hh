#ifndef ULYSSES_UTILITIES__STATISTICS_H_
#define ULYSSES_UTILITIES__STATISTICS_H_

#include <map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>

#include "Kernel/globals.hh"

class Statistics
{
 public:
  typedef std::chrono::time_point<std::chrono::steady_clock> timepoint_t;
  typedef std::pair<size_t, double> wvalue_t;
  Statistics() { }
  
  // It registers a new timer. 
  // This function is usually called by the DCOP algorithm
  static void registerTimer(std::string measure, oid_t agent_id=0)
  {
    if(p_map_timer.find(measure) != p_map_timer.end() && 
      p_map_timer[measure].find(agent_id) != p_map_timer[measure].end())
      return;

      p_map_timer[ measure ][ agent_id ] = 0;
      p_map_timer_start[ measure ][ agent_id ] = timepoint_t::min();
  }

  // It registers a new counter. Used for example to count the number
  // of messages.
  // This function is usually called by the DCOP algorithm
  static void registerCounter(std::string measure, oid_t agent_id=0)
  {
    if(p_map_counter.find(measure) != p_map_counter.end() && 
      p_map_counter[measure].find(agent_id) != p_map_counter[measure].end())
      return;
    
    p_map_counter[ measure ][ agent_id ].first = 0;
    p_map_counter[ measure ][ agent_id ].second = 1.0;
  }

  // It associates a cost to an already existing countable measure.
  static void registerCost(const std::string measure, oid_t agent_id=0, double cost=1)
  {
    ASSERT( p_map_counter.find(measure) != p_map_counter.end(), 
	    "The measure: " << measure << " does not exists");
    p_map_counter[ measure ][ agent_id ].second = cost;
  }

  // It set to 0 all the statistical measures.
  static void reset(oid_t agent_id);

  // Start the specific timer as this time instance. 
  static void startTimer(std::string measure, oid_t agent_id=0)
  {
    p_map_timer_start[ measure ][ agent_id ] = std::chrono::steady_clock::now();
  }

  static void copyTimer(std::string measure, size_t timer, oid_t agent_id=0)
  {
    p_map_timer[ measure ][ agent_id ] = timer;
  }

  static size_t getTimer (std::string measure, oid_t agent_id=0)
  {
    return p_map_timer[ measure ][ agent_id ]; 
  }
  

  static void addTimer (std::string measure, size_t time_ms, oid_t agent_id=0)
  {
    p_map_timer[ measure ][ agent_id ] += time_ms;
  }

  static void stopwatch (std::string measure, oid_t agent_id=0)
  {
    ASSERT( p_map_timer_start.find(measure) != p_map_timer_start.end(), 
	    "The measure: " << measure << " does not exists");
    
    timepoint_t now = std::chrono::steady_clock::now();
    timepoint_t start = p_map_timer_start[ measure ][ agent_id ];
    std::chrono::milliseconds elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
    p_map_timer[ measure ][ agent_id ] += elapsed.count();
  }

  static void setCounter(std::string measure, oid_t agent_id=0, size_t msg_size=0)
  {
    ASSERT( p_map_counter.find(measure) != p_map_counter.end(), 
	    "The measure: " << measure << " does not exists");
    
    p_map_counter[ measure ][ agent_id ].first = msg_size;
  }

  static void increaseCounter(std::string measure, oid_t agent_id=0, size_t c=1)
  {
    ASSERT( p_map_counter.find(measure) != p_map_counter.end(), 
	    "The measure: " << measure << " does not exists");
    
      ASSERT( p_map_counter[ measure ].find(agent_id) != 
        p_map_counter[ measure ].end(), 
  	    "The agent: " << agent_id << " does not exists for measure "
          << measure << ".");
    
    p_map_counter[ measure ][ agent_id ].first += c;
  }
  
  static double getCounter(std::string measure, oid_t agent_id)
  {
    return p_map_counter[ measure ][ agent_id ].first * 
      p_map_counter[ measure ][ agent_id ].second;
  }

  static double getMaxCounter(std::string measure)
  {
    double res = 0;
    for (auto kv : p_map_counter[ measure ])
      res = std::max(res, kv.second.first * kv.second.second);
    return res;
  }

  static double getTotalCounter(std::string measure)
  {
    double res = 0;
    for (auto kv : p_map_counter[ measure ])
      res += kv.second.first * kv.second.second;
    return res;
  }
  
  static size_t getTotalTimer(std::string measure)
  {
    size_t res = 0;
    for (auto kv : p_map_timer[ measure ])
      res += kv.second;
    return res;
  }

  static size_t getMaxTimer(std::string measure)
  {
    size_t res = 0;
    for (auto kv : p_map_timer[ measure ])
      res = std::max(res, kv.second);
    return res;
  }

  static std::string dump();

  static std::string dumpCSV();

  static std::string dumpEmptyCSV(std::string txt, int n=9);

  static std::string dumpOutOfLimitsCSV(std::string limit);

private:
  // Start timer for an agent. 
  static std::map<std::string, std::map<oid_t, timepoint_t> > p_map_timer_start;

  // Maps a timer unit measure to a timer. 
  static std::map<std::string, std::map<oid_t, size_t> > p_map_timer;

  // Maps a countable unit measure to a counter. To the counter is also 
  // associated a cost, which multies the counter.
  static std::map<std::string, std::map<oid_t,wvalue_t> > p_map_counter;
};


#endif // ULYSSES_UTILITIES__STATISTICS_H_
