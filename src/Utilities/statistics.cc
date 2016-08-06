#include "Utilities/statistics.hh"
#include "Kernel/globals.hh"
#include "Kernel/agent.hh"
#include "Kernel/variable-factory.hh"
#include "Kernel/int-variable.hh"
#include "Problem/dcop-instance.hh"
#include "preferences.hh"

#include <cmath>

using namespace std;

// Start timer for an agent. 
map<string, map<oid_t, Statistics::timepoint_t> > Statistics::p_map_timer_start;

// Maps a timer unit measure to a timer. 
map<string, map<oid_t, size_t> > Statistics::p_map_timer;

// Maps a countable unit measure to a counter. To the counter is also 
// associated a cost, which multies the counter.
map<string, map<oid_t, Statistics::wvalue_t> > Statistics::p_map_counter;


void Statistics::reset(oid_t agent_id)
{
  for (auto& kv : p_map_timer)
    kv.second[ agent_id ] = 0;

  for (auto& kv : p_map_counter)
    kv.second[ agent_id ].first = 0;
}


std::string Statistics::Statistics::dump ()
{
  string res = "\t============ GPU U-DPOP statistics ============\n";
  size_t nccc = 0;
  size_t t_sim = 0;
  for(Agent* a : g_dcop->agents())
  {
    nccc = max(nccc, a->statistics().NCCC());
    t_sim = max(t_sim, a->statistics().simulatedTime());
  }
  // t_sim += getMaxTimer("simulated@initialization");
  
  //res += "\nProblem Solving\n";  
  // res += "NCCC                : " + to_string(nccc) + '\n';
  if (preferences::singleAgent) {
    res += "Tot Wall clock Time : " 
      + to_string( getTimer("wallclock") - getTimer("init") - getTimer("gpu-alloc") ) + '\n';
  } else {
    res += "Simulated Time      : " 
      + to_string(int(t_sim / 1000.00)) + '\n';
    res += "Tot Wall clock Time : " 
      + to_string( getTimer("wallclock") - getTimer("gpu-alloc") ) + '\n';
  }
  // res += "Algorithm Init time : ["
  //     + to_string(getTotalTimer("simulated@initialization")) + " : "
  //     + to_string(getMaxTimer("simulated@initialization")) +"]\n";
  return res;
}

  // res +="PD:NCCCs\tPD:smt\tPD:wct\tPS:NCCCs\tPS:nbInMsg\tPS:nbOutMsg\tPS:smt\tPS:wct\n";
std::string Statistics::dumpEmptyCSV(std::string txt, int n)
{
  std::string res = "NA";
  for (int i=1; i<n; i++)
    res += ",NA";
  res += "," + txt;
  return res;
}

std::string Statistics::dumpOutOfLimitsCSV(std::string limit)
{
  std::string res;
  if( !g_dcop)
  {    
    res += to_string((int)getMaxCounter("NCCCs@decomposition")) + ',';
    res += to_string(getMaxTimer("simulated@decomposition")) + ',';
    res += to_string(getTimer("wallclock@decomposition")) + ',';

    if (limit.compare("OOT") == 0) 
      res += dumpEmptyCSV("OOT", 5);
    else if (limit.compare("OOM") == 0) 
      res += dumpEmptyCSV("OOM", 5);
    else 
      res += dumpEmptyCSV("NA", 5);
  }
  else
    res = dumpCSV();
  return res;
}


std::string Statistics::dumpCSV()
{
  std::string cost = Constants::isFinite(g_dcop->cost()) ?
    to_string(g_dcop->cost()) :  "UNSAT";
  
  for(Agent* a : g_dcop->agents()) {
    if(a->statistics().simulatedTimeout()) cost = "OOT";
    else if(a->statistics().memoryLimit()) cost = "OOM";
  }
  
  std::string res;
    
  size_t nccc = 0;
  size_t t_sim = 0;
  size_t max_nb_inner_msg = 0;
  size_t max_nb_outer_msg = 0;
  size_t sum_nb_inner_msg = 0;
  size_t sum_nb_outer_msg = 0;
  for(Agent* a : g_dcop->agents())
  {
    nccc = max(nccc, a->statistics().NCCC());
    max_nb_inner_msg = max(max_nb_inner_msg,
      a->statistics().nbInnerMessageSent());
    max_nb_outer_msg = max(max_nb_outer_msg, 
      a->statistics().nbOuterMessageSent());
    sum_nb_inner_msg += a->statistics().nbInnerMessageSent();
    sum_nb_outer_msg += a->statistics().nbOuterMessageSent();        
    t_sim = max(t_sim, a->statistics().simulatedTime());
  }
  // t_sim += getMaxTimer("simulated@initialization");

  // res +="PD:NCCCs\tPD:smt\tPD:wct\tPS:NCCCs\tPS:nbInMsg\tPS:nbOutMsg\tPS:smt\tPS:wct\n";
  res += to_string((int)getMaxCounter("NCCCs@decomposition")) + ',';
  res += to_string(getMaxTimer("simulated@decomposition")) + ',';
  res += to_string(getTimer("wallclock@decomposition")) + ',';
  res += to_string(nccc) + ',';
  res += to_string(sum_nb_inner_msg) + ',';
  res += to_string(sum_nb_outer_msg) + ',';
  res += to_string(int(t_sim / 1000.00)) + ',';
  res += to_string(getTimer("wallclock")) + ',';
  res += cost;
  return res;
}
