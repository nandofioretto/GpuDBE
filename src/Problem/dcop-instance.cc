#include "Problem/dcop-instance.hh"
#include "Communication/scheduler.hh"
#include "Kernel/agent.hh"

DCOPinstance::DCOPinstance()
  : p_nb_agents(0), p_nb_variables(0), p_nb_constraints(0),
    p_elected_root(0), p_heuristic(2)
  { }


std::string DCOPinstance::dump() {
    std::string res = "DCOP sol:\t";
    
    for(auto &kv: p_dcop_solution)
      res += p_variables[ kv.first ]->name() + "\t";      
    res += "util \n";

    for(auto &kv: p_dcop_solution)
   	  res += std::to_string(kv.second) + "\t";

    res+= std::to_string(get_util()) + "\n";

    return res;
  }
