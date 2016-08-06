#ifndef ULYSSES_PROBLEM__IO__INPUT_SETTINGS_H_
#define ULYSSES_PROBLEM__IO__INPUT_SETTINGS_H_

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <rapidxml.hpp>

#include "Problem/IO/input.hh"

class InputSettings : public Input
{
public:
  typedef std::pair<std::string, std::vector<std::string> > solving_t;
  
  InputSettings(int argc, char* argv[]);

  virtual ~InputSettings() {};
        
  // Parses the statistical measures from an XML object and saves them into 
  // the object.
  void import(std::string p_filename);
  
  // It returns the preprocess description associated to this DCOP model.
  std::string preprocess() const
    { return p_preprocess; }
    
  // It returns the resolution method specified by this DCOP model. 
  std::string resolutionMethod() const
    { return p_resolution_method; }

  // It returns the parameters associated to the DCOP resolution method.
  std::vector<std::string> resolutionParameters() const
    { return p_resolution_params; }
  
  // It returns the cost of sending an intra-agent variables message.
  size_t innerMessageCost() const
    { return p_inner_message_cost; }

  // It returns the cost of sending an inter-agent variables message.  
  size_t outerMessageCost() const
    { return p_outer_message_cost;  }
    
  // It returns the simluated time timeout
  size_t simulatedTimeout() const
    { return p_simulated_timeout_ms; }

  // It returns the wallclock time timeout
  size_t wallclockTimeout() const
    { return p_wallclock_timeout_ms; }
  
  // It returns the maximum number of Non-Concurrent Constraint Checks.
  size_t NCCCsLimit() const
  {
    return p_NCCCs_limit;
  }
  
  // It returns the memory limit in bytes.
  size_t memoryLimit() const
  {
    return p_memory_limit;
  }

  // It returns true if a single solving strategy has been defined
  // for all agents. 
  bool singlePrivateSolver()
  {
    return (p_map_agent_search_strategy_params.find("*") != 
      p_map_agent_search_strategy_params.end());
  }
  
  // It returns the agent solving strategy
  solving_t privateSolver(std::string agent)
    { return p_map_agent_search_strategy_params[ agent ]; }
  
  // It returns the agent's solving strategy for its boundary variables, 
  // given it's preprocessing type and its' resolution method.
  solving_t boundarySolver();

  int get_elected_root() {
    return p_elected_root;
  }

  int get_max_gpu_size() {
    return p_max_gpu_size;
  }
  
  int get_heuristic() {
	  return p_heuristic;
  }

  void set_heuristic(int heuristic) {
	  p_heuristic = heuristic;
  }

  virtual std::string dump();
  
private:
  
  void parseDCOPsolver(rapidxml::xml_node<>* xdcop);
  
  void parseMetrics(rapidxml::xml_node<>* xmeasures);
  
  void parseTimeouts(rapidxml::xml_node<>* xmeasures);
  
  bool existsSavedPseudoTree(std::string file);

  void loadPseudoTree(std::string file);

private:
  // The preprocessing to be adopted by this DCOP
  std::string p_preprocess;

  // The DCOP resolution method.
  std::string p_resolution_method;

  // The parameters attached to the resolution method, i.e., the number of 
  // samples in DGibbs
  std::vector<std::string> p_resolution_params;
  
  // agent_name -> <strategy_name, {strategy parameters}>
  std::map<std::string, solving_t> p_map_agent_search_strategy_params;
  
  int p_inner_message_cost;

  int p_outer_message_cost;

  // Maximum Number of Non-Concurrent Constraint Checks
  int p_NCCCs_limit;
  
  // Maximum memory allocable per agent.
  int p_memory_limit;
    
  // The timout for the simulated running time in milliseconds.
  int p_simulated_timeout_ms;
  
  // The timout for the wall clock running time in milliseconds.
  int p_wallclock_timeout_ms;

  int p_elected_root;

  int p_heuristic;

  int p_max_gpu_size;

};
  
#endif // ULYSSES_INSTANCE_GENERATOR__IO__INPUT_SETTINGS_H_
