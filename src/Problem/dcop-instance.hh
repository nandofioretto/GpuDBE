#ifndef ULYSSES_PROBLEM__DCOP_INSTANCE_H_
#define ULYSSES_PROBLEM__DCOP_INSTANCE_H_

#include <unordered_map>
#include <vector>
#include <string>

#include "Kernel/globals.hh"
#include "Communication/scheduler.hh"
#include "Kernel/agent.hh"
// #include "Problem/dcop.hh"
// #include "Problem/dcop-model.hh"

class Agent;
class IntVariable;
class IntDomain;
class Constraint;

// The DCOP instance abstract class.
class DCOPinstance
{
public:
  
  DCOPinstance();
  
  // It solves the DCOP problem
  virtual void solve()
  {
    for(auto& kv : p_agents)
      Scheduler::FIFOinsert(kv.second->id());
    Scheduler::run();
  }
  
  // It returns the DCOP agent with id given as a parameter.
  Agent& agent(oid_t id)
  {
    ASSERT(id < p_nb_agents, "No agent with id "<<id<<" was found.");
    return *p_agents[ id ];
  }

  // It returns the DCOP variable with id given as a paramter.
  IntVariable& variable(oid_t id)
  {
    ASSERT(id < p_nb_variables, "No variable with id "<<id<<" was found.");
    return *p_variables[ id ]; 
  }

  // It returns the DCOP constraint with id given as a paramter.
  Constraint& constraint(oid_t id)
  { 
    ASSERT(id < p_nb_constraints, "No constraint with id "<<id<<" was found.");
    return *p_constraints[ id ];
  }

  // It returns the DCOP agents with list of id given as a parameter.
  std::vector<Agent*> agents(std::vector<oid_t> ids)
  {
    std::vector<Agent*> res(ids.size());
    for(int i=0; i<ids.size(); ++i)
      res[ i ] = p_agents[ ids[ i ] ];
    return res;
  }

  // It returns the DCOP variables with list of id given as a parameter.
  std::vector<IntVariable*> variables(std::vector<oid_t> ids)
  {
    std::vector<IntVariable*> res(ids.size());
    for(int i=0; i<ids.size(); ++i)
      res[ i ] = p_variables[ ids[ i ] ];
    return res;
  }

  // It returns the DCOP variables.
  std::vector<IntVariable*> variables()
  {
    std::vector<IntVariable*> res;
    for(auto& kv : p_variables)
      res.push_back(kv.second);
    return res;
  }

  // It returns the DCOP constraints with list of id given as a parameter.
  std::vector<Constraint*> constraints(std::vector<oid_t> ids)
  {
    std::vector<Constraint*> res(ids.size());
    for(int i=0; i<ids.size(); ++i)
      res[ i ] = p_constraints[ ids[ i ] ];
    return res;
  }

  // It returns all the DCOP agents.
  std::vector<Agent*> agents()
  {
    std::vector<Agent*> res(p_agents.size());
    int i=0;
    for(auto& kv : p_agents)
      res[ i++ ] = kv.second;
    return res;
  }


  // It returns all the DCOP agents.
  std::vector<Constraint*> constraints()
  {
    std::vector<Constraint*> res(p_constraints.size());
    int i=0;
    for(auto& kv : p_constraints)
      res[ i++ ] = kv.second;
    return res;
  }

  
  // It returns the number of the DCOP agents.
  size_t nbAgents() const 
  { 
    return p_nb_agents; 
  }

  // It returns the number of the DCOP variables.
  size_t nbVariables() const 
  { 
    return p_nb_variables; 
  }

  // It returns the number of the DCOP constraints.
  size_t nbConstraints() const 
  { 
    return p_nb_constraints; 
  }
  
  // It returns true if the DCOP instance is of type maximization.
  bool maximize()
  { 
    return p_optimization == DCOPinfo::kMaximize; 
  }

  // It returns true if the DCOP instance is of type minimization.
  bool minimize()
  { 
    return p_optimization == DCOPinfo::kMinimize; 
  }
  

  virtual std::vector<std::pair<std::string,int> > decodeSolution()  = 0;

  // It sets the value associated to variable given as a parameter, to be 
  // part of the final solution.
  void set_var_solution(oid_t vid, int value)
  {
    p_dcop_solution[ vid ] = value;
  }

  int get_var_solution(oid_t vid)
  {
    return p_dcop_solution[ vid ];
  }
  
  void setCost(cost_t cost)
  { 
    p_cost = cost; 
  }

  cost_t cost() const
  { 
    return p_cost; 
  }
  
  // Given an agent id in the current model it returns the agent id associated
  // to the agent of the original std model.
  oid_t stdModelAgentId(oid_t agent_id)
  {
    return p_map_dcop_std_agents_id[ agent_id ];
  }

  void set_util(int util)
  {
    p_dcop_util = util;
  }

  cost_t get_util()
  {
    return p_dcop_util;
  }

  void set_elected_root(int id) {
    p_elected_root = id;
  }

  int get_elected_root() {
    return p_elected_root;
  }

  void set_heuristic(int heur) {
	  p_heuristic = heur;
  }

  int get_heuristic() {
    return p_heuristic;
  }

  std::string dump() ;


protected:
  // The DCOP agents
  std::unordered_map<oid_t, Agent*> p_agents;
  
  // The number of agents in the preprocessed DCOP model.
  size_t p_nb_agents;
  
  // The DCOP variables
  std::unordered_map<oid_t, IntVariable*> p_variables;
  
  // The number of variables in the preprocessed DCOP model.
  size_t p_nb_variables;
    
  // The DCOP constraints
  std::unordered_map<oid_t, Constraint*> p_constraints;
  
  // The number of constraints in the preprocessed DCOP model.
  size_t p_nb_constraints;
  
  // From current model agent id -> to std model agent id.
  std::unordered_map<oid_t, oid_t> p_map_dcop_std_agents_id;
  
  // The optimization type of this DCOP model.
  DCOPinfo::optType p_optimization;
    
  // The DCOP best solution found. Note that this is the solution of this
  // problem instance.
  std::unordered_map<oid_t, int > p_dcop_solution;
  
  // The cost associated to the DCOP best solution found.
  cost_t p_cost;

  cost_t p_dcop_util;

  int p_elected_root;

  int p_heuristic;
};

#endif // ULYSSES_PROBLEM__DCOP_INSTANCE_H_
