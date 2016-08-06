#ifndef ULYSSES_PROBLEM__DCOP_MODEL_H_
#define ULYSSES_PROBLEM__DCOP_MODEL_H_

#include <string>
#include <vector>
#include <rapidxml.hpp>

#include "Problem/dcop.hh"
#include "Kernel/globals.hh"
#include "Kernel/agent.hh"

class Agent;
class IntVariable;
class IntDomain;
class Constraint;
class InputProblem;


// The DCOP model class which implements the DCOP.
// It holds agents, variables, domains and constraints of the Problem.
class DCOPmodel : public DCOP
{
public:  
  DCOPmodel(InputProblem& problem);
    
  virtual ~DCOPmodel();
  
  // It imports the DCOP structures, parsing the file given as a parameter.
  void import(std::string file);
  
  // It returns the DCOP agents
  std::vector<Agent*> agents() const
  { 
    return p_agents; 
  }

  // It returns the DCOP agent associated to the ID given as a parameter.
  Agent* agent(oid_t agent_id) const
  { 
    for (Agent* a : p_agents)
      if( a->id() == agent_id)
        return a;
    return nullptr;
  }

  // It returns the DCOP variables
  std::vector<IntVariable*> variables() const
  { 
    return p_variables; 
  }
  
  // It returns the DCOP constraints
  std::vector<Constraint*> constraints() const
  { 
    return p_constraints; 
  }
  
  // It returns the optimization type assiciated to this DCOP model.
  DCOPinfo::optType optimization() const
  { 
    return p_optimization; 
  }
    
  // It returns the number of agents of the DCOP model.
  size_t nbAgents() const
  { 
    return p_agents.size(); 
  }

  // It returns the number of variables of the DCOP model.
  size_t nbVariables() const
  { 
    return p_variables.size(); 
  }

  // It returns the number of constraints of the DCOP model.
  size_t nbConstraints() const
  { 
    return p_constraints.size(); 
  }
  
private:      
  // Parses the agents from an XML object and saves them into p_agents.
  void parseXMLAgents(rapidxml::xml_node<>* root);

  // Parses the variables from an XML object and saves them into p_variables.
  void parseXMLVariables(rapidxml::xml_node<>* root);

  // Parses the constraints from an XML object, saves them into p_constraints.
  void parseXMLConstraints(rapidxml::xml_node<>* root);

private:
  // The DCOP agents
  std::vector<Agent*> p_agents;
  
  // The DCOP variables
  std::vector<IntVariable*> p_variables;
  
  // The DCOP domains
  std::vector<IntDomain*> p_domains;
  
  // The DCOP constraints
  std::vector<Constraint*> p_constraints;
  
  // The optimization type of this DCOP model.
  DCOPinfo::optType p_optimization;

};

#endif // ULYSSES_PROBLEM__DCOP_MODEL_H_
