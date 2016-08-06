#include <unordered_map>

#include "Problem/dcop-standard.hh"
#include "Kernel/agent.hh"
#include "Kernel/int-variable.hh"
#include "Kernel/constraint.hh"
#include "Problem/IO/input-settings.hh"

using namespace std;

DCOPstandard::DCOPstandard(DCOPmodel& model, InputSettings& settings)
{
  // Map Agents
  for (Agent* a : model.agents()) {
    Statistics::registerTimer("simulated@decomposition", a->id());
    Statistics::startTimer("simulated@decomposition", a->id());
    
    p_agents[ a->id() ] = a;
    p_map_dcop_std_agents_id [ a->id() ] = a->id();
    
    InputSettings::solving_t private_solver = 
      (settings.singlePrivateSolver()) ? settings.privateSolver("*") 
      : settings.privateSolver(a->name());
    
    InputSettings::solving_t boundary_solver = settings.boundarySolver();

    a->setPrivateSolver(private_solver);
    a->setBoundarySolver(boundary_solver);

    Statistics::stopwatch("simulated@decomposition", a->id());
  }
   
  p_nb_agents = model.nbAgents();

  // Map Variables
  for (IntVariable* v : model.variables()){
    p_variables[ v->id() ] = v;
    p_map_dcop_var_names[ v->id() ] = v->name();
  }
  p_nb_variables = model.nbVariables();
  
  // Map Constraints
  for (Constraint* c : model.constraints())
    p_constraints[ c->id() ] = c;
  p_nb_constraints = model.nbConstraints();
  
  for (auto& kv : p_agents) {
    kv.second->orderContextVariables();
  }
  
  p_optimization = model.optimization(); 
}


DCOPstandard::~DCOPstandard()
{
  
}
