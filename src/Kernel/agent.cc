#include "Kernel/agent.hh"
#include "Kernel/int-variable.hh"
#include "Utilities/utils.hh"
#include "Utilities/variable-utils.hh"
#include "Problem/dcop-instance.hh"
#include <set>
#include <vector>
#include <algorithm>

using namespace std;

Agent::Agent()
  : nb_boundary_vars_(0), nb_private_vars_(0), nb_local_cons_(0),
    ordering_(nullptr), DCOP_protocol_(nullptr)
{ }


Agent::~Agent()
{ }


void Agent::registerConstraint(Constraint* con)
{
  // Add constraint Neighbours
  for (IntVariable* v : con->scope()) 
  {
    if (v->ownerId() != this->id())
      addNeighbour( v->ownerId() );
  }

  bool found = 
    std::find(constraints_.begin(), constraints_.end(), con) 
    != constraints_.end();
  
  // If the constraint contains at least one variable owned by 
  // some other agent, than that is a inter-agent constraint. Otherwise
  // it is a local constraint.
  for (IntVariable* v : con->scope())
  {
    if (v->ownerId() != this->id() and not found) {
      constraints_.push_back( con );
      return;
    }
  }
  
  if (!found) {
    constraints_.insert(constraints_.begin() + nb_local_cons_, con);
    nb_local_cons_++;  
  }
}


void Agent::addNeighbour(oid_t id)
{
  if (std::find(neighbours_.begin(), neighbours_.end(), id) 
      == neighbours_.end())
    neighbours_.push_back(id);
}


//#ifdef FALSE
void Agent::orderContextVariables()
{
  std::vector<IntVariable*> v_boundary;
  std::vector<IntVariable*> v_private;
  std::vector<IntVariable*> v_context;

  // Scan boundary and non-local variables: this is done by scanning
  // all inter-agent constraints and analyzing which variables in their
  // scope is owned by this agent (boundary) or not (other)
  for (int i=0; i<nbInterAgentConstraints(); ++i)
  {
    for (IntVariable* v : interAgentConstraintAt(i).scope())
      if (v->ownerId() == this->id())
        Utils::insertOnce(v, v_boundary);
      else
        Utils::insertOnce(v, v_context);
  }

  // Removes elements of v_boundary from v_private
  // std::set_difference(p_variables.begin(), p_variables.end(),
  //         v_boundary.begin(), v_boundary.end(),
  //         std::inserter(v_private, v_private.begin()));
  //
  
  v_private = Utils::exclude(Utils::merge(v_boundary, v_context), p_variables);

  // Initialize context_vars vector as specifed in its description.
  p_variables.clear();
  for (IntVariable* v : v_boundary) p_variables.push_back(v);
  for (IntVariable* v : v_private)  p_variables.push_back(v);
  for (IntVariable* v : v_context)  p_variables.push_back(v);
  nb_boundary_vars_ = v_boundary.size();
  nb_private_vars_ = v_private.size();

  ASSERT(Utils::intersect(boundaryVariableIDs(), privateVariableIDs()).empty(),
    "Boundary and private variables overlap!\n");
}
//#endif

#ifdef FALSE // This is wrong!
void Agent::orderContextVariables()
{  
  std::vector<oid_t> vcontext;
  std::vector<oid_t> vboundary;
  std::vector<oid_t> vprivate;
  std::vector<oid_t> vall = VariableUtils::getID( p_variables );
  
  // Scan boundary and non-local variables: this is done by scanning
  // all inter-agent constraints and analyzing which variables in their 
  // scope is owned by this agent (boundary) or not (other)
  for (Constraint* c : intraAgentConstraints()) {
    for (IntVariable* v : c->scope()) {
      if (v->ownerId() == ObjectInfo::id()) {
        Utils::insertOnce(v->id(), vboundary);        
      }
      else{
        Utils::insertOnce(v->id(), vcontext);        
      }
    }
  }
  
  vprivate = Utils::exclude( Utils::merge(vboundary, vcontext), vall );

  ASSERT(Utils::intersect(vboundary, vprivate).empty(), 
    "Boundary and private variables overlap!\n");
  
  // Initialize context_vars vector as specifed in its description. 
  std::vector<IntVariable*> V;
  for (oid_t v : vboundary) 
    V.push_back(p_variables[ Utils::findIdx(vall, v) ]);
  for (oid_t v : vprivate)
  V.push_back(p_variables[ Utils::findIdx(vall, v) ]);
  for (oid_t v : vcontext)
    V.push_back(p_variables[ Utils::findIdx(vall, v) ]);
    
  p_variables = V;
    
  nb_boundary_vars_ = vboundary.size();
  nb_private_vars_ = vprivate.size();
}
#endif

string Agent::dump() const
{
  string result;
  result += "Agent " + this->name() + " (Boundary: ";
  for (int i=0; i<nbBoundaryVariables(); ++i)
    result += boundaryVariableAt(i).name() + " ";
  result += "|Private: ";
  for (int i=0; i<nbPrivateVariables(); ++i)
    result += privateVariableAt(i).name() + " ";
  result += "|Context: ";
  for (int i=0; i<nbContextVariables(); ++i)
    result += contextVariableAt(i).name() + " ";
  result += ")  Constraints: ";
  
  for (int i=0; i<nbConstraints(); ++i)
    result += constraintAt(i).name() + " ";
  result += "\n  ";
  // result += "local cos: " + std::to_string(nb_local_cons_) + " / " 
  //   + std::to_string(constraints_.size());

  result += "  Private solver: " + p_private_solver.first + " ";
  for(std::string s  : p_private_solver.second)
    result += s + " ";

  result += "  Boundary solver: " + p_boundary_solver.first + " ";
  for(std::string s  : p_boundary_solver.second)
    result += s + " ";
  
  return result;
}
