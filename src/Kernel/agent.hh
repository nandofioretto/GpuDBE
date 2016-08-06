#ifndef ULYSSES_KERNEL__AGENTS__AGENT_H_
#define ULYSSES_KERNEL__AGENTS__AGENT_H_

#include <vector>
#include <string>
#include <memory>

#include "Algorithms/algorithm.hh"
#include "Algorithms/Orderings/ordering.hh"
#include "Communication/mailbox.hh"
#include "Kernel/globals.hh"
#include "Kernel/object-info.hh"
#include "Kernel/constraint.hh"
#include "Kernel/int-variable.hh"
#include "Kernel/solution.hh"
#include "Utilities/Statistics/local-statistics.hh"
#include "Problem/IO/input-settings.hh"
#include "Utilities/statistics.hh"

//class Protocol;
class Ordering;
class PseudoTreeOrdering;
class MessageStatistics;

class Agent : public ObjectInfo
{
public:
  
  // Constructs an agent and initializes private members to default values.
  Agent();

  ~Agent();
  
  // It returns the number of the agent boundary variables. 
  size_t nbBoundaryVariables() const
  {
    return nb_boundary_vars_;
  }
  
  // It returns the number of the agent private variables. 
  size_t nbPrivateVariables() const
  {
    return nb_private_vars_;
  }

  // It returns the number of the agent local variables.
  size_t nbLocalVariables() const
  {
    return nb_boundary_vars_ + nb_private_vars_;
  }

  // It returns the number of the neighbouring agent variables which share
  // a constraint with the boundary variables of this agent.
  size_t nbContextVariables() const
  {
    return p_variables.size() - nbLocalVariables();
  }

  // It returns the size of the context variables. 
  size_t size() const
  {
    return p_variables.size();
  }

  // It constructs and returs the IDs of the boundary variables.
  std::vector<oid_t> boundaryVariableIDs() const
  {
    std::vector<oid_t> res(nbBoundaryVariables());
    for (int i=0; i<nbBoundaryVariables(); ++i)
      res[ i ] = boundaryVariableAt(i).id();
    return res;
  }

  // It constructs and returs the ID of the private variables.
  std::vector<oid_t> privateVariableIDs() const
  {
    std::vector<oid_t> res(nbPrivateVariables());
    for (int i=0; i<nbPrivateVariables(); ++i)
      res[ i ] = privateVariableAt(i).id();
    return res;
  }

  // It constructs and returs the ID of set of the local variables.
  std::vector<oid_t> localVariableIDs() const
  {
    std::vector<oid_t> res(nbLocalVariables());
    for (int i=0; i<nbLocalVariables(); ++i)
      res[ i ] = localVariableAt(i).id();
    return res;
  }

  // It constructs and returs the ID of the context variables.
  std::vector<oid_t> contextVariablesID() const
  {
    std::vector<oid_t> res(nbContextVariables());
    for (int i=0; i<nbContextVariables(); ++i)
      res[ i ] = contextVariableAt(i).id();
    return res;
  }
  
  // It constructs and returs the set of the private variables.
  std::vector<IntVariable*> privateVariables()
  {
    std::vector<IntVariable*> res(nb_private_vars_);
    for(int i = nb_boundary_vars_; i < nbLocalVariables(); ++i)
      res[ i - nb_boundary_vars_ ] = p_variables[ i ];
    return res;
  }

  // It constructs and returs the set of boundary variables.
  std::vector<IntVariable*> boundaryVariables()
  {
    std::vector<IntVariable*> res(nb_boundary_vars_);
    for(int i = 0; i < nb_boundary_vars_; ++i)
      res[ i ] = p_variables[ i ];
    return res;
  }

  // It constructs and returs the set of local variables.
  std::vector<IntVariable*> localVariables()
  {
    std::vector<IntVariable*> res(nbLocalVariables());
    for(int i = 0; i < nbLocalVariables(); ++i)
      res[ i ] = p_variables[ i ];
    return res;
  }

  // It constructs and returs the set of context variables.
  std::vector<IntVariable*> contextVariables()
  {
    std::vector<IntVariable*> res(nbContextVariables());
    for(int i = nbLocalVariables(); i < p_variables.size(); ++i)
      res[ i ] = p_variables[ i ];
    return res;
  }

  // It returns the i-th boundary variable of the agent.
  IntVariable& boundaryVariableAt(int i) const
  {
    return *p_variables[ i ];
  }

  // It returns the i-th private variable of the agent.
  IntVariable& privateVariableAt(int i) const
  {
    return *p_variables[ nb_boundary_vars_ + i ];
  }

  // It returns the i-th local variable of the agent.
  IntVariable& localVariableAt(int i) const
  {
    return *p_variables[ i ];
  }

  // It returns the i-th variable non-local variable.
  IntVariable& contextVariableAt(int i) const
  {
    return *p_variables[ nbLocalVariables() + i ];
  }

  // It returns the i-th variable in the agent context.
  IntVariable& variableAt(int i) const
  {
    return *p_variables[ i ];
  }

  size_t nbConstraints() const
  {
    return constraints_.size();
  }

  size_t nbIntraAgentConstraints() const
  {
    return nb_local_cons_;
  }

  size_t nbInterAgentConstraints() const
  {
    return nbConstraints() - nb_local_cons_;
  }

  // It constructs and returs the IDs of the agent's constraints.
  std::vector<oid_t> constraintIDs() const
  {
    std::vector<oid_t> res(nbConstraints());
    for (int i=0; i<nbConstraints(); ++i)
      res[ i ] = constraintAt(i).id();
    return res;
  }

  // It constructs and returs the IDs of the agent's local constraints.
  std::vector<oid_t> intraAgentConstraintIDs() const
  {
    std::vector<oid_t> res(nbIntraAgentConstraints());
    for (int i=0; i<nbIntraAgentConstraints(); ++i)
      res[ i ] = intraAgentConstraintAt(i).id();
    return res;
  }

  std::vector<Constraint*> intraAgentConstraints() const
  {
    std::vector<Constraint*> res(nbIntraAgentConstraints());
    for (int i=0; i<nbIntraAgentConstraints(); ++i)
      res[ i ] = constraints_[ i ];
    return res;
  }

  // It constructs and returs the IDs of the inter-agent constraints.
  std::vector<oid_t> interAgentConstraintIDs() const
  {
    std::vector<oid_t> res(nbInterAgentConstraints());
    for (int i=0; i<nbInterAgentConstraints(); ++i)
      res[ i ] = interAgentConstraintAt(i).id();
    return res;
  }

  std::vector<Constraint*> interAgentConstraints() const
  {
    std::vector<Constraint*> res(nbInterAgentConstraints());
    for (int i=0; i<nbInterAgentConstraints(); ++i)
      res[ i ] = constraints_[ i + nb_local_cons_ ];
    return res;
  }

  // It returns the i-th constraint among those involving the variables
  // in the agent context.
  Constraint& constraintAt(int i) const
  {
    return *constraints_[ i ];
  }

  // It returns the i-th constraint among those involving exclusively the 
  // agent local variables.
  Constraint& intraAgentConstraintAt(int i)  const
  {
    return *constraints_[ i ];
  }

  // It returns the i-th constraint among those shared with other agents.
  Constraint& interAgentConstraintAt(int i) const
  {
    return *constraints_[ i + nb_local_cons_ ];
  }

  // Order the variables contained in p_variables as specified in 
  // the container description.
  void orderContextVariables();

  // It returns the set of the agent neigbours id.
  std::vector<oid_t>& neighbours() // DEPRECATED
  {
    return neighbours_;
  }

  std::vector<oid_t>& neighboursID()
    { return neighbours_; }

  // It returns the number of neighbours of this agent.
  size_t nbNeighbours() const
  {
    return neighbours_.size();
  }

  // It returns the mailbox associated to the agent.
  Mailbox& openMailbox()
  {
    return mailbox_;
  }

  // It sets the agent ordering as the one given as a parameter.
  // The algorithm initializing the agent ordering has unique control of the 
  // ordering it creates. When the algorithm ends, it passes the control of 
  // the ordering to this agent.
  void setOrdering(std::shared_ptr<Ordering> ordering)
  {
    ordering_ = std::move(ordering);
  }
  
  // It returns the agent Ordering
  Ordering& ordering() const
  {
    ASSERT(ordering_, "Trying to access to the agent ordering which "
	   << "has not being specified.");
    return *ordering_;
  }

  void setPtNode(std::shared_ptr<PseudoTreeOrdering> pt)
  {
    pt_node = std::move(pt);
  }

  PseudoTreeOrdering& ptNode() {
    return *pt_node;
  }

  // It register the DCOP protocol associated to this agent. 
  void registerProtocol(Algorithm* protocol)
  {
    DCOP_protocol_ = std::unique_ptr<Algorithm>(protocol);
    DCOP_protocol_->initialize();
  }

  // It executes the protocol associated with the agent.
  void runProtocol()
  {
    DCOP_protocol_->run();
  }
  
  void updateStatState()
  {
    //size_t sim_time = statistics().stopwatch();
    //statistics().setSimulatedTime(sim_time);    
    Statistics::stopwatch("wallclock");
    if (!g_dcop) {
      // we could also update DCOPs here!
      Statistics::copyTimer("simulated@decomposition", (statistics().simulatedTime() / 1000.00), id());
      Statistics::copyTimer("wallclock@decomposition", Statistics::getTimer("wallclock"));
      // Statistics::setCounter("NCCCs@decomposition", a->id(), a->statistics().NCCC());
    }
  }

  // It abort the execution of the protocol associated with the agent
  void checkOutOfLimits()
  {
    if (statistics().simulatedTimeout()) 
    {
      updateStatState();      
      std::cout << Statistics::dumpOutOfLimitsCSV("OOT") << std::endl;
      exit(1);
    }
    else if (statistics().memoryLimit()) 
    {
      updateStatState();
      std::cout << Statistics::dumpOutOfLimitsCSV("OOM") << std::endl;
      exit(2);
    }
  }

  void saveSolution(std::vector<std::pair<oid_t, int> > sol)
  {
    dcop_solution_ = sol;
    // for (std::pair<oid_t, int> p : sol)
    //   DCOPinfo::solution[ p.first ] = p.second;
  }

  std::vector<std::pair<oid_t, int> >& solution()
  {
    return dcop_solution_;
  }

  // Sets the agent solving strategy to the type given as a prameter. 
  void setPrivateSolver(InputSettings::solving_t strategy)
  {
    p_private_solver = strategy;
  }

  // Sets the agent solving strategy to the type given as a prameter. 
  void setBoundarySolver(InputSettings::solving_t strategy)
  {
    p_boundary_solver = strategy;
  }

  // Returns the agent solving strategy type for the private variables
  std::string privateSolver()
  { 
    return p_private_solver.first; 
  }

  // Returns the agent solving strategy type for the boundary variables
  std::string boundarySolver()
  { 
    return p_boundary_solver.first; 
  }

  // Returns the agent solving strategy parameters for the private variables
  std::vector<std::string> privateSolverParameters()
  { 
    return p_private_solver.second; 
  }

  // Returns the agent solving strategy parameters for the boundary variables
  std::vector<std::string> boundarySolverParameters()
  { 
    return p_boundary_solver.second; 
  }  
  
  // Returns the agent description.
  std::string dump() const;

  // It register the variable given as a parameter as a variable owned
  // by the agent.
  void registerVariable(IntVariable* var)
  {
    if (std::find(p_variables.begin(), p_variables.end(), var) 
      == p_variables.end())
      p_variables.push_back(var);
  }

  // It register a constraint in the constraints list in the order specified
  // by the container description.
  void registerConstraint(Constraint* con);

  
  void addNeighbour(oid_t id);

  // It returns the agent local statics.
  LocalStatistics& statistics()
  {
    return p_local_statistics;
  }
  
  // Calls the routine to update the agent's statistics given the statistics
  // carried in the message given as a parameter.
  void updateStatistics(MessageStatistics& msg_stats)
  {
     p_local_statistics.update(msg_stats);
  }

  void set_cuda_data(int nbc, int nci, int max_ch_sep_size, int nb_unary) {
    p_nb_binary_con = nbc;
    p_children_info_size = nci;
    p_has_unary = nb_unary > 0;
    p_max_ch_set_size = max_ch_sep_size;
  }

  int get_max_ch_set_size() {
    return p_max_ch_set_size;
  }

  int get_cuda_nb_binary() {
    return p_nb_binary_con;
  }

  int get_cuda_children_info_size() {
    return p_children_info_size;
  }

  bool get_cuda_has_unary() {
    return p_has_unary;
  }


  
private:
  // CUDA AUX DATA ----------
  int p_nb_binary_con;
  int p_children_info_size;
  bool p_has_unary;
  int p_max_ch_set_size;
  // ------------------------
  // The agent context. Includes the variables controlled by the agent, 
  // and the boundary variables of the agent neigbours which are affected
  // by the DCOP protocol adopted.
  // 
  // @note: The notation is WRONG - context means something else.
  // @note: The variables are listed according to the following order:
  //        1. boundary variables
  //        2. private variables
  //        and by increasing variable id.
  std::vector<IntVariable*> p_variables;
  int nb_boundary_vars_;
  int nb_private_vars_;
  
  // The constraints involving the variables controlled by this agent. 
  // It includes the constraints whose scope contains also boundary variables
  // of neighbouring agents.
  // 
  // @note: Constraints are stored in the following order:
  //        1. Local constraints 
  //        2. inter-agent constraints 
  std::vector<Constraint*> constraints_;
  // Number of constraints involving exclusively the agent local variables
  int nb_local_cons_;
  
  // The agent neighbours.
  std::vector<oid_t> neighbours_;

  // This is the strategy used to order the DCOP agents.
  // It contains all elements associated to this node. I.e., if the ordering is
  // a pseudo-tree then it contains information about the agent parent, children,
  // pseudo-parent etc. 
  std::shared_ptr<Ordering> ordering_;

  // u-dcop 
  std::shared_ptr<PseudoTreeOrdering> pt_node;

  // The agent mailbox.
  Mailbox mailbox_;

  // The local statistics of this agent 
  LocalStatistics p_local_statistics;

  // The agent protocol. It defines the DCOP algorithm and the communcation 
  // protocol among agents.
  std::unique_ptr<Algorithm> DCOP_protocol_;

  // The agent's DCOP solution: the assignment to its variable.
  std::vector<std::pair<oid_t, int> > dcop_solution_;

  // The type of the agent solving strategy for its private variables i.e., the
  // type of search engine used to generate its private solution space. 
  InputSettings::solving_t p_private_solver;

  // The type of the agent solving strategy for its boundary variables i.e., the
  // type of search engine used to generate its boundary solution space. 
  InputSettings::solving_t p_boundary_solver;
};

#endif
