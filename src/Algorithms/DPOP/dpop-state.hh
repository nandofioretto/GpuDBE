#ifndef ULYSSES_ALGORITHMS_DPOP_STATE_H__H_
#define ULYSSES_ALGORITHMS_DPOP_STATE_H__H_

#include "Kernel/globals.hh"
#include "Kernel/codec.hh"

#include <memory>
#include <map>

class UtilMsgHandler;
class TableConstraint;
class Agent;
class IntVariable;

class DPOPstate // public AlgorithmState
{
public:
  typedef std::shared_ptr<DPOPstate> sptr;
  
  DPOPstate() :
    ai(nullptr), xi(nullptr), hostUtilTable(nullptr) 
  { }

  ~DPOPstate() 
  {
    if(hostUtilTable != nullptr)
      delete[] hostUtilTable;
  }

  void initialize(Agent& agent, std::shared_ptr<UtilMsgHandler> h);

  void reset(Agent& agent);
  
  void updateState();

  void updateRootState();

  // Phase 1
  cost_t get_util(int row)
  {
    return p_util_table[ row ];
  }

  // Phase 1
  int get_xi_value(int row)
  {
    return p_xi_assignment[row];
  }

  // Phase 2
  void set_sep_value(int val, oid_t vid)
  {
    p_sep_values[ vid ] = val;
  }

  // Phase 2
  int get_sep_value(oid_t vid)
  {
    return p_sep_values[ vid ];
  }

  // Get the best value for this variable
  int get_xi_best_value();


  IntVariable& get_xi() 
  {
    return *xi;
  }

private:
  void solveUnaryConstraint(Agent& agent, std::vector<oid_t> c_unary);

  int* hostUtilTable;
  size_t nbRowsUT;
  
  // for all combination of values of variables in the 
  // agent separator set contains the list of utilities
  // one for each belief state.
  util_table_t p_util_table;
  std::shared_ptr<Codec> p_util_table_rows;

  // Holds the assignment for variable xi after projection, for each
  // combo of the vars in sep(xi)
  std::vector< int > p_xi_assignment;

  // The unary constraint on the variable of this agent
  std::vector<cost_t > p_unary;
  
  std::vector<TableConstraint*> p_constraints;

  // Values retured after the first optimization (VALUE PHASE 1)
  // For each variable of the separator set we have #nb world values
  std::map<oid_t, int > p_sep_values;
    
  Agent* ai;
  IntVariable* xi;

  // The message handler associated to this algorithm
  std::shared_ptr<UtilMsgHandler> p_util_msg_handler;
};

#endif
