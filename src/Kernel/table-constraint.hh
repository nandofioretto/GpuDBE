#ifndef ULYSSES_KERNEL__CONSTRAINTS__PROB_CONSTRAINT_H_
#define ULYSSES_KERNEL__CONSTRAINTS__PROB_CONSTRAINT_H_

#include <map>
#include <vector>

#include "Kernel/globals.hh"
#include "Kernel/constraint.hh"
#include "Utilities/utils.hh"

class IntVariable;

// It defines the Abstract class for Extensional Soft Constraints.
// TODO: The aim is to implement it as a TABLE CONSTRAINT
class TableConstraint : public Constraint 
{
 public:

  // It sets the constraint type, scope, number of tuples contained in 
  // the explicit relation and default cost.
  TableConstraint
  (std::vector<IntVariable*> scope, cost_t def_cost)
 	 : m_default_cost(def_cost), m_best_finite_cost(Constants::worstvalue), m_worst_finite_cost(Constants::bestvalue)
 {
    scope_ = scope; 
    type_  = kExtSoft;
    for(IntVariable* v : scope)
      scope_ids_.push_back(v->id());
  }

  ~TableConstraint() { };
  
  void setCost(std::vector<int> K, cost_t cost)
  {
	  m_values[ K ] = cost;
  }

  // It returns the cost associated to the value assignemnt passed as a 
  // parameter.
  cost_t getCost(std::vector<int> scope)
  {
	auto it = m_values.find(scope);
    return (it != m_values.end()) ? it->second : m_default_cost;
  }


  cost_t defaultCost() const
  { 
    return m_default_cost;
  }

  size_t sizeBytes() const
  {
    return m_values.size() * (arity()+1) * sizeof( size_t );
  }

  // It returns a Summary Description.
  virtual std::string dump() {
    std::string result = std::to_string(ObjectInfo::id())
      + " " + ObjectInfo::name() + "( ";  
    for( int i=0; i<arity(); i++ )
      result += scope_[ i ]->name() + ", ";
    result += ") ";

    for (auto& kv: m_values) {
      result += "<"; 
      for (int i : kv.first)
    	  result += std::to_string(i) + " ";
      result += "> : ";
   	  result += std::to_string(kv.second) + " ";
      result += "\n";
    }

    return result;
  }
  
 protected:
  // It disallows the copy constructor and operator=
  DISALLOW_COPY_AND_ASSIGN(TableConstraint);

  std::map<std::vector<int>, cost_t> m_values;

  // Default cost, that is the cost associated to any value combination that is
  // not explicitally listed in the relation.
  cost_t m_default_cost;

  // The best and worst finite costs of this constraint which is used as bounds
  // in some searches.
  cost_t m_best_finite_cost;
  cost_t m_worst_finite_cost;

};



#endif // ULYSSES_KERNEL__CONSTRAINTS__EXT_HARD_CONSTRAINT_H_ 
