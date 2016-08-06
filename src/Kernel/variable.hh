#ifndef ULISSE_KERNEL_VARIABLES_VARIABLE_H_
#define ULISSE_KERNEL_VARIABLES_VARIABLE_H_

#include "Kernel/globals.hh"
#include "Kernel/object-info.hh"
#include "Kernel/domain.hh"
#include "Kernel/constraint.hh"
//class Constraint;

#include <vector>
#include <string>
#include <list>

class Variable : public ObjectInfo
{
public:
  // Sets a variable ID, to a counter which indicates the number
  // of domains created.
  Variable();

  virtual ~Variable() 
  { }

  // It returns the current domain of the variable.
  virtual Domain& domain() const = 0;
  
  // It returns the size of the current domain.
  virtual size_t size() const = 0;

  // It returns true if the domain is empty.
  virtual bool isEmpty() const = 0;

  // It returns true if the domain contains only one value.
  virtual bool isSingleton() const = 0;

  // It sets the agent owner id.
  void registerOwner(size_t aid)
  {
    owner_id_ = aid;
  }
    
  // It returns the agent owner id.
  int ownerId() const
  {
    return owner_id_;
  }

  // It returns all constraints which are associated with variable, even the
  // ones which are already satisfied.
  // The constraints are distinguished by their type.
  size_t nbConstraints(Constraint::ConstraintType type=Constraint::kAny) const;

  // It returns the i-th constraint stored in "constraints_".
  Constraint& constraintAt(int pos) const 
  {
    ASSERT( pos < nbConstraints(), "Requiring invalid constraint position");
    return (*constraints_[ pos ]);
  }
  

  // It registers constraint with current variable, so anytime this variable
  // is changed the constraint is reevaluated.
  //
  // Sorts the constraints so that they appear in the following order:
  // 1. intensional hard constraints
  // 2. intensional soft constraints
  // 3. extensional hard constraints
  // 4. extensional soft constraints
  void registerConstraint( Constraint& c );

  // It returns the constraint
  std::vector<Constraint*>& constraints()
  {
    return constraints_;
  }

  // It returns a description of the variable 
  virtual std::string dump() const = 0;


protected:
  // It disallows the copy constructor and operator=
  DISALLOW_COPY_AND_ASSIGN(Variable);
  
  // It specifies the current weight of the variable.
  int weight_;

  // The ID of the agent owner of this object.
  // it will be used to retrieve the agent address.
  oid_t owner_id_;
  
  // List of the constraints to be checked after the variable is changed.
  std::vector<Constraint*> constraints_;

  // The number of each constraint class.  
  size_t nb_int_hard_c_;
  size_t nb_int_soft_c_;
  size_t nb_ext_hard_c_;
  size_t nb_ext_soft_c_;

};

#endif  // ULISSE_KERNEL_VARIABLES_VARIABLE_H_
