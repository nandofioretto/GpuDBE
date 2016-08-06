#ifndef ULYSSES_KERNEL__CONSTRAINTS__CONSTRAINT_H_
#define ULYSSES_KERNEL__CONSTRAINTS__CONSTRAINT_H_

#include <vector>
#include <string>

#include "Kernel/globals.hh"
#include "Kernel/object-info.hh"

class IntVariable;

// Standard interface class for all constraints.
//
// Defines how to construct a constraint, impose, check satisfiability,
// notSatisfiability, enforce consistency.
class Constraint : public ObjectInfo
{
public:

  enum ConstraintType {
    kIntHard=0,
    kIntSoft,
    kExtHard,
    kExtSoft,
//    kProb,
    kAny
  };

  // It creates a new constraint with empty name and automatically generated ID.
  Constraint();

  virtual ~Constraint();
  
  // Returns the constraint arity.
  int arity() const
  { 
    return scope_.size();
  }

  // Returns the constraint scope.
  std::vector< IntVariable* >& scope()
  {
    return scope_;
  }

  std::vector< oid_t >& scopeIds()
  {
    return scope_ids_;
  }

  // Get the pos-th variable in the constraint scope.
  IntVariable& variableAt(size_t pos) const
  { 
    return *scope_[ pos ]; 
  }

  oid_t variableIdAt(size_t pos) const
  {
    return scope_ids_[pos]; 
  }

  void updateScope(std::vector<IntVariable*> newvars);

  // It returns the constraint type.
  ConstraintType type() const
  {
    return type_;
  }

  // It returns a Summary Description.
  virtual std::string dump() = 0;

  // It returns the size of the constraint in bytes.
  virtual size_t sizeBytes() {return 0; }

protected:
  // Avoids calls to copy constructor and assign operator.
  DISALLOW_COPY_AND_ASSIGN(Constraint);

  // The constraint type
  ConstraintType type_;
  
  // The scope of the constraint
  std::vector< IntVariable* > scope_;
  std::vector< oid_t > scope_ids_;
};


#endif // ULYSSES_KERNEL__CONSTRAINTS__CONSTRAINT_H_
