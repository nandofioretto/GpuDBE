#ifndef ULYSSES_UTILITIES__CONSTRAINTS_UTILS_H_
#define ULYSSES_UTILITIES__CONSTRAINTS_UTILS_H_

#include <vector>

#include "Kernel/globals.hh"

class Constraint;

namespace ConstraintUtils
{  
  // It returns the list of constraints ID involving any of the 
  // variables given as a paramter.
  std::vector<oid_t> involvingAny(const std::vector<oid_t> vars);

  // Return the list of contraint IDs which involves exclusively the 
  // variables in the set given as a parmater
  std::vector<oid_t> involvingExclusively
    (const std::vector<oid_t> vars, oid_t agent=-1);

  // Extracts all the variables in the scope of the list of constraints 
  // given as a parameter
  std::vector<oid_t> extractScope(const std::vector<oid_t> constraints);
  
  // Extract the subset of constraints which have owner at least one of the
  // owner in the list given as a parameter
  std::vector<oid_t> extractOwnedByAny
    (const std::vector<oid_t> constraints, const std::vector<oid_t> owners);

  // Extract the subset of constraints whose each variable in their scope
  // is owned by some agent in the owners list given as a parameter
  std::vector<oid_t> extractOwnedByAll
    (const std::vector<oid_t> constraints, const std::vector<oid_t> owners);

  std::vector<oid_t> getID(std::vector<Constraint*> objects);

};


#endif // ULYSSES_UTILITIES__CONSTRAINTS_UTILS_H_
