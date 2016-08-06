#ifndef ULYSSES_UTILITIES__VARIABLE_UTILS_H_
#define ULYSSES_UTILITIES__VARIABLE_UTILS_H_

#include <vector>
#include "Kernel/globals.hh"
#include "Kernel/int-variable.hh"

class IntVariable;

namespace VariableUtils
{
  // Extract all the subset of variables given as parameter which are 
  // owned by the owner given as a parameter
  std::vector<oid_t> extractOwnedBy
    (const std::vector<oid_t> variables, oid_t owner);
  
  static std::vector<oid_t> getID(std::vector<IntVariable*> vars)
  {
    std::vector<oid_t> res; res.reserve(vars.size());
    for (int i=0; i<vars.size(); ++i)
      res.push_back(vars[ i ]->id());
    return res;
  }
};


#endif // ULYSSES_UTILITIES__VARIABLE_UTILS_H_
