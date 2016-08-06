#include <vector>

#include "Utilities/variable-utils.hh"
#include "Problem/dcop-instance.hh"
  
// Extract all the subset of variables given as parameter which are 
// owned by the owner given as a parameter
static std::vector<oid_t> extractOwnedBy(const std::vector<oid_t> variables, 
                                         oid_t owner)
{
  std::vector<oid_t> res;
  for (oid_t vid : variables)
  {
    if (g_dcop->variable(vid).ownerId() == owner)
      res.push_back( vid );
  }
  return res;
}