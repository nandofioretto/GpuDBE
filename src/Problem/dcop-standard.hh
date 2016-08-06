#ifndef ULYSSES_PROBLEM__DCOP_STANDARD_H_
#define ULYSSES_PROBLEM__DCOP_STANDARD_H_

#include <unordered_map>

#include "Problem/dcop-instance.hh"
#include "Problem/dcop-model.hh"

class InputSettings;

// The DCOP Standard problem encoding.
// This class produces the same DCOP formulation as the one given in the model 
// description.
class DCOPstandard : public DCOPinstance
{
public:
  
  DCOPstandard(DCOPmodel& model, InputSettings& settings);
  
  virtual ~DCOPstandard();
  
  
  virtual std::vector<std::pair<std::string,int> > decodeSolution()
  {
    std::vector<std::pair<std::string, int> > res;
    // for(auto& kv : p_dcop_solution)
    // {
    //   std::string vname = p_map_dcop_var_names[ kv.first ];
    //   res.push_back( std::make_pair( vname, kv.second ));
    // }
    return res;
  }
  
  
  // It solves the DCOP problem
  // void solve();
    
private:
  
  // map std variable id -> std variable name
  std::map<oid_t, std::string> p_map_dcop_var_names;

};

#endif // ULYSSES_PROBLEM__DCOP_STANDARD_H_
