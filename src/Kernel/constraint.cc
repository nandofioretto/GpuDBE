#include <vector>

#include "Kernel/constraint.hh"
#include "Kernel/int-variable.hh"

using namespace std;

Constraint::Constraint() 
{ }


Constraint::~Constraint() 
{ }


void Constraint::updateScope(vector<IntVariable*> newvars)
{
  ASSERT(newvars.size() == scope_.size(), "Error in updating constraint scope");

  for(int i=0; i<newvars.size(); ++i)
  {
    scope_[i] = newvars[i];
    scope_ids_[i] = newvars[i]->id();
  } 
}
