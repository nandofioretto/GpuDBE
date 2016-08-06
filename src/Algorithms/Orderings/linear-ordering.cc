#include "Algorithms/Orderings/linear-ordering.hh"

#include <string>

LinearOrdering::LinearOrdering(Agent& a) 
  : Ordering(a)
{ }


std::string LinearOrdering::dump() const
{
  std::string result;
  if (predecessor())
    result += predecessor()->name() + " <-- :";
  else result += "[ :";
  result += agent().name();
  if (successor())
    result += ": --> " + successor()->name();
  else result += ": ]";
  return result;
}
