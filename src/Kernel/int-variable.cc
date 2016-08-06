#include "Kernel/int-variable.hh"
#include "Kernel/int-domain.hh"
#include "Kernel/bound-domain.hh"

IntVariable::IntVariable()
  : domain_(nullptr), last_choice_(0)
{ }


void IntVariable::setDomain(int min, int max)
{
  // if( max - min > 63 )
  last_choice_ = (min-1);
  setDomain( new BoundDomain(min, max) );
  //   setDomain(new IntervalDomain(min, max));
  // else
  //   setDomain(new SmallDenseDomain(min, max))
}


IntVariable::~IntVariable()
{
  if( domain_ ) delete domain_;
}


void IntVariable::setDomain( IntDomain* dom )
{
  domain_ = dom;
  last_choice_ = dom->min() - 1;
}


std::string IntVariable::dump() const
{
  std::string result = ObjectInfo::dump();
  result += " owner(" + std::to_string(owner_id_) + ")";

  if (domain_->isSingleton())
    result += " = ";
  else
    result += "::";
  		
  result += domain_->dump();
  return result;
}
