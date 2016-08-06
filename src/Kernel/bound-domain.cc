#include "Kernel/globals.hh"
#include "Kernel/bound-domain.hh"


BoundDomain::BoundDomain()
  : value_iterator_(nullptr), min_(0), max_(-1)
{ 
  domain_type_ = IntDomain::kBound;
}


BoundDomain::BoundDomain(int min, int max)
  : min_(min), max_(max)
{
  ASSERT( min <= max, "Creating Instance of BoundDomain with min: " << min 
	  << " > max: " << max);

  domain_type_ = IntDomain::kBound;
  value_iterator_ = new BoundValueIterator(this);
}


// Note: this is a protected copy constructor - it can only called 
// by this object. It copies the minimal information needed for the 
// clone function. 
BoundDomain::BoundDomain(const BoundDomain& other)
  : value_iterator_(nullptr)  
{
  domain_type_ = IntDomain::kBound;
  min_ = other.min_;
  max_ = other.max_;
}


BoundDomain* BoundDomain::clone() const
{
  return new BoundDomain(*this);
}


void BoundDomain::copy(IntDomain* other)
{
  ASSERT(other->type() == IntDomain::kBound, 
	 "Trying to copy an incompatible domain type: " 
	 << other->type());
  min_ = (dynamic_cast<BoundDomain*>(other))->min();
  max_ = (dynamic_cast<BoundDomain*>(other))->max();
}


BoundDomain::~BoundDomain()
{
  if( value_iterator_ ) 
    delete value_iterator_;
}


bool BoundDomain::operator==(const IntDomain& other)
{
  if (other.isEmpty() and this->isEmpty()) 
    return true;
    
  if (min_ == other.min() and max_ == other.max() and
      (max_ - min_ + 1) == other.size())
    return true;
  
  return false;
}


std::string BoundDomain::dump() const
{
  if( isEmpty() ) return "{}";

  std::string result="BD: ";
  result += std::to_string(min_);
  if (max_ != min_)
    result += ".." + std::to_string(max_);
  return result;
}
