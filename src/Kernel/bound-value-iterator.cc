#include "Kernel/globals.hh"
#include "Kernel/value-iterator.hh"
#include "Kernel/bound-domain.hh"

BoundValueIterator::BoundValueIterator()
  : p_domain_(nullptr), current_(-1)
{ }


BoundValueIterator::BoundValueIterator(const BoundDomain* dom)
  : p_domain_(dom), current_(dom->min() - 1)
{ }

  
void BoundValueIterator::initialize(const BoundDomain& dom)
{
  p_domain_ = &dom;
  current_ = p_domain_->min() - 1;
}


bool BoundValueIterator::next() const 
{ 
  return ( current_ < p_domain_->max() ); 
}  


int BoundValueIterator::getAdvance() 
{ 
  ASSERT( current_ < p_domain_->max(), "iterator out of bounds");
  return ++current_; 
}


void BoundValueIterator::signalDomainChanges() 
{
  if( current_ < p_domain_->min() - 1 )
    current_ = p_domain_->min() - 1;
}


void BoundValueIterator::resetIteration()
{
  current_ = p_domain_->min() - 1;
}
