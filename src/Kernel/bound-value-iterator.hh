#ifndef ULISSE_KERNEL_DOMAINS_BOUNDVALUEITERATOR_H_
#define ULISSE_KERNEL_DOMAINS_BOUNDVALUEITERATOR_H_

#include "Kernel/globals.hh"
#include "Kernel/value-iterator.hh"

class BoundDomain;

// It defines an iterator over the elements of a BoundDomain.
class BoundValueIterator : public ValueIterator
{
public:
  
  BoundValueIterator();

  // It construct and initializes the bound value iterator by setting minimum 
  // and maximum values of the domain given as a argumrnt. It sets the current 
  // position to default: the leftmost position of the interval. 
  BoundValueIterator(const BoundDomain* dom);

  // It initializes the bound value iterator with the minimum and maximum values
  // of the domain given as a argumrnt. It sets the current position to default:
  // the leftmost position of the interval.
  void initialize(const BoundDomain& dom);

  // It returns true if the iterator has not reach the last
  // element of the domain.
  bool next() const;
  
  // It returns the element currently pointed by the iterator and it
  // advances the iterator to the next position.
  int getAdvance();

  // It signal that a change in the domain it enumerates has been
  // changed. The ValueIterator update its private data 
  // structures and if possible adapt the current iterator to
  // the closest (on the right) which is still in the domain.
  void signalDomainChanges();

  // It sets the current position of the iterator to its inital state. 
  void resetIteration();

  
private:
  // It disables copy constructor and operator= functions.
  DISALLOW_COPY_AND_ASSIGN(BoundValueIterator);

  // current element being pointed
  int current_;

  // The domain being pointed.
  const BoundDomain* p_domain_;

};

#endif // ULISSE_KERNEL_DOMAINS_BOUNDVALUEITERATOR_H_
