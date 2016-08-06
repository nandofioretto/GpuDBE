#ifndef ULISSE_KERNEL_DOMAIN_VALUEITERATOR_H_
#define ULISSE_KERNEL_DOMAIN_VALUEITERATOR_H_

#include "Kernel/globals.hh"

// It defines an iterator over the elements of a domain.
class ValueIterator
{
public:
  ValueIterator() { }
  
  // It returns true if the iterator has not reach the last
  // element of the domain.
  virtual bool next() const = 0;
  
  // It returns the element currently pointed by the iterator and it
  // advances the iterator to the next position.
  virtual int getAdvance() = 0;

  // It signal that a change in the domain it enumerates has been
  // changed. The ValueIterator update its private data 
  // structures and if possible adapt the current iterator to
  // the closest (on the right) which is still in the domain.
  virtual void signalDomainChanges() = 0;
  
  // It sets the current position of the iterator to its inital state. 
  virtual void resetIteration() = 0;

private:
  // It disables copy constructor and operator= functions.
  //DISALLOW_COPY_AND_ASSIGN(ValueIterator);

};

#endif // ULISSE_KERNEL_DOMAIN_VALUEITERATOR_H_
