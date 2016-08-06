#ifndef ULISSE_KERNEL_DOMAINS_BOUNDDOMAIN_H_
#define ULISSE_KERNEL_DOMAINS_BOUNDDOMAIN_H_

#include "Kernel/globals.hh"
#include "Kernel/int-domain.hh"
#include "Kernel/bound-value-iterator.hh"

#include <string>
#include <vector>

class ValueIterator;

// Defines an Integer Bound domain and the related operations on it.
// The BoundDomain is an implementation of the IntDomain object.
class BoundDomain : public IntDomain
{
  public:
  // Creates an empty domain, which is a domain with minimum > maximum.
  BoundDomain();

  // Creates an instance of BoundDomain.
  // @note It must have min <= max
  BoundDomain(int min, int max);

  // It creates a copy of this object by cloning its state. Only the 
  // minimal information needed to reproduce this element is  copied. 
  // It is used to save and restore the domain state during backtrack. 
  virtual BoundDomain* clone() const;

  // It copies the other domain passed as a parameter into this object.
  // The other domain is expected to contain only minimial information 
  // about the domain values.
  // It is used to restore the domain state during backtrack.
  virtual void copy(IntDomain* other);

  virtual ~BoundDomain();

  // Returns true if the current domain is equal to the one given as a 
  // paramenter.
  virtual bool operator==(const IntDomain& other);

  // It returns the idx-th element of the domain.
  virtual int operator[](int idx) const
  { 
    return min_ + idx;
  }
  
  // It returns the idx-th element of the domain.
  virtual int elementAt(int idx) const
  {
    return min_ + idx; 
  }

  // It removes all elements.
  virtual void clear()
  { 
    min_ = 1; 
    max_ = 0; 
  }

  // It returns true if the domain is empty.
  virtual bool isEmpty() const
  { 
    return min_ > max_; 
  }

  // It returns true if the domain has only one element.
  virtual bool isSingleton() const
  {
    return min_ == max_; 
  }

  // It returns true if the domain has only one element, and 
  // if it is equal to the value given as a parameter.
  virtual bool isSingleton(int c) const 
  {
    return (min_ == c and max_ == c); 
  }

  // It specifies wether the domain type is more suited to a sparse 
  // representation of values.
  virtual bool isSparse() const
  {
    return false; 
  }

  // It specifies if domain is a finite domain of numeric values (integers).
  virtual bool isNumeric() const 
  { 
    return true; 
  }
    
  // It returns the size of the domain.
  virtual size_t size() const 
  {
    return (max_ - min_ + 1); 
  }

  // It returns the previous value in the domain w.r.t. the given 'value'.
  // The next relation is intended as the lexicographic order.
  // If no value can be found then it returns the same value. 
  virtual int previousValue(int value) const
  {
    if (value > min_) return value-1;
    return value;
  }

  // It returns the next value in the domain w.r.t. the given 'value'.
  // The next relation is intended as the lexicographic order.
  // If no value can be found then it returns the same value. 
  virtual int nextValue(int value) const
  {
    if (value < min_) return min_;
    if (value < max_) return value + 1;
    return value;
  }

  // It returns the value iterator of the domain values.
  // The value iterator iterates over the elements of this domain. It is
  // initialized so to point to the first element.
  virtual ValueIterator& valueIterator() 
  { 
    value_iterator_->resetIteration();
    return *value_iterator_;
  }

  // It returns the smaller domain value.
  virtual int min() const
  {
    return min_; 
  }
  
  // It returns the largest domain value.
  virtual int max() const
  { 
    return max_; 
  }

  // It checks whether the value belongs to the domain.
  virtual bool contains(int value)
  { 
    return (min_ <= value and max_ >= value); 
  }

  // It checks whether the interval min..max belongs to current the domain.
  virtual bool contains(int min, int max) 
  { 
    return (min_ <= min and max_ >= max); 
  }

  // It updates the domain by setting as new minimum value, the value
  // given as a parameter.
  // @note: no checks are done here.
  virtual void setMin(int min)
  {
    min_ = min;
  }

  // It updates the domain by setting as new maximum value, the value
  // given as a parameter.
  // @note: no checks are done here.
  virtual void setMax(int max) 
  { 
    max_ = max; 
  }

  // It returns the domain description.
  virtual std::string dump() const;


protected:
  // Avoids calls to copy constructor and assign operator.
  DISALLOW_COPY_AND_ASSIGN(BoundDomain);

  // The minimal value of the domain.
  int min_;
  
  // The maximal value of the domain.
  int max_;

  // The bound interval iterator, to iterate through all the elements 
  // of the bound domain.
  BoundValueIterator* value_iterator_;

};


#endif // ULISSE_KERNEL_DOMAINS_BOUNDDOMAIN_H_
