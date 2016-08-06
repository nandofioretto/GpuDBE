#ifndef ULISSE_KERNEL_DOMAINS_INTDOMAIN_H_
#define ULISSE_KERNEL_DOMAINS_INTDOMAIN_H_

#include "Kernel/globals.hh"
#include "Kernel/domain.hh"

#include <vector>
#include <map>
#include <memory>

class ValueIterator;

// Defines an Integer domain and the related operations on it.
// The IntDomain is designed as an abstract object.
class IntDomain : public Domain
{
  public:

  enum IntDomainType{
    kInterval = 0,
    kBound,
    kDense,
    kUndefined,
    kNbIntDomains
  };

  // Initializes the events which are subsumed by the events: 'singleton',
  // 'bound' and 'any'.
  IntDomain();

  // It creates a copy of this object by cloning its state. Only the 
  // minimal information needed to reproduce this element is  copied. 
  // It is used to save the domain state in a backtrack based search. 
  virtual IntDomain* clone() const = 0;

  // It copies the other domain passed as a parameter into this object.
  // The other domain is expected to contain only minimial information 
  // about the domain values.
  // It is used to restore the domain state during backtrack.
  virtual void copy(IntDomain* other) = 0;

  virtual ~IntDomain() 
  { }

  // It returns true if the current domain is equal to the one given as a 
  // paramenter.
  virtual bool operator==(IntDomain& other);

  // It returns the idx-th element of the domain.
  virtual int operator[](int idx) const = 0;

  // It returns the idx-th element of the domain.
  virtual int elementAt(int idx) const = 0;

  // It removes all elements.
  virtual void clear() = 0;
 
  // It returns true if the domain is empty.
  virtual bool isEmpty() const = 0;

  // It returns true if the domain has only one element.
  virtual bool isSingleton() const = 0;

  // It returns true if the given domain has a single value, equal to c.
  virtual bool isSingleton(int c) const
  { 
    return (min() == c and size() == 1); 
  }

  // It specifies wether the domain type is more suited to a sparse 
  // representation of values.
  virtual bool isSparse() const = 0;

  // It specifies if domain is a finite domain of numeric values (integers).
  virtual bool isNumeric() const  = 0;
    
  // It returns the size of the domain.
  virtual size_t size() const = 0;

  // It returns the previous value in the domain w.r.t. the given 'value'.
  // The next relation is intended as the lexicographic order.
  // If no value can be found then it returns the same value. 
  virtual int previousValue(int value) const = 0;

  // It returns the next value in the domain w.r.t. the given 'value'.
  // The next relation is intended as the lexicographic order.
  // If no value can be found then it returns the same value. 
  virtual int nextValue(int value) const = 0;

  // It returns the value iterator of the domain values.
  // The value iterator iterates over the elements of this domain. It is
  // initialized so to point to the first element.
  virtual ValueIterator& valueIterator() = 0;

  // It returns the smaller domain value.
  virtual int min() const = 0;
  
  // It returns the largest domain value.
  virtual int max() const = 0;

  // It returns the type of this domain.
  virtual IntDomainType type() const
  { 
    return domain_type_;
  }

  // It constructs and returns the the array containing all elements
  // int the domain.
  virtual std::vector<int> content();

  // It checks whether the value belongs to the domain.
  virtual bool contains(int value)
  { 
    return contains(value,value); 
  }

  // It checks whether the interval min..max belongs to current the domain.
  virtual bool contains(int min, int max) = 0;

  // It returns the value of the domain, when it is assigned.
  // NOTE: The domain must be singleton at the time of the function call.
  virtual int value() const
  {
    ASSERT( isSingleton(), " domain is not assigned.");
    return min();
  }
  
  // Returns the lexicogrpahic ordering between the current domain
  // and the one given as a parameter.
  // @param The domain (set) to be lexically compared to current domain (set)
  // @return -1 if other is greather than this one
  //          0 if they are equal
  //          1 otherwise
  virtual int lex(IntDomain& other);
  
  // It returns the list of events that should be executed when a given event 
  // occurs.
  virtual std::vector<EventType> subsumedEvents(EventType pruning_event)
  { 
    return subsumed_events_[ pruning_event ]; 
  }

  // It updates the domain by setting as new minimum value, the value
  // given as a parameter.
  virtual void setMin(int min) = 0;

  // It updates the domain by setting as new maximum value, the value
  // given as a parameter.
  virtual void setMax(int max) = 0;

  // It returns the domain description.
  virtual std::string dump() const = 0;
  

protected:

  static std::map<EventType, std::vector<EventType> > subsumed_events_;
  
  IntDomainType domain_type_;

};


#endif // ULISSE_KERNEL_DOMAINS_DOMAIN_H_
