#ifndef ULISSE_KERNEL_DOMAINS_DOMAIN_H_
#define ULISSE_KERNEL_DOMAINS_DOMAIN_H_

#include "Kernel/object-info.hh"
#include "Kernel/globals.hh"

//#include <memory>
#include <vector>
#include <string>

class ValueIterator;

// TODO: function copyMinimalChanges - or copy minimal information
// TODO: function restore previous changes.

// Defines a Domain and the related operations on it.
// The Domain is designed as an abstract object.
class Domain : public ObjectInfo
{
 public:
  
  enum EventType{
    kNone = 0,			// no event is triggered
    kNotChanged,		// domain not changed by last action
    kSingleton,			// domain contains a single element 
    kBoundChanged,	// bounds changed by last action
    kDomainChanged,	// element changed by last action
    kAny,			      // any of the above, excluding none
    kFailed,			  // domain is empty
    kNbEvents
  };
  
  Domain() 
  { }
  
  // It creates a copy of this object by cloning its state. Only the 
  // minimal information needed to reproduce this element is  copied. 
  // It is used to save and restore the domain state during backtrack. 
  virtual Domain* clone() const = 0;

  virtual ~Domain() 
  { }
  
  // It removes all elements.
  virtual void clear() = 0;
 
  // It returns true if the domain is empty.
  virtual bool isEmpty() const = 0;

  // It returns true if the domain has only one element.
  virtual bool isSingleton() const = 0;

  // It specifies wether the domain type is more suited to a sparse 
  // representation of values.
  virtual bool isSparse() const = 0;

  // It specifies if domain is a finite domain of numeric values (integers).
  virtual bool isNumeric() const  = 0;
    
  // It returns the size of the domain.
  virtual size_t size() const = 0;

  // It specifies what events should be executed when a given event occurs.
  virtual std::vector<EventType> subsumedEvents(EventType pruning_event) = 0;

  // It returns the value iterator of the domain values.
  // The value iterator iterates over the elements of this domain. It is
  // initialized so to point to the first element.
  virtual ValueIterator& valueIterator() = 0;
  
  // It returns the domain description.
  virtual std::string dump() const = 0;  

  static bool changed(EventType event)
  {
    return (event < 2 );
  }

protected:
  // Avoids calls to copy constructor and assign operator.
  DISALLOW_COPY_AND_ASSIGN(Domain);
};

#endif // ULISSE_KERNEL_DOMAINS_DOMAIN_H_
