#ifndef ULISSE_KENERL_VARIABLES_INTVARIABLE_H_
#define ULISSE_KENERL_VARIABLES_INTVARIABLE_H_

#include "Kernel/globals.hh"
#include "Kernel/variable.hh"
#include "Kernel/int-domain.hh"

#include <string>

// TODO: domain changes management

// Defines a Finite Domain Variable (FDV) and related operations on it.
class IntVariable : public Variable
{
public:
 
  // Craete an empty variable.
  IntVariable();
	 
  virtual ~IntVariable();

  // Initializes the variable domain.
  virtual void setDomain(IntDomain* dom);

  // Initializes the variable domain.
  virtual void setDomain(int min, int max);

  // It checks if the domains of this variable is equal to the domain of 
  // the variable given as a argument.
  virtual bool operator==(const IntVariable& other) const
  {
    return *domain_ == other.domain();
  }

  // This function returns current value in the domain of the variable. If
  // current domain of variable is not singleton then warning is printed and
  // minimal value is returned.
  int value() const 
  {
    //ASSERT(domain_->isSingleton(), "Request for a value of not assigned variable.");
    return domain_->min();
  }
	
  // It returns the current maximal value in the domain.
  int max() const 
  {
    return domain_->max();
  }

  // It returns the current maximal value in the domain.
  int min() const 
  {
    return domain_->min();
  }

  void resetLastChoice()
  {
    last_choice_ = (min() - 1);
  }

  void setLastChoice(int val)
  {
    last_choice_ = val;
  }

  // It returns the last choice made during search. Note that this value
  // is not restored.
  int lastChoice() const 
  {
    return last_choice_;
  }

  // It returns the variable domain.
  virtual IntDomain& domain() const
  {
    return *domain_;
  }

  // It returns the number of elements contained in the current domain.
  virtual size_t size() const
  {
    return domain_->size();
  }

  // It checks if the domain is empty.
  virtual bool isEmpty() const 
  {
    return domain_->isEmpty();
  }

  // It returns true if the domain contains only one value.
  virtual bool isSingleton() const
  { 
    return domain_->isSingleton(); 
  }

  // It returns true if the domain contains only one value equal to c.
  virtual bool isSingleton(int val) 
  {
    return domain_->isSingleton( val );
  }

  virtual std::string dump() const; 

  // Returns the size of the variable's domain in bytes.
  virtual size_t sizeBytes() const
  {
    return domain_->size() * sizeof(int);
  }

private:
  // A the current domain of the variable/.
  IntDomain* domain_;
  
  // The last value choosen in a labeling step. It is used to resume a search.
  int last_choice_;
};


#endif  // ULISSE_KENERL_VARIABLES_INTVARIABLE_H_
