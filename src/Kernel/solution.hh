#ifndef _ULYSSES__SEARCHENGINES__SOLUTION_H_
#define _ULYSSES__SEARCHENGINES__SOLUTION_H_

#include "Kernel/globals.hh"
#include <vector>
#include <string>

// Consider specializing solution for integer and doubles
class Solution 
{
public:
  typedef std::unique_ptr<Solution> uptr;
  typedef std::shared_ptr<Solution> sptr;
  
  // It creates a solution allocating an array of values of size given as a 
  // parameter.
  Solution( int size=0 );

  ~Solution();

  // Generates a solution from the other given as a parameter.
  Solution(const Solution& other);

  // Generates a solution from the other given as a parameter.
  Solution& operator=(const Solution& other); 

  // It returns the value at position pos.
  int operator [](size_t pos) const
  {
    return p_values[ pos ];
  }

  // It returns the value at position pos.
  int& operator[](size_t pos)
  {
    return p_values[ pos ];
  }

  // It allocates the solution data structures.
  void initialize(int size);

  // It resets the solution cost.
  void reset()
  {
    p_cost = 0;
    p_set = false;
  }

  // It returs the solution cost.
  cost_t cost() const
  {
    return p_set ? p_cost : Constants::worstvalue;  
  }

  // It sets the pos-th element to the value given as a parameter.
  void setValue( size_t pos, int value )
  {
    p_values[ pos ] = value;
  }

  // It sets the solution cost to the value given as a parameter.
  void setCost(cost_t val)
  {
    p_cost = val;
    p_set = true;
  }

  // It increases the solution cost by a value given as a parameter.
  void increaseCost( cost_t val )
  {
    p_cost += val;
    p_set = true;
  }

  // It decreases the solution cost by a value given as a parameter.
  void decreaseCost( cost_t val )
  {
    p_cost -= val;
    p_set = true;
  }

  // It returns the set of values stored as a solution.
  std::vector<int>& values()
  {
    return p_values;
  }

  // It returs the size of the solution (number of values).
  size_t size() const
  {
    return p_values.size();
  }

  // It returns a string representation of the solution.
  std::string dump() const;


private:
  // solution state
  std::vector<int> p_values;

  // cost associated to the value assignment
  cost_t p_cost;

  // It mark whether the solution has been set (and its cost updated 
  // for the first time).
  bool p_set;
};


#endif // _ULYSSES__SEARCHENGINES__SOLUTION_H_
