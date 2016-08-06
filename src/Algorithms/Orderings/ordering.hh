#ifndef ULYSSES_ALGORITHMS__ORDERINGS__ORDERING_H_
#define ULYSSES_ALGORITHMS__ORDERINGS__ORDERING_H_

#include <string>

class Agent;

// The Abstract class for an Ordering defined over the set of agents.
class Ordering
{
public: 
  
  Ordering(Agent& a);

  virtual ~Ordering();

  // It returns the agent associated with this ordering.
  Agent& agent() const
  {
    return *p_agent;
  }
  
  virtual std::string dump() const = 0;


private:
  // The agent associated with this ordering.
  Agent* p_agent;

};

#endif // ULYSSES_ALGORITHMS__ORDERINGS__ORDERING_H_
