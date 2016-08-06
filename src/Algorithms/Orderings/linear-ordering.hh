#ifndef ULYSSES_ALGORITHMS__ORDERINGS__LINEAR_ORDERING_H_
#define ULYSSES_ALGORITHMS__ORDERINGS__LINEAR_ORDERING_H_

#include "Kernel/globals.hh"
#include "Algorithms/Orderings/ordering.hh"
#include "Kernel/agent.hh"
#include "Kernel/agent-factory.hh"

#include "Problem/dcop-instance.hh"

// It constructs a linear ordering for the DCOP agents.
// 
// The linar ordering is based on the ID if the agents, and it does not follow
// the constraint graph. 
class LinearOrdering : public Ordering
{
public:
  typedef std::unique_ptr<LinearOrdering> uptr;
  typedef std::shared_ptr<LinearOrdering> sptr;  

  // Constructs a linear ordering of the DCOP agents based
  // on a topological sort.
  LinearOrdering(Agent& a);
  
  virtual ~LinearOrdering() { }
   
  // Get the next agent in the linear order.
  Agent* successor() const
  {
    if (tail()) 
      return nullptr; 
    return &(g_dcop->agent(agent().id() + 1));
  }

  // Get the previous agent in the linear order.
  Agent* predecessor() const
  {
    if (head()) 
      return nullptr;
    return &(g_dcop->agent(agent().id() - 1));
  }

  // Returns true if the agent associated to this ordering is the one with one
  // with the greatest ID.
  bool tail() const
  {
    return (agent().id() == (g_dcop->nbAgents()-1));
  }

  // Returns true if the agent associated to this ordering is the one with one
  // with the smallest ID.
  bool head() const
  {
    return (agent().id() == 0);
  }

  // Returns the position of the current agent in the linear ordering. 
  int position() const
  {
    return agent().id();
  }

  // It returns a description of the agent in the linear ordering.
  virtual std::string dump() const;

};


#endif // ULYSSES_ALGORITHMS__ORDERINGS__LINEAR_ORDERING_H_
