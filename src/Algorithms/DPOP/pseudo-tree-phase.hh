#ifndef ULYSSES_ALGORITHMS__PSEUDO_TREE__PSEUDO_TREE_CONSTRUCTION_H_
#define ULYSSES_ALGORITHMS__PSEUDO_TREE__PSEUDO_TREE_CONSTRUCTION_H_

#include "Kernel/globals.hh"
#include "Kernel/agent.hh"

#include "Problem/dcop-instance.hh"
#include <memory>

typedef PseudoTreeOrdering PseudoNode;

// It implements the distributed pseudo-tree construction.
class PseudoTreeConstruction : public Algorithm
{
public:
  typedef std::unique_ptr<PseudoTreeConstruction> uptr;
  typedef std::shared_ptr<PseudoTreeConstruction> sptr;  
  
  // Initializes the flag marked and the pseudo-tree node 
  // which this algorithm will construct within a pseudo-tree.
  PseudoTreeConstruction(Agent& owner);

  virtual ~PseudoTreeConstruction();
  
 // It initializes the algorithm.
  virtual void initialize();

 // It initializes the algorithm.
  virtual void finalize();

  // It returns true if the algorithm can be executed in this agent.
  virtual bool canRun();

  // It executes the algorithm.
  virtual void run();

  // It stops the algorithm saving the current results  and states if provided
  // by the algorithm itself.
  virtual void stop()
  { }

  // It returns whether the algorithm has terminated.
  virtual bool terminated()
  {
    return p_status == k_terminated;
  }

  std::vector<oid_t> get_separator_set(PseudoNode& n);

  void set_elected_root(int pId) { p_elected_root = pId; }

  void set_heuristic(int heur) { p_heuristic = heur; }

private:
  // The Agent Id of the elected agent root.
  int p_elected_root;
  int p_heuristic;

  void run_construction(int heur);
  int get_induced_width();

};

#endif // ULYSSES_ALGORITHMS__PSEUDO_TREE__PSEUDO_TREE_H_

