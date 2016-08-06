#ifndef ULYSSES_ALGORITHMS__ORDERINGS__PSEUDO_TREE_ORDERING_H_
#define ULYSSES_ALGORITHMS__ORDERINGS__PSEUDO_TREE_ORDERING_H_

#include "Kernel/globals.hh"
#include "Algorithms/Orderings/ordering.hh"
#include "Kernel/agent.hh"
#include "Utilities/utils.hh"

#include <algorithm>
#include <string>
#include <vector>

// Pseudo-Tree ordering associated to one agent. (a.k.a. pseudo-node).
// This object is constructed by the PseudoTreeAlgorithm, and it is linked
// to the PseudoTreeMsg and the PseudoTreeMsgHandler
class PseudoTreeOrdering : public Ordering
{
public:
  typedef std::unique_ptr<PseudoTreeOrdering> uptr;
  typedef std::shared_ptr<PseudoTreeOrdering> sptr;   
  
  PseudoTreeOrdering(Agent& owner);

  virtual ~PseudoTreeOrdering(); 

  virtual std::string dump() const;

  // Getters
  oid_t parent() const // Deprecated
    { return p_parent; }

  oid_t parentID() const
    { return p_parent; }

  bool isRoot() const // Deprecated
    { return (p_parent == Constants::nullid); }
  
  bool root() const
    { return (p_parent == Constants::nullid); }

  bool isLeaf() const // Deprecated
    { return p_children.empty(); }  

  bool leaf() const
    { return p_children.empty(); }

  oid_t content() const
    { return p_this_node; }

  std::vector<oid_t>& children()
    { return p_children; }

  oid_t nbChildren()
    { return p_children.size(); }
  
  std::vector<oid_t>& pseudoChildren()
    { return p_pseudo_children; }

  oid_t nbPseudoChildren()
    { return p_pseudo_children.size(); }

  std::vector<oid_t>& pseudoParents()
    { return p_pseudo_parents; }

  oid_t nbPseudoParents()
    { return p_pseudo_parents.size(); }

  std::vector<oid_t>& separator()
    { return p_separator; }

  oid_t nbSeparator()
    { return p_separator.size(); }

  std::vector<oid_t>& ancestors()
    { return p_ancestors; }

  oid_t nbAncestors()
    { return p_ancestors.size(); }

  // Setters 
  void setParent(oid_t a_id) 
    { p_parent = a_id; }

  void addChild(oid_t a_id)
  { 
    if (!Utils::find(a_id, p_children))
      p_children.push_back(a_id);
  }

  void removeChild(oid_t a_id)
  {
    auto it = std::find(p_children.begin(), p_children.end(), a_id);
    if (it != p_children.end())
      p_children.erase( it );
  }

  void addPseudoChild(oid_t a_id)
  { 
    if (!Utils::find(a_id, p_pseudo_children))
      p_pseudo_children.push_back(a_id); 
  }

  void removePseudoChild(oid_t a_id)
  {
    auto it = std::find(p_pseudo_children.begin(), p_pseudo_children.end(), a_id );
    if (it != p_pseudo_children.end()) 
      p_pseudo_children.erase( it );
  }

  void addPseudoParent(oid_t a_id)
  {
    if (!Utils::find(a_id, p_pseudo_parents))
      p_pseudo_parents.push_back(a_id); 
  }

  void removePseudoParent(oid_t a_id)
  {
    auto it = std::find(p_pseudo_parents.begin(), p_pseudo_parents.end(), a_id );
    if (it != p_pseudo_parents.end())
      p_pseudo_parents.erase( it );
  }

  // Forces parent to always be the first element of separator
  void addSeparator(oid_t a_id)
  {
    if( a_id != p_this_node and !Utils::find(a_id, p_separator))
      p_separator.push_back(a_id);

    int i = Utils::findIdx(p_separator, p_parent);
    if( i > 0 )
      std::swap(p_separator[i], p_separator[0]);
  }

  void addAncestor(oid_t a_id)
  { 
    if (!Utils::find(a_id,  p_ancestors))
      p_ancestors.push_back(a_id); 
  }
  
  bool constructed() {
    return p_constructed;
  }
  
  void setConstructed(bool p=true) {
    p_constructed = p;
  }

  void reset() {
    p_constructed = false;
    p_parent = Constants::nullid;
    p_children.clear();
    p_pseudo_children.clear();
    p_pseudo_parents.clear();
    p_separator.clear();
    p_ancestors.clear();
  }

private:
  bool p_constructed;
  
  oid_t p_this_node;

  oid_t p_parent;
  
  // The set of descendant nodes
  std::vector<oid_t> p_children;

  // The set of descendant nodes which are connected with this node through a 
  // back edge.
  std::vector<oid_t> p_pseudo_children;

  // The set of ancestor nodes which are connected with this node through a 
  // back edge.
  std::vector<oid_t> p_pseudo_parents;

  // The separator set of this agent: all ancestor nodes which are connected 
  // with this node (through any edge)  or which are connected with its 
  // descendants.
  std::vector<oid_t> p_separator;

  // These are the agent that need to be traversed in a path from the 
  // current node to the root node, traversing only tree-edges. 
  std::vector<oid_t> p_ancestors;

};

#endif // ULYSSES_ALGORITHMS__ORDERINGS__PSEUDO_TREE_ORDERING_H_
