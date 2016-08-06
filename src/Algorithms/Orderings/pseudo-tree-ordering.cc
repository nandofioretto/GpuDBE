#include <string>

#include "Algorithms/Orderings/pseudo-tree-ordering.hh"
#include "Kernel/globals.hh"
#include "Kernel/agent.hh"
#include "Kernel/agent-factory.hh"
#include "Problem/dcop-instance.hh"

PseudoTreeOrdering::PseudoTreeOrdering(Agent& owner)
  : Ordering(owner), p_this_node(owner.id()), p_parent(Constants::nullid), p_constructed(false)
{ }


PseudoTreeOrdering::~PseudoTreeOrdering() 
{ }


std::string PseudoTreeOrdering::dump() const 
{
  std::string result = "PseudoNode: " + g_dcop->agent(p_this_node).name();
  if( isLeaf() ) result += " LEAF ";
  if( isRoot() ) result += " ROOT ";
  result += "\n  Parent: [";
  if (p_parent != Constants::nullid)
    result += g_dcop->agent(p_parent).name();
  result += "]\n";
  result += "  Children: {";
  for (oid_t a : p_children)
    result += g_dcop->agent(a).name() + " ";
  result += "}\n";
  result += "  Pseudo-Chl: {";
  for (oid_t a : p_pseudo_children)
    result += g_dcop->agent(a).name() + " ";
  result += "}\n";
  result += "  Pseudo-Par: {";
  for (oid_t a : p_pseudo_parents)
    result += g_dcop->agent(a).name() + " ";
  result += "}\n";
  result += "  Separator: {";
  for (oid_t a : p_separator)
    result += g_dcop->agent(a).name() + " ";
  result += "}\n";
  result += "  Ancestor: {";
  for (oid_t a : p_ancestors)
    result += g_dcop->agent(a).name() + " ";
  result += "}\n";
  
  return result;
}

