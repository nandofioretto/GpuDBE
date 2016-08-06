#include "Kernel/globals.hh"
#include "Kernel/agent.hh"
#include "Communication/scheduler.hh"
#include "Algorithms/DPOP/pseudo-tree-phase.hh"
#include "Algorithms/Orderings/pseudo-tree-ordering.hh"
#include "Utilities/utils.hh"
#include "preferences.hh"

#include <algorithm>
#include <stack>
#include <string>

using namespace std;

bool order_des(oid_t LHS, oid_t RHS) { return LHS > RHS; }
bool order_asc(oid_t LHS, oid_t RHS) { return LHS < RHS; }

bool lex_asc(oid_t LHS, oid_t RHS) {
	std::string strL = std::to_string(LHS);
	std::string strR = std::to_string(RHS);
	return //std::to_string(LHS).compare(std::to_string(RHS));
			std::lexicographical_compare(strL.begin(), strL.end(), strR.begin(), strR.end());
}

bool lex_des(oid_t LHS, oid_t RHS) {
	std::string strL = std::to_string(LHS);
	std::string strR = std::to_string(RHS);
	return //std::to_string(RHS).compare(std::to_string(LHS));
			std::lexicographical_compare(strR.begin(), strR.end(), strL.begin(), strL.end());
}

bool order_neig_asc(oid_t LHS, oid_t RHS)  {
  return g_dcop->agent( LHS ).nbNeighbours() < g_dcop->agent( RHS ).nbNeighbours();
}

bool order_neig_des(oid_t LHS, oid_t RHS)  {
  return g_dcop->agent( LHS ).nbNeighbours() > g_dcop->agent( RHS ).nbNeighbours();
}


PseudoTreeConstruction::PseudoTreeConstruction(Agent& owner)
  : Algorithm(owner), p_elected_root(0), p_heuristic(2)
{
  PseudoTreeOrdering::sptr p_tree_node(new PseudoTreeOrdering(owner));
  owner.setPtNode( p_tree_node );
}


PseudoTreeConstruction::~PseudoTreeConstruction() 
{ }


// It register messages and message handler
void PseudoTreeConstruction::initialize()
{ }


void PseudoTreeConstruction::finalize()
{
  if( g_dcop->agent( p_elected_root ).ptNode().constructed() )
    p_status = k_terminated;
  
  // Sequential Hack:
  // Reschedule the agent running this algorithm to let the 
  // calling routine to continue.
  if(owner().id() == p_elected_root) {
    p_status = k_terminated;
    g_dcop->agent( p_elected_root ).ptNode().setConstructed();
    Scheduler::initialize(g_dcop->agents());
  }
}


bool PseudoTreeConstruction::canRun()
{
  return (!terminated());
}


// TODO: This version of the algorithm is centralized - need a decentralized one.
void PseudoTreeConstruction::run()
{
  if (owner().id() != p_elected_root) {
    finalize();
    return;
  }

  if (!preferences::silent) {
	  std::cout << "PseudoTree Elected root is: " << p_elected_root << std::endl;
	  std::cout << "Heuristic root is: " << p_heuristic << std::endl;
  }

  int best_w = -1;
  int best_h = -1;

//  for (int h=0; h<6; h++) {
//    run_construction( h );
//    int w = get_induced_width();
//    if(best_w == 1 || w < best_w) {
//      best_w = w;
//      best_h = h;
//    }
// }

  best_h = p_heuristic; // // default heuristics 2
  run_construction( best_h );

  finalize();

  // Set separator set.
  for (Agent* a : g_dcop->agents()) {
    PseudoNode& n = a->ptNode();
    for( oid_t sp : get_separator_set( n ) ) {
      n.addSeparator( sp );
    }
  }

}


void PseudoTreeConstruction::run_construction(int heur) {

  std::vector<Agent*> agents = g_dcop->agents();
  std::map<oid_t, bool> discovered;

  for (Agent* a : agents) {
    a->ptNode().reset();	   // reset pseudo-node
    discovered[ a->id() ] = false; // set discoverable
  }

  oid_t root = p_elected_root; 
  std::stack<oid_t> S;
  S.push( root );
  g_dcop->agent(root).ptNode().setParent( Constants::nullid );

  // DFS exploration
  while (!S.empty()) {
    oid_t ai = S.top(); S.pop();
      
    if (!discovered[ ai ]) {
      // Get neighbors of ai and order them 
      std::vector<oid_t> N = g_dcop->agent( ai ).neighboursID();

      if (heur == 0)
    	  std::sort(N.begin(), N.end(), order_asc); // default
      else if (heur == 1)
    	  std::sort(N.begin(), N.end(), order_des);  //
      else if (heur == 2)
    	  std::sort(N.begin(), N.end(), order_neig_asc);  // (frodo default?)
      else if (heur == 3)
    	  std::sort(N.begin(), N.end(), order_neig_des);  //
      else if (heur == 4)
    	  std::sort(N.begin(), N.end(), lex_asc);  //
      else if (heur == 5)
    	  std::sort(N.begin(), N.end(), lex_des);  //

      // std::cout << owner().name() << " N: " << Utils::dump( N ) << "\n";
      
      for( oid_t ci : N ) {
	if( ci == g_dcop->agent(ai).ptNode().parentID() ) 
	  continue;
	
	S.push( ci );
	
	// Children of ai
	if (!discovered[ ci ]) {
	  g_dcop->agent(ai).ptNode().addChild( ci );  // ci is child of ai
	  g_dcop->agent(ci).ptNode().setParent( ai ); // ai is parent of ci
	}
	else {
	  // Set back-edges
	  g_dcop->agent(ai).ptNode().addPseudoParent( ci );
	  g_dcop->agent(ci).ptNode().addPseudoChild( ai );
	  g_dcop->agent(ci).ptNode().removeChild( ai );
	}
      }//-neighbors

      discovered[ ai ] = true;

    }//-not discovered
  }// while Stack is not empty

}


int PseudoTreeConstruction::get_induced_width() {
  int w_star = -1;
  for (Agent* a : g_dcop->agents()) {
    PseudoNode& n = a->ptNode();
    int w = get_separator_set( n ).size();
    if (w > w_star) w_star = w;
  }
  return w_star;
}


std::vector<oid_t> PseudoTreeConstruction::get_separator_set(PseudoNode& n) 
{
  if (n.isRoot())
    return std::vector<oid_t>();

  std::vector<oid_t> sep = Utils::concat( n.parent(), n.pseudoParents() );

  if (n.isLeaf())
    return sep;
  
  for (oid_t cid : n.children())
  {
    PseudoNode& c = g_dcop->agent(cid).ptNode();
    Utils::merge_emplace(sep, get_separator_set( c )); 
  }

  Utils::exclude_emplace(n.children(), sep);
  return sep;
}
