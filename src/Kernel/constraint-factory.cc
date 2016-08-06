#include <string>
#include <sstream>
#include <rapidxml.hpp>
#include <iterator>
#include <vector>

#include "Kernel/agent.hh"
#include "Kernel/int-variable.hh"
#include "Kernel/constraint-factory.hh"
#include "Kernel/constraint.hh"
#include "Kernel/table-constraint.hh"

#include "Problem/dcop-model.hh"
#include "Kernel/globals.hh"
#include "Utilities/utils.hh"

using namespace rapidxml;
using namespace std;

// Initializes static members
int ConstraintFactory::p_constraints_ct = 0;

// It constructs and returns a new constraint.
Constraint* ConstraintFactory::create(xml_node<>* conXML,
                                      xml_node<>* relsXML,
                                      vector<Agent*> agents,
                                      vector<IntVariable*> variables)
{
  string name = conXML->first_attribute("name")->value();
  string rel = conXML->first_attribute("reference")->value();

  // Retrieve the relation associated to this constraint:
  int size = atoi(relsXML->first_attribute("nbRelations")->value());
  ASSERT(size > 0, "No Relation associated to constraint " << name);

  xml_node<>* relXML = relsXML->first_node("relation");
  while (rel.compare(relXML->first_attribute("name")->value()) != 0)
  {
    relXML = relXML->next_sibling();
    ASSERT(relXML, "No Relation associated to constraint " << name);
  }

  // Proces constraints according to their type.
  string semantics = relXML->first_attribute("semantics")->value();

  // Ext-Soft-Constraint
  if (semantics.compare("soft") == 0 or // back-compatibility with FRODO 2.11
      semantics.compare("hard") == 0 or // back-compatibility with FRODO 2.11
      semantics.compare("extensionalSoft") == 0 or
      semantics.compare("extensionalHard") == 0 )
  {
    TableConstraint* c = createExtSoftConstraint(conXML, relXML, variables);
    setProperties(c, name, agents);
    ++p_constraints_ct;
    return c;
  }
  else if (semantics.compare("intensionalHard") == 0)
  {
    ASSERT(false, "Int Hard Constraint processing Not implemented");
  }
  else if (semantics.compare("intensionalSoft") == 0)
  {
    ASSERT(false, "Int Soft Constraint processing Not implemented");
  }
  else
    ASSERT(false, "Error in the relation semantics: " << semantics);

  return nullptr;
}


// Jul 5, ok
void ConstraintFactory::setProperties
  (Constraint* c, string name, vector<Agent*> agents)
{
  c->setId( p_constraints_ct );
  c->setName( name );

  // Registers this constraint in the agents owning the variables of the 
  // constraint scope.
  for (IntVariable* v : c->scope())
  {
    Agent* v_owner = nullptr;
    for (Agent* a : agents) if( a->id() == v->ownerId() ) { v_owner = a; }
      ASSERT(v_owner, "Error in finding variable owner\n");
      v_owner->registerConstraint( c );
  }
}


// Jul 5, ok
vector<IntVariable*> ConstraintFactory::getScope
  (xml_node<>* conXML, vector<IntVariable*> variables)
{
  int arity = atoi(conXML->first_attribute("arity")->value());
  
  // Read constraint scope
  string p_scope = conXML->first_attribute("scope")->value();
  vector<IntVariable*> scope(arity, nullptr);
  stringstream ss(p_scope); int c = 0; string var;
  while( c < arity ) 
  {
    ss >> var; 
    IntVariable* v_target = nullptr;
    for (IntVariable* v : variables) 
      if( v->name().compare(var) == 0 ) 
        v_target = v;
    ASSERT(v_target, "Error in retrieving scope of constraint\n");
     
    scope[ c++ ] = v_target;
  }
  return scope;
}


// Jul 5, ok
TableConstraint* ConstraintFactory::createExtSoftConstraint
(xml_node<>* conXML, xml_node<>* relXML, std::vector<IntVariable*> variables)
{
  // Read Relation Properties
  string name = relXML->first_attribute("name")->value();
  int arity = atoi( relXML->first_attribute("arity")->value() );
  size_t nb_tuples = atoi( relXML->first_attribute("nbTuples")->value() );
  ASSERT( nb_tuples > 0, "Extensional Soft Constraint " << name << " is empty");

  vector<IntVariable*> scope = getScope( conXML, variables );

  // Get the default cost
  cost_t def_cost = Constants::worstvalue;

  if (relXML->first_attribute("defaultCost"))
  {
    string cost = relXML->first_attribute("defaultCost")->value();
    if (cost.compare("infinity") == 0 )
      def_cost = Constants::infinity;
    else if( cost.compare("-infinity") == 0 )
      def_cost = -Constants::infinity;
    else
      def_cost = atoi( cost.c_str() );
  }

  //
  // ExtSoftConstraint* con =
  //  createExtSoftConstraint( arity, scope, nb_tuples, def_cost );

  TableConstraint* con = new TableConstraint(scope, def_cost);

  string content = relXML->value();
  size_t lhs = 0, rhs = 0;

  // replace all the occurrences of 'infinity' with a 'INFTY'
  while (true)
  {
    rhs = content.find("infinity", lhs);
    if (rhs != string::npos)
      content.replace( rhs, 8, to_string(Constants::infinity) );
    else break;
  };

  // replace all the occurrences of ':' with a '\b'
  // cost_t best_bound = Constants::worstvalue;
  // cost_t worst_bound = Constants::bestvalue;

  cost_t m_cost; bool multiple_cost;
  int* tuple = new int[ arity ];
  int trim_s, trim_e;
  size_t count = 0;
  string str_tuples;
  lhs = 0;
  while (count < nb_tuples)
  {
    //multiple_cost = true;
    rhs = content.find(":", lhs);
    if (rhs < content.find("|", lhs))
    {
      if (rhs != string::npos)
      {
        m_cost = atoi( content.substr(lhs,  rhs).c_str() );

        // Keep track of the best/worst bounds
        // best_bound = Utils::getBest(m_cost, best_bound);
        // worst_bound = Utils::getWorst(m_cost, worst_bound);

        lhs = rhs + 1;
      }
    }

    rhs = content.find("|", lhs);
    trim_s = lhs, trim_e = rhs;
    lhs = trim_e+1;

    if (trim_e == string::npos) trim_e = content.size();
    else while (content[ trim_e-1 ] == ' ') trim_e--;

    str_tuples = content.substr( trim_s, trim_e - trim_s );
    str_tuples = Utils::rtrim(str_tuples);
    stringstream ss( str_tuples );

    //int tmp;
    while( ss.good() )
    {
      for (int i = 0; i < arity; ++i) {
        // ss >> tmp;
        // tuple[ i ] = scope[ i ]->getDomain().get_pos( tmp );
        ss >> tuple[ i ];
      }
      std::vector<int> v(tuple, tuple + arity);
      con->setCost( v, m_cost );
      count++;
    }
  }

  // con->setBestCost(best_bound);
  // con->setWorstCost(worst_bound);

  delete[] tuple;

  return con;
}
