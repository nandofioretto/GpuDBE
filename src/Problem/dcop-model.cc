#include <fstream>
#include <iostream>
#include <string>
#include <rapidxml.hpp>

#include "Problem/dcop.hh"
#include "Problem/dcop-model.hh"
#include "Algorithms/algorithm-factory.hh"
#include "Kernel/agent-factory.hh"
#include "Kernel/agent.hh"
#include "Kernel/globals.hh"
#include "Kernel/variable-factory.hh"
#include "Kernel/constraint-factory.hh"
#include "Utilities/Statistics/local-statistics.hh"
#include "Problem/IO/xml.hh"
#include "Problem/IO/input-problem.hh"

using namespace rapidxml;
using namespace std;

DCOPmodel::DCOPmodel(InputProblem& problem)
{
  this->import(problem.getFile());
}


DCOPmodel::~DCOPmodel()
{
  for (Agent* a : p_agents) delete a;
  for (IntVariable* v : p_variables) delete v;
  for (IntDomain* d : p_domains) delete d;
  for (Constraint* c : p_constraints) delete c;
}


void DCOPmodel::import(string xml_file)
{
  int size = 0;
  string input_xml;
  string line;
  ifstream in( xml_file.c_str() );

  ASSERT (in.is_open(), "Error: cannot open the input file.");
  
  while (getline( in, line ))
    input_xml += line;
 
  // make a safe-to-modify copy of input_xml
  vector<char> xml_copy(input_xml.begin(), input_xml.end());
  xml_copy.push_back('\0');
  xml_document<> doc;
  doc.parse<parse_declaration_node | parse_no_data_nodes>( &xml_copy[ 0 ] );
  xml_node<>* root = doc.first_node("instance");
  xml_node<>* xpre = root->first_node("presentation");
  
  // Set the problem optimization type
  ASSERT(xpre->first_attribute("maximize"), 
	  "Invalid optimization Problem (maximize/minimize) specified \
      in the XML file!");
  
  string optMax = xpre->first_attribute("maximize")->value();  
  if (optMax.compare("true") == 0) {
    p_optimization = DCOPinfo::kMaximize;
    Constants::worstvalue = -Constants::infinity;
    Constants::bestvalue  = Constants::infinity;
    DCOPinfo::optimization = DCOPinfo::kMaximize;
  }
  else {
    p_optimization = DCOPinfo::kMinimize;
    Constants::worstvalue = Constants::infinity;
    Constants::bestvalue  = -Constants::infinity;
    DCOPinfo::optimization = DCOPinfo::kMinimize;
  }

  parseXMLAgents(root);
  parseXMLVariables(root);
  parseXMLConstraints(root); 

  for (Agent* a : p_agents) {
    a->orderContextVariables();
  }
  
  in.close();
}


void DCOPmodel::parseXMLAgents(xml_node<>* root)
{
  xml_node<>* xagents = root->first_node("agents");
  int nb_agents = atoi( xagents->first_attribute("nbAgents")->value() );
  xml_node<>* xagent  = xagents->first_node("agent");

  do
  {
    string name = xagent->first_attribute("name")->value();
    Agent* agent = AgentFactory::create( name );
    p_agents.push_back( agent );
    
    xagent = xagent->next_sibling();
  } while (xagent);

  ASSERT( nb_agents == p_agents.size(), "Number of agents read " 
    << p_agents.size() << " differs from the number of agents declared.");
}


void DCOPmodel::parseXMLVariables(xml_node<>* root)
{
  xml_node<>* xdoms = root->first_node("domains");
  xml_node<>* xvars = root->first_node("variables");
  xml_node<>* xvar = xvars->first_node("variable");
  int nb_variables = atoi( xvars->first_attribute("nbVariables")->value() );

  do
  {
    IntVariable* var = VariableFactory::create( xvar, xdoms, p_agents );
    p_variables.push_back(var);
    
    xvar = xvar->next_sibling();
  } while ( xvar );

  ASSERT( nb_variables == p_variables.size(), "Number of variables read " 
    << p_variables.size() << " differs from the number of variables declared.");
}


void DCOPmodel::parseXMLConstraints(xml_node<>* root)
{
  // Parse and create constraints
  xml_node<>* xrels = root->first_node("relations");
  xml_node<>* xcons = root->first_node("constraints");
  int size = atoi( xcons->first_attribute("nbConstraints")->value() );
  if (size > 0)
  {
    xml_node<>* xcon  = xcons->first_node("constraint");
    do {
      Constraint* con = 
        ConstraintFactory::create( xcon, xrels, p_agents, p_variables );
      p_constraints.push_back(con);
      
      xcon = xcon->next_sibling();
    } while ( xcon );
  }

  ASSERT( size == p_constraints.size(), "Number of constraints read " 
    << p_constraints.size() << " differs from the number of items declared.");
}
