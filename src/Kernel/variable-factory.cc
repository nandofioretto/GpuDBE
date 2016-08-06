#include "Kernel/variable-factory.hh"
#include "Kernel/int-variable.hh"
#include "Kernel/domain-factory.hh"
#include "Kernel/int-domain.hh"
#include "Kernel/agent-factory.hh"
#include "Kernel/agent.hh"

using namespace rapidxml;

// Initializes static members
int VariableFactory::p_variables_ct = 0;


IntVariable* VariableFactory::create(xml_node<>* varXML, 
                                     xml_node<>* domsXML, 
                                     std::vector<Agent*> agents)
{
  // after you create the variable you can use the var id to identify it
  std::string name = varXML->first_attribute("name")->value();
  std::string owner_name = varXML->first_attribute("agent")->value();
  std::string domain = varXML->first_attribute("domain")->value();
  
  // Retrieve domain xml_node:
  xml_node<>* domXML = domsXML->first_node("domain");
  while (domain.compare(domXML->first_attribute("name")->value()) != 0)
  {
    domXML = domXML->next_sibling();
    ASSERT(domXML, 
      "No domain associated to variable " << name << " could be found.");
  }

  // look for owner in agents vector:
  Agent* owner = nullptr;
  for (Agent* a : agents) {
    if( a->name().compare(owner_name) == 0 )
      owner = a;
  }
  ASSERT(owner, 
    "No agent associated to variable " << name << " could be found.");
  
  IntVariable* var = new IntVariable();
  var->setId( p_variables_ct );
  var->setName( name );
  var->registerOwner( owner->id() );
  var->setDomain(DomainFactory::create( domXML ));

  // Register variable in the agent owning it
  owner->registerVariable( var );
  
  ++p_variables_ct;

  return var;
}


IntVariable* VariableFactory::create(std::string name, Agent* owner, IntDomain& dom)
{
  ASSERT(owner, "No agent associated to variable " << name << " given.");
    
  IntVariable* var = new IntVariable();
  var->setId( p_variables_ct );
  var->setName( name );
  var->registerOwner( owner->id() );
  // Creates new domain
  var->setDomain(dom.min(), dom.max());

  // Register variable in the agent owning it
  owner->registerVariable( var );
  
  ++p_variables_ct;

  return var;
}


IntVariable* VariableFactory::create(std::string name, Agent* owner, 
				     int min, int max)
{
  ASSERT(owner, "No agent associated to variable " << name << " given.");
    
  IntVariable* var = new IntVariable();
  var->setId( p_variables_ct );
  var->setName( name );
  var->registerOwner( owner->id() );
  // Creates new domain
  var->setDomain(min, max);

  // Register variable in the agent owning it
  owner->registerVariable( var );
  
  ++p_variables_ct;

  return var;
}
