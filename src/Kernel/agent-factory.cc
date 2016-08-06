#include "Kernel/agent-factory.hh"
#include "Kernel/agent.hh"

#include <string>


// Initializes static members
int AgentFactory::p_agents_ct = 0;

Agent* AgentFactory::create(std::string name)
{
  ASSERT(!name.empty(), "Error: agent name cannot be empty");

  Agent* agent = new Agent();
  agent->setId( p_agents_ct );
  agent->setName( name );
  ++p_agents_ct;
  
  return agent;
}
