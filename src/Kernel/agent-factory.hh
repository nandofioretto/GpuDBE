#ifndef ULYSSES_KERNEL__AGENTS__AGENT_FACTORY_H_
#define ULYSSES_KERNEL__AGENTS__AGENT_FACTORY_H_

#include <string>

class Agent;

// The Agent Factory class.
class AgentFactory
{
public:
  // Construct and returns a new agent, given its name.
  static Agent* create(std::string name); 
  
  // It resets the agents counter.
  static void resetCt() 
    { p_agents_ct = 0; }

private:
  // The Agent counter. It holds the ID of the next agent to be created.
  static int p_agents_ct;
};

#endif // ULYSSES_KERNEL__AGENTS__AGENT_FACTORY_H_
