#ifndef ULYSSES_COMMUNICATION__SCHEDULER_H_
#define ULYSSES_COMMUNICATION__SCHEDULER_H_

#include <string>
#include <vector>
#include <deque>

class Agent;

// This will be used as a global class when simulating a 
// decentralized system.
class Scheduler
{

public:

  // Initializes the scheduler by inserting the intial agents given as a 
  // parameter.
  static void initialize(std::vector<Agent*> initial_nodes = std::vector<Agent*>());

  // It executes the scheduler. Elements are extracted from deque::front.
  static void run();

  // It schedule the execution of the agent given as a parameter. The agent 
  // is inserted in the deque::back
  static void FIFOinsert(size_t agent_id);

  // It schedule the execution of the agent given as a parameter. The agent 
  // is inserted in the deque::front
  static void LIFOinsert(size_t agent_id);
  
  // It schedule the execution of the agents given as a parameter. The agents
  // are inserted in the given order in the deque::back
  static void FIFOinsert(std::vector<size_t> agents_id);

  // It schedule the execution of the agents given as a parameter. The agents
  // are inserted in the given order in the deque::front
  static void LIFOinsert(std::vector<size_t> agents_id);

  // It removes all the scheduled agents from the queue;
  static void clearQueue();
  
  // It returns a description of this scheduler.
  static std::string dump();

private:
  // The queue of agents to be scheduled.
  static std::deque<size_t> p_agent_queue;
  // later manage presence:
  // (L/F)IFOinsertUniqe
  // have a map of presence of agents -> bool y/n
  
};

#endif // ULYSSES_COMMUNICATION__QUEUE_SCHEDULER_H_
