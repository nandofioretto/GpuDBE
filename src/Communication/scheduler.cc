#include <deque>
#include <vector>
#include <string>

#include "Communication/scheduler.hh"
#include "Kernel/agent.hh"
#include "Kernel/agent-factory.hh"
#include "Problem/dcop-instance.hh"

std::deque<size_t> Scheduler::p_agent_queue;


void Scheduler::initialize(std::vector<Agent*> initial_nodes)
{
  p_agent_queue.clear();
  // Schedule the initial agents to be executed.
  for (Agent* a : initial_nodes)
    p_agent_queue.push_back( a->id() );
}


void Scheduler::run()
{
  while (not p_agent_queue.empty())
  {    
    Agent& exe = g_dcop->agent( p_agent_queue.front() );
    p_agent_queue.pop_front();    

    exe.statistics().resetPseudoMeasures(); // set pseudoNCCC = pseudoST = 0
    exe.statistics().setStartTimer();
    exe.runProtocol();
    exe.statistics().setSimulatedTime(exe.statistics().stopwatch());
    exe.checkOutOfLimits();
  }
}


void Scheduler::FIFOinsert(size_t agent_id)
{
  if(std::find(p_agent_queue.begin(), p_agent_queue.end(), agent_id) == 
     p_agent_queue.end() )
    p_agent_queue.push_back( agent_id );
}


void Scheduler::LIFOinsert(size_t agent_id)
{
  if(std::find(p_agent_queue.begin(), p_agent_queue.end(), agent_id) == 
     p_agent_queue.end() )
    p_agent_queue.push_front( agent_id );
}


void Scheduler::FIFOinsert(std::vector<size_t> agents_id)
{
  for (size_t a : agents_id)
    p_agent_queue.push_back(a);
}


void Scheduler::LIFOinsert(std::vector<size_t> agents_id)
{
  for (size_t a : agents_id)
    p_agent_queue.push_front(a);
}


void Scheduler::clearQueue()
{
  p_agent_queue.clear();
}


std::string Scheduler::dump()
{
  std::string result = "Scheduler content: <-[";
  for (size_t a : p_agent_queue)
    result += std::to_string(a) + " ";

  return result += "]";
}
