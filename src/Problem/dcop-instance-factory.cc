#include "Algorithms/algorithm-factory.hh"
#include "Kernel/globals.hh"
#include "Problem/dcop-instance-factory.hh"
#include "Problem/dcop-standard.hh"
// #ifdef USING_GPU
//  #include "Problem/dcop-standard-gpu.hh"
//#endif
#include "Problem/IO/input-settings.hh"
#include "Utilities/statistics.hh"

#include <iostream>
#include <string>

using namespace std;

DCOPinstance* DCOPinstanceFactory::create
  (DCOPmodel& model, InputSettings& settings)
{
  Statistics::registerTimer("wallclock@decomposition");
  
  for (Agent* old_a : model.agents()) {
    old_a->statistics().setMessageCosts(settings.innerMessageCost(),
					settings.outerMessageCost());
    old_a->statistics().setTimeouts(settings.simulatedTimeout(),
				    settings.wallclockTimeout(),
				    settings.NCCCsLimit(), 
				    (settings.memoryLimit() / model.nbAgents()));
  }
  
  DCOPinstance* dcop = nullptr;

  Statistics::startTimer("wallclock@decomposition");
  if(settings.preprocess().compare("standard") == 0)
    dcop = new DCOPstandard(model, settings);
  //#ifdef USING_GPU
  //  else if(settings.preprocess().compare("standard-gpu") == 0)
  //    dcop = new DCOPstandardGPU(model, settings);
  //#endif
  else
    ASSERT(dcop, "No DCOP model specified.");  
  
  // Update links:
  g_dcop = dcop; // TODO: to remove!
  g_dcop->set_elected_root( settings.get_elected_root() );
  g_dcop->set_heuristic( settings.get_heuristic() );

  // Creates the algorithm for the DCOP resolution method, in each of the 
  // DCOP agent.
  for (Agent* a : dcop->agents())
  {    
    Statistics::startTimer("simulated@decomposition", a->id());
    
    a->statistics().setMessageCosts(settings.innerMessageCost(), settings.outerMessageCost());
    a->statistics().setTimeouts(settings.simulatedTimeout(), settings.wallclockTimeout(), settings.NCCCsLimit(),
        (settings.memoryLimit() / model.nbAgents()));

    std::string method = settings.resolutionMethod();
    std::vector<std::string> params = settings.resolutionParameters();
    
    a->registerProtocol( AlgorithmFactory::create(*a, method, params) );
    
    Statistics::stopwatch("simulated@decomposition", a->id());    
  }
  Statistics::stopwatch("wallclock@decomposition");

  // Reset statistics for the solving phase.
  for(Agent* a : g_dcop->agents())
  {
    Statistics::registerCounter("NCCCs@decomposition", a->id());
    Statistics::setCounter("NCCCs@decomposition", a->id(), a->statistics().NCCC());
    a->statistics().reset();
  }
  return dcop;
}
