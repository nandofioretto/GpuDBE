#include "Communication/message.hh"
#include "Kernel/globals.hh"
#include "Kernel/agent.hh"
#include "Problem/dcop-instance.hh"
#include "Utilities/Statistics/local-statistics.hh"


void Message::updateStatistics(LocalStatistics& localstats)
{
  if(g_dcop->stdModelAgentId(p_source_id) == 
     g_dcop->stdModelAgentId(p_destination_id)) 
  {
    p_stats.setNCCC(localstats.NCCC() + localstats.innerMessageCost());
    localstats.incrNbInnerMessageSent();
    // stores in the message the last number of NCCCs recorded while running
    // the algorithm. 
    p_stats.setPseudoNCCCs(localstats.pseudoNCCCs());
    p_stats.setPseudoSimulatedTime(
      localstats.stopwatch() - localstats.simulatedTime());
  }
  else {
    p_stats.setNCCC(localstats.NCCC() + localstats.outerMessageCost());    
    localstats.incrNbOuterMessageSent();
    p_stats.setPseudoNCCCs(0);
    p_stats.setPseudoSimulatedTime(0);
  }

  // Sets the message time to be the total agent runtime (recorder up to 
  // this instant).
  p_stats.setSimulatedTime(localstats.stopwatch());
}


std::string Message::dump() const
{
  std::string result;
  if( source() != Constants::nullid)
    result += g_dcop->agent(source()).name();
  result+=" -[" + type() + "]-> ";
  if( destination() != Constants::nullid)
    result += g_dcop->agent(destination()).name();
  return result;
}
