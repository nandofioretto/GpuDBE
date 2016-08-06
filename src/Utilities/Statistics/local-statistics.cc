#include "Utilities/Statistics/local-statistics.hh"
#include "Utilities/Statistics/message-statistics.hh"


std::string LocalStatistics::dump() const
{
  std::string res = "NCCCs: " + std::to_string(p_NCCCs) + "\n";
  res += "sim time       : " + std::to_string(p_simulated_us);
  res += " / " + std::to_string(Timeouts::simulatedTimeout()) + "\n";
  res += "wallclock time : " + std::to_string(p_wallclock_us);
  res += " / " + std::to_string(Timeouts::wallclockTimeout()) + "\n";
  res += "Nb msg sent    : " + std::to_string(p_nb_inner_msg_sent);
  res += " / " + std::to_string(p_nb_outer_msg_sent) + "\n";
  res += "Memory used    : " + std::to_string(p_used_memory);
  res += " / " + std::to_string(Timeouts::memoryLimit()) + "\n";
  
  return res;
}
