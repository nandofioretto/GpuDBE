#include "Algorithms/DPOP/value-msg.hh"

using namespace std;

ValueMsg::ValueMsg()
{ }


ValueMsg::~ValueMsg()
{ }


// Note: this is a protected copy constructor - it can only called 
// by this object and used by the clone function. 
ValueMsg::ValueMsg(const ValueMsg& other)
  : Message(other)
{
  p_sep_vars = other.p_sep_vars;
  p_sep_var_values = other.p_sep_var_values;
}

bool ValueMsg::operator==(const ValueMsg& other)
{
  return (source() == other.source() && destination() == other.destination());
}


ValueMsg* ValueMsg::clone()
{
  return new ValueMsg(*this);
}


string ValueMsg::dump() const
{
  string result = type() += Message::dump();

  result+="\n  Content: vars :";
  for( auto& kv: p_sep_var_values) { 
    result += std::to_string(kv.first) + ": ";
    result += std::to_string(kv.second) + "\n";
  }
  return result;

}
