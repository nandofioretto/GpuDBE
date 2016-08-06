#include "Algorithms/DPOP/util-msg.hh"

using namespace std;

UtilMsg::UtilMsg()
{ }


UtilMsg::~UtilMsg()
{ }


// Note: this is a protected copy constructor - it can only called 
// by this object and used by the clone function. 
UtilMsg::UtilMsg(const UtilMsg& other)
  : Message(other)
{
  p_util_table = other.p_util_table;
  p_util_table_rows = other.p_util_table_rows;
  p_nb_worlds = other.p_nb_worlds;
}

bool UtilMsg::operator==(const UtilMsg& other)
{
  return (source() == other.source() && destination() == other.destination());
}


// Note: This action here, invalidate the further use of codec in the current
// object!
// This make sense in the current implementation, as the messages are cloned
// just befored being sent, then destroyed.
UtilMsg* UtilMsg::clone()
{  
  UtilMsg* msg = new UtilMsg(*this);
  // msg->p_util_table_rows = std::move(this->p_util_table_rows);
  return msg;
}


string UtilMsg::dump() const
{
  string result = type() += Message::dump();
  return result;
}
