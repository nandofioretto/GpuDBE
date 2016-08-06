#include "Kernel/globals.hh"
#include "Kernel/solution.hh"
#include <vector>
#include <string>

Solution::Solution(int size)
  : p_cost(0), p_set( false )
{
  initialize(size);
}


Solution::~Solution() 
{ }


Solution::Solution(const Solution& other)
{
  p_values = other.p_values;
  p_cost   = other.p_cost;
  p_set    = other.p_set;
}


Solution& Solution::operator=(const Solution& other)
{
  if (this != &other)
  { 
    p_values = other.p_values;
    p_cost   = other.p_cost;
    p_set    = other.p_set;
  }
  return *this;
}


void Solution::initialize(int size)
{
  p_set = false;
  if( size > 0 )
    p_values.resize(size, Constants::NaN);
  p_cost = 0;
}


std::string Solution::dump() const
{
  std::string result;
  for( auto &v : p_values ) {
    result += (v == Constants::NaN) ? "? " : std::to_string(v) + " ";
  }
  result += " | " + std::to_string(p_cost);
  result += p_set ? " (set)" :  " (unset)";
 return result;
}
