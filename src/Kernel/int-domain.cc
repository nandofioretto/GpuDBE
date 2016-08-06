#include "Kernel/domain.hh"
#include "Kernel/int-domain.hh"
#include "Kernel/value-iterator.hh"

#include <math.h>       /* round, floor, ceil, trunc */
#include <map>

using namespace std;

// Initializes the Subsumed events map
std::map< Domain::EventType, std::vector<Domain::EventType> > 
IntDomain::subsumed_events_ = {
  {Domain::kSingleton, {Domain::kSingleton, Domain::kBoundChanged, Domain::kAny}},
  {Domain::kBoundChanged, {Domain::kBoundChanged, Domain::kAny}},
  {Domain::kAny, {Domain::kAny}}};


IntDomain::IntDomain()
  : domain_type_(kUndefined)
{ } 


bool IntDomain::operator==(IntDomain& other)
{
  if (this->size() != other.size())
    return false;
		
  ValueIterator& vit = other.valueIterator();
  while (vit.next()) 
  {
    int next = vit.getAdvance();
    if (not contains(next))
      return false;
  }
  
  return true;
}


vector<int> IntDomain::content()
{
  vector<int> result( this->size() );
  ValueIterator& vit = valueIterator();
  int i = 0;
  while (vit.next())
    result[ i++ ] = vit.getAdvance();

  return result;    
}


int IntDomain::lex(IntDomain& other)
{
  // value iterator associated to current domain
  ValueIterator& cit = valueIterator();
  // value iterator associated to 'other' domain
  ValueIterator& oit = other.valueIterator();
  
  int i,j;
		
  while (cit.next()) 
  {
    i = cit.getAdvance();
    
    if (oit.next() ) 
    {
      j = oit.getAdvance();
      
      if( i < j ) return -1;
      else if( j < i ) return 1;
    }
    else  // current domain size > other domain size
      return 1;
  }  
  if (oit.next()) return -1;  // other domain size > current domain size
  
  return 0;  // domain sizes are equal
}
