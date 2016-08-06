#include <string>
#include <rapidxml.hpp>

#include "Kernel/domain-factory.hh"
#include "Kernel/int-domain.hh"
#include "Kernel/bound-domain.hh"

using namespace rapidxml;

// Initializes static members
int DomainFactory::p_domains_ct = 0;


// Very important!
// TODO: always use interval domains!!
// Decides which type of domain to create:
// 1. Bound [min max]
// 2. Interval 
// 3. SmallDense ( < 64 )
// TODO: Remove assumption domains must be integer
IntDomain* DomainFactory::create(xml_node<>* domXML)
{
  std::string name = domXML->first_attribute("name")->value();
  size_t nb_vals   = atoi(domXML->first_attribute("nbValues")->value());
  std::string content = domXML->value();

  size_t ival = content.find("..");
  ASSERT (ival != std::string::npos, "Cannot handle not contiguous domains");
  int min = atoi( content.substr(0, ival).c_str() ); 
  int max = atoi( content.substr(ival+2).c_str() ); 

  WARNING (nb_vals == max-min+1, 
	   "Content of domain " << name << " is inconsistent with the declared size");
    
  BoundDomain* domain = new BoundDomain(min, max);

  ///// PROCESS ALL VALUES (small dense domain)
  // size_t ival = content.find("..");
  // if (ival != std::string::npos)
  // {
  //   values[ 0 ] = atoi( content.substr(0, ival).c_str() ); 
  //   for (int i=1; i<nb_vals; i++)
  //   {
  //     values[ i ] = values[ i - 1 ] + 1;
  //   }
  // }
  // else
  // {
  //   stringstream ss( content );
  //   int i = 0;
  //   while( i < nb_vals )
  //   {
  //     ss >> values[ i++ ];
  //   }
  // }

  domain->setId( p_domains_ct );
  domain->setName( name );
  ++p_domains_ct;

  return domain;
}
