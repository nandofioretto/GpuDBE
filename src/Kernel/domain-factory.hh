#ifndef ULYSSES_KERNEL__DOMAINS__DOMAIN_FACTORY_H_
#define ULYSSES_KERNEL__DOMAINS__DOMAIN_FACTORY_H_

#include "Kernel/globals.hh"

#include "Kernel/globals.hh"
#include <string>
#include <rapidxml.hpp>

class IntDomain;

// The Domain Factory class.
class DomainFactory
{
public:
  // Construct and returns a new domain.
  static IntDomain* create(rapidxml::xml_node<>* agentXML); 

  // It resets the variables counter.
  static void resetCt()
    { p_domains_ct = 0; }

private:
  // The domains counter. It holds the ID of the next domain to be created.
  static int p_domains_ct;
};

#endif // ULYSSES_KERNEL__DOMAINS__DOMAIN_FACTORY_H_
