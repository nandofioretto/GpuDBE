#ifndef ULYSSES_KERNEL__VARIABLES__VARIABLE_FACTORY_H_
#define ULYSSES_KERNEL__VARIABLES__VARIABLE_FACTORY_H_

#include "Kernel/globals.hh"
#include <string>
#include <vector>
#include <rapidxml.hpp>

class IntVariable;
class IntDomain;
class Agent;

// The Variable Factory class. 
// It is used to create a new variable from input.
// TODO: Generalize this class to handle any variable type (e.g., floats)
class VariableFactory
{
public:
  // It constructs and returns a new variable.
  // It also construct a new domain to be associated to the new variable.
  static IntVariable* create(rapidxml::xml_node<>* varXML, 
                             rapidxml::xml_node<>* domsXML, 
                             std::vector<Agent*> agents);

  // It constructs and returns a new variable, given its name, it's agent
  // owner, and its domain.
 static IntVariable* create(std::string name, Agent* owner, IntDomain& dom);

  // It constructs and returns a new variable, given its name, it's agent
  // owner, and its domain bounds.
  static IntVariable* create(std::string name, Agent* owner, 
			     int min, int max);

  // It resets the variables counter.
  static void resetCt()
    { p_variables_ct = 0; }

private:
  // The variable counter. It holds the ID of the next variable to be created.
  static int p_variables_ct;

};

#endif // ULYSSES_KERNEL__VARIABLES__VARIABLE_FACTORY_H_
