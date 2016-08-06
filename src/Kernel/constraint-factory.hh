#ifndef ULYSSES_KERNEL__CONSTRAINTS__CONSTRAINT_FACTORY_H_
#define ULYSSES_KERNEL__CONSTRAINTS__CONSTRAINT_FACTORY_H_

#include "Kernel/globals.hh"

#include <rapidxml.hpp>
#include <string>
#include <vector>

class Constraint;
class TableConstraint;
class IntVariable;
class DCOPmodel;
class Codec;

class ConstraintFactory
{
public:
  // It resets the constraints counter.
  static void resetCt() 
  { p_constraints_ct = 0; }
  
  // It constructs and returns a new constraint.
  static Constraint* create(rapidxml::xml_node<>* conXML, 
                            rapidxml::xml_node<>* relsXML,
                            std::vector<Agent*> agents,
                            std::vector<IntVariable*> variables);

private:
  
  // The Constraints counter. It holds the ID of the next constraint to be
  // created.
  static int p_constraints_ct;

  // It returns the scope of the constraint 
   static std::vector<IntVariable*> getScope
     (rapidxml::xml_node<>* conXML, std::vector<IntVariable*> variables);

   // It constructs an extensional hard constraint from the xml bit
   static TableConstraint* createExtSoftConstraint
     (rapidxml::xml_node<>* conXML, rapidxml::xml_node<>* relXML, std::vector<IntVariable*> variables);

   // Sets common constraint properties and initializes mappings.
   static void setProperties(Constraint* c, std::string name, std::vector<Agent*> agents);


   static std::string findNext(std::string content, size_t& lhs, size_t& rhs, cost_t& m_cost);

};


#endif // ULYSSES_KERNEL__CONSTRAINTS__CONSTRAINT_FACTORY_H_
