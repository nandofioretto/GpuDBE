#ifndef ULYSSES_PROBLEM__DCOP_INSTANCE_FACTORY_H_
#define ULYSSES_PROBLEM__DCOP_INSTANCE_FACTORY_H_

#include "dcop-instance.hh"

class Model;
class InputSettings;
class DCOPmodel;

// The DCOP instance factory class.
class DCOPinstanceFactory
{
public:
  // Creates a specific DCOP instance associated to a given preprocessing type,
  // which is specified in the DCOP model given as a parameter.
  static DCOPinstance* create(DCOPmodel& model, InputSettings& settings);
};

#endif // ULYSSES_PROBLEM__DCOP_INSTANCE_FACTORY_H_
