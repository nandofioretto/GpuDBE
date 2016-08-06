#ifndef ULYSSES_ALGORITHMS__DPOP__VALUE_MSG_H_
#define ULYSSES_ALGORITHMS__DPOP__VALUE_MSG_H_

#include <string>
#include <vector>
#include <map>
#include <memory>

#include "Communication/message.hh"
#include "Kernel/codec.hh"
#include "Kernel/globals.hh"
#include "Problem/dcop-instance.hh"

// The VALUE message of basic DPOP.
// It collects the values associated to the DCOP solution detected
// during the UTIL propagation phase. 
class ValueMsg : public Message
{
public:
  typedef std::unique_ptr<ValueMsg> uptr;
  typedef std::shared_ptr<ValueMsg> sptr;

  ValueMsg();

  virtual ~ValueMsg();

  // Check equality of two Value messages. It only checks message source and
  // destination.
  bool operator==(const ValueMsg& other);

  // It creates a copy of this message. 
  virtual ValueMsg* clone();

  // It returns the message type.
  virtual std::string type() const
  {
    return "VALUE";
  }

  // It resets the message content (without affeting the message header).
  virtual void reset()
  { }

  // It returns a message description.
  virtual std::string dump() const;


  void set_sep_var_id(oid_t var) {
    p_sep_vars.push_back( var );
  }
  
  std::vector<oid_t>& get_sep_variables() {
    return p_sep_vars;
  }

  void set_value(int value, int vid){
    p_sep_var_values[vid] = value;
  }

  int get_value(int vid)
  {
    return p_sep_var_values[vid];
  }


protected:
  DISALLOW_COPY_AND_ASSIGN(ValueMsg);


private:
  // it includes this variable.
  std::vector<oid_t> p_sep_vars;

  // The values for each of the world 
  std::map<oid_t, int> p_sep_var_values;
 
};

#endif // ULYSSES_ALGORITHMS__DPOP__VALUE_MSG_H_
