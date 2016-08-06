#ifndef ULYSSES_ALGORITHMS__DPOP__UTIL_MSG_H_
#define ULYSSES_ALGORITHMS__DPOP__UTIL_MSG_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Communication/message.hh"
//#include "Kernel/codec.hh"
#include "Kernel/globals.hh"


// The util message of basic DPOP.
// For each *valid* value combination of the boundary variables it stores the 
// utility value associated to such value combination and the the best value 
// combination of its private variables.
// 
// @note:
// The values are hence retrieved in the value propagation phase, either by a 
// search or by retrieving the values stored.
class UtilMsg : public Message
{
public:
  typedef std::unique_ptr<UtilMsg> uptr;
  typedef std::shared_ptr<UtilMsg> sptr;

  typedef size_t code_t; // The code associated to the tuple of values
  typedef int idx_t;	 // the index of the array UTILS
  
public:
  UtilMsg();

  virtual ~UtilMsg();

  // Check equality of two Util messages. It only checks message source and
  // destination.
  bool operator==(const UtilMsg& other);

  // It creates a copy of this message. 
  virtual UtilMsg* clone();

  // It returns the message type.
  virtual std::string type() const
  {
    return "UTIL";
  }

  // It resets the message content (without affeting the message header).
  virtual void reset()
  { }

  // It returns a message description.
  virtual std::string dump() const;

  // It returns the list of the variables of the message UTIL.
  // std::vector<oid_t>& getVariables()
  // {
  //   return p_util_table_rows->variables();
  // }

  // void setCodec(Codec::sptr rows_code) 
  // {
  //   p_util_table_rows = rows_code;
  // }

  // It returns the UTIL value, associated to the value combination given 
  // as a parameter
  // cost_t getUtil(std::vector<int> query, int world_id)
  // {
  //   size_t code = p_util_table_rows->encode(query);
  //   return (*p_util_table)[ code ][ world_id ];
  // }

  // It saves the cost associated to the tuple of values given as a paramter 
  // void setUtilTable(util_table_t* table_ptr)
  // {
  //   p_util_table = table_ptr;
  // }
  
  // It returns the number of utils in the UTIL vector.
  // size_t nbUtils() const
  // {
  //   return p_util_table->size();
  // }

  // It returns the number of variables in the UTIL vector.
  // size_t nbVars() const
  // {
  //   return p_util_table_rows->variables().size();
  // }

  void setContent(int* utilTable, size_t nbRows) {
    p_util_table = utilTable;
    p_util_table_rows = nbRows;
  }

  size_t getUtilTableRows() { return p_util_table_rows; }
  size_t getUtilTableCols() { return p_nb_worlds; }
  int* getUtilTablePtr() { return p_util_table; }
  
protected:  
  DISALLOW_COPY_AND_ASSIGN(UtilMsg);
    
private:
  // The util vector which contains the util costs assiciated to each 
  // value combination of the variables in p_variables.
  int* p_util_table;
  size_t p_util_table_rows;
  size_t p_nb_worlds;

  // The codec for encoding and decoding the util messages.
  // Codec::sptr p_util_table_rows;

};

#endif // ULYSSES_ALGORITHMS__DPOP__UTIL_MSG_H_
