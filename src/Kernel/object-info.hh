#ifndef ULYSSES_KERNEL__OBJECTINFO_H_
#define ULYSSES_KERNEL__OBJECTINFO_H_

#include <string>
#include <iostream>
#include "globals.hh"

// Generic Object Information. Uniquely identifies an object and 
// defines standard comparison and ordering operators based on 
// the object ID.
// Every object for which instances need a unique identification
// and a name, must inherit this Class.
class ObjectInfo
{
 public:

  // It initializes the object with an empty id and name. 
  ObjectInfo()
    : id_(Constants::nullid), name_("")
  { }
  
  // It generates a copy of the current object.
  ObjectInfo(const ObjectInfo& other)
    : id_ (other.id_), name_ (other.name_)
  { }
  
  virtual ~ObjectInfo() 
  { }
  
  // It generates a copy of the current object.
  virtual ObjectInfo& operator=(const ObjectInfo& other)
  {
    if (this != &other)
    {
      id_   = other.id_;
      name_ = other.name_;
    }
    return *this;
  }
  
  virtual bool operator==(const ObjectInfo& other)
  {
    return (id_ == other.id_);
  }
  
  virtual bool operator<(const ObjectInfo& other)
  {
    return (id_ < other.id_); 
  }

  // It sets the object id.
  void setId(oid_t id)
  { 
    id_ = id; 
  } 
  
  // It returns the object id.
  oid_t id() const 
  { 
    return id_; 
  }
  
  // It sets the object name.
  void setName(std::string name) 
  {
    name_ = name; 
  } 
  
  // It returns the object name.
  std::string name() const
  {
    return name_;
  }

  // It returns the object information.
  std::string dump() const
  { 
    return ("Object: " + name_  + " (" + std::to_string(id_) + ")"); 
  }
    
 private:
  // The object unique ID
  oid_t id_;
  // The object name
  std::string name_;
};


#endif // ULYSSES_KERNEL__OBJECTINFO_H_
