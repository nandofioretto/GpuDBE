#ifndef ULYSSES_PROBLEM__IO__XML_H_
#define ULYSSES_PROBLEM__IO__XML_H_

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <rapidxml.hpp>

namespace xml
{

  // It returns the value of the query attribute 'attr_query'  
  inline std::string getParamValue(std::string attr_target, 
    std::string attr_query, std::string query, rapidxml::xml_node<>* node)
  {
    std::string attr_val;
    do{
      attr_val = node->first_attribute(attr_query.c_str())->value();
      if (attr_val.compare(query) == 0 || attr_val.compare("*") == 0)
        return node->first_attribute(attr_target.c_str())->value();
    
      node  = node->next_sibling();
    } while (node);

    return "";
  }

  // Read all values of an xml node. All values are treated as integers.  
  inline std::vector<int> getIntValues(rapidxml::xml_node<>* node)
  {
    std::vector<int> values;
    std::string params = node->value();
    if(!params.empty())
    {
      std::stringstream ss(params);
      int val;
      while(ss.good())
      { ss >> val; values.push_back(val); }      
    }
    return values;
  }


  inline std::vector<std::string> getStringValues(rapidxml::xml_node<>* node)
  {
    std::vector<std::string> values;
    std::string params = node->value();
    if(!params.empty())
    {
      std::stringstream ss(params);
      std::string val;
      while(ss.good())
      { ss >> val; values.push_back(val); }      
    }
    return values;
  }
  
}
  
#endif // ULYSSES_PROBLEM__IO__XML_H_
