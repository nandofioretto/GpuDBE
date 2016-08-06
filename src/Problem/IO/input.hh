#ifndef ULYSSES_PROBLEM__IO__INPUT_H_
#define ULYSSES_PROBLEM__IO__INPUT_H_

#include <string>
#include <iostream>

class Input
{
public:
  Input()
  { }

  // It returns the output file
  std::string getFile() const
  { 
    return p_filename;
  }

  // It returns the description of the usage help 
  std::string usage() const
  {
    return"ulysses dcop.xcsp settings.xml";
  }
      
  void checkParams(int argc, char* argv[])
  {
    if(argc < 2) {
      std::cout << "Wrong number of parameters received" << std::endl
        << usage() << std::endl;
      exit(1);
    }
  }
  
  virtual std::string dump() = 0;
  
protected:
  std::string p_filename;
};
  
#endif // ULYSSES_INSTANCE_GENERATOR__IO__INPUT_H_
