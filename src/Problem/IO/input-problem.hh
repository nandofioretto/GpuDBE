#ifndef ULYSSES_PROBLEM__IO__INPUT_PROBLEM_H_
#define ULYSSES_PROBLEM__IO__INPUT_PROBLEM_H_

#include <string>
#include <iostream>

#include "Problem/IO/input.hh"

class InputProblem : public Input
{
public:
  InputProblem(int argc, char* argv[])
  {
    // checkParams(argc, argv);
    p_filename = argv[ 1 ];
  }

  virtual ~InputProblem()
  { }

  virtual std::string dump()
  {
    std::string res = "Filename                : " + p_filename;
    return res;
  }
  
};
  
#endif // ULYSSES_PROBLEM__IO__INPUT_PROBLEM_H_
