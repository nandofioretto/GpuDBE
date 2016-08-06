#ifndef ULYSSES__KERNEL__GLOBALS_H_
#define ULYSSES__KERNEL__GLOBALS_H_

#include <cassert>
#include <iostream>
#include <limits>       // std::numeric_limits
#include <map>
#include <string>
#include <memory>
#include <vector>


class DCOPinstance;
class Statistics;

// The objective function cost type
typedef int    cost_t;
typedef cost_t util_t;
// The objects ID type 
typedef int oid_t;
typedef std::vector<cost_t> util_table_t;


// A macro to disallow the copy constructor and operator= functions.
// It should be used in the private declarations for a class.
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&);               \
  void operator=(const TypeName&)


// A macro to attach a message to the assert command.
#ifndef NDEBUG
# define ASSERT(condition, message) \
  do {									\
    if (! (condition)) {						\
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__	\
		<< " line " << __LINE__ << ": " << message << std::endl; \
	std::exit(EXIT_FAILURE);					\
    }									\
  } while (false)
#else
# define \
  ASSERT(condition, message) do { } while (false)
#endif

// A macro to print a worning message when a condition is not satisfied
# define WARNING(condition, message) \
  do {									\
    if (! (condition)) {						\
      std::cerr << "Warning: " << message << std::endl;	      		\
    }									\
  } while (false)


// General Information about the problem instance to be solved.
class DCOPinfo{
 public: 
   enum optType{kMaximize, kMinimize};
   static optType optimization; // default
   static bool maximize() { return optimization == kMaximize; }
   static bool minimize() { return optimization == kMinimize; }
  static std::vector<int> world_probs;
};

extern DCOPinstance* g_dcop;

// Constants 
class Constants {
public:
  // NaN should not be a cost_t but an int
  static constexpr int NaN = std::numeric_limits<int>::max() - 1;
  static constexpr cost_t infinity = -123456;
  static constexpr cost_t sat = 0;
  static constexpr cost_t unsat = -123456;
  static constexpr oid_t nullid = std::numeric_limits<oid_t>::max();

  // move this to u_math:: namespace
  static constexpr bool isFinite(cost_t c) { return c != infinity && c != -infinity; } 
  static constexpr bool isSat(cost_t c) { return c != unsat; }
  // renmae isFinite into isfinite(c)
  //static constexpr bool isnan(cost_c c) {return a == NaN; }

  static cost_t worstvalue;
  static cost_t bestvalue;
  // static bool maximize; // referred to the DCOP instance 
  // static bool minimize; // referred to the DCOP instance
private:
  Constants(){ }
  DISALLOW_COPY_AND_ASSIGN(Constants);
};


class ConstraintCatalog{
public:
  enum constrName{
    XdivYeqZ,
    XeqC,
    XeqY,
    XexpYeqZ,
    XgtC,
    XgteqC,
    XgtY,
    XgteqY,
    XltC,
    XlteqC,
    XltY,
    XlteqY,
    XmodYeqZ,
    XmulCeqZ,
    XmulYeqC,
    XmulYeqZ,
    XneqC,
    XneqY,
    XplusCeqZ,
    XplusClteqZ,
    XplusYeqC,
    XplusYeqZ,
    XplusYgtC,
    XplusYlteqZ,
    XplusYplusCeqZ,
    XplusYplusQeqZ,
    XplusYplusQgtC
  };

  static constrName get(std::string name) 
  { 
  //   int c = mapName_.count(name);
  //   ASSERT( c>0, "Constraint " << name 
  // 	    << " not registered in Ulysses catalog"); 
    return mapName_[ name ]; 
  }

  static void registerConstraint(std::string name, constrName c)
  {
    mapName_[ name ] = c;
  }

  // Maps constraint names to constrName types 
  static std::map<std::string, constrName> mapName_;

private:
  ConstraintCatalog() { }
  DISALLOW_COPY_AND_ASSIGN(ConstraintCatalog);
};


#endif // ULYSSES__KERNEL__GLOBALS_H_

