#include <map>

#include "Kernel/globals.hh"
#include "preferences.hh"

constexpr int Constants::NaN;
constexpr oid_t Constants::nullid;
constexpr cost_t Constants::infinity;
constexpr cost_t Constants::unsat;

// CUDA-DBE Preferences
constexpr bool preferences::verbose;
constexpr bool preferences::verboseDevInit;
constexpr bool preferences::silent;
constexpr bool preferences::usePinnedMemory;
constexpr bool preferences::singleAgent;
constexpr float preferences::streamSizeMB;
constexpr float preferences::maxHostMemory;
constexpr float preferences::maxDevMemory;


cost_t Constants::worstvalue = 0;
cost_t Constants::bestvalue = 0;
DCOPinfo::optType DCOPinfo::optimization = DCOPinfo::kMinimize; // default
std::vector<int> DCOPinfo::world_probs;

DCOPinstance* g_dcop = nullptr;

std::map< std::string, ConstraintCatalog::constrName > 
ConstraintCatalog::mapName_ = {
  {"XeqY", ConstraintCatalog::XeqY},
  {"int_EQ", ConstraintCatalog::XeqY}
};
