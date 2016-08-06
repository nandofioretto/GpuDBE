#include <algorithm>
#include <set>
#include <vector>
#include <cmath>

#include "Kernel/globals.hh"
#include "Utilities/constraint-utils.hh"

#include "Kernel/int-variable.hh"
#include "Kernel/constraint.hh"
#include "Kernel/agent.hh"
#include "Problem/dcop-instance.hh"

using namespace std;

vector<oid_t> ConstraintUtils::involvingAny(const vector<oid_t> vars)
{
  set<oid_t> constr;
  for (IntVariable* v : g_dcop->variables(vars))
    for (Constraint* c : v->constraints())
      constr.insert(c->id());
  return vector<oid_t>(constr.begin(), constr.end());
}

  
vector<oid_t> ConstraintUtils::involvingExclusively
(const vector<oid_t> vars, oid_t agent)
{
  vector<oid_t> res;
  vector<IntVariable*> _vars = g_dcop->variables(vars);
  vector<oid_t> cons;
  bool insert;
  
  if (agent == -1) 
    cons = involvingAny(vars);
  else
    cons = g_dcop->agent(agent).constraintIDs();

  for (oid_t cid : cons)
  {
    insert = true;
    Constraint& c = g_dcop->constraint(cid);
    for (oid_t vscope : c.scopeIds())
    {
      // scope var not in var set -- do not insert
      if (find(vars.begin(), vars.end(), vscope) == vars.end())
        { insert = false; break;}
    }
    if(insert) res.push_back(cid);
  }
  
  return res;
}


vector<oid_t> ConstraintUtils::extractScope(const vector<oid_t> constraints)
{
  set<oid_t> res_set;
  for (oid_t cid : constraints)
  {
    Constraint& c = g_dcop->constraint(cid);
    res_set.insert(c.scopeIds().begin(), c.scopeIds().end());
  }
  return vector<oid_t>(res_set.begin(), res_set.end());
}


vector<oid_t> ConstraintUtils::extractOwnedByAny
  (const vector<oid_t> constraints, const vector<oid_t> owners)
{
  vector<oid_t> res;
  for (oid_t cid : constraints)
  {
    Constraint& c = g_dcop->constraint(cid);
    for (IntVariable* v : c.scope())
      if (find(owners.begin(), owners.end(), v->ownerId())
        != owners.end()) // is owned by owner
          { res.push_back( cid ); break; }
  }
  return res;
}


vector<oid_t> ConstraintUtils::extractOwnedByAll
  (const vector<oid_t> constraints, const vector<oid_t> owners)
{
  vector<oid_t> res;
  for (oid_t cid : constraints)
  {
    bool failed = false;
    Constraint& c = g_dcop->constraint(cid);
    for (IntVariable* v : c.scope())
      if (find(owners.begin(), owners.end(), v->ownerId())
        == owners.end()) // is not owned by any owner
          { failed=true; break; }
    if (!failed) res.push_back( cid );
  }
  return res;
}


std::vector<oid_t> ConstraintUtils::getID(std::vector<Constraint*> objects)
{
  std::vector<oid_t> res(objects.size());
  for(int i=0; i<objects.size(); ++i)
    res[ i ] = objects[ i ]->id();
  return res;
}
