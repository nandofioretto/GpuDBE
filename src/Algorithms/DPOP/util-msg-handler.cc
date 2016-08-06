#include <memory>
#include <vector>
#include <string>

#include "Algorithms/DPOP/util-msg-handler.hh"
#include "Algorithms/DPOP/util-msg.hh"
#include "Communication/scheduler.hh"
#include "Kernel/globals.hh"
#include "Kernel/agent.hh"
#include "Utilities/utils.hh"
#include "Utilities/statistics.hh"
#include "GPU/cuda_dpop_state.hh"

using namespace std;

UtilMsgHandler::UtilMsgHandler(Agent& a)
  : MessageHandler(a)
{
  p_outgoing = UtilMsg::uptr(new UtilMsg);
}


UtilMsgHandler::~UtilMsgHandler()
{ }


void UtilMsgHandler::processIncoming()
{
  if (owner().openMailbox().isEmpty("UTIL"))
    return;

  for (Message::sptr r : owner().openMailbox().readAll("UTIL"))
  {
    UtilMsg::sptr s = dynamic_pointer_cast<UtilMsg>(r);
    if (not Utils::find(s, p_received))
      p_received.push_back(std::move(s));
  }
}

void UtilMsgHandler::setMsgContent(CUDA::DPOPstate& state) {
  p_outgoing->setContent( state.getUtilTablePtr(), state.getUtilTableRowsAfterProj());
}

void UtilMsgHandler::prepareOutgoing()
{ 
  p_outgoing->setSource(owner().id());
}


void UtilMsgHandler::send(oid_t dest_id)
{ 
  ASSERT(dest_id != Constants::nullid, "Trying to send a message with empty scheduler.");

  UtilMsg::sptr to_send(p_outgoing->clone());
  to_send->setDestination(dest_id);

  owner().openMailbox().send(to_send);
  Scheduler::FIFOinsert(to_send->destination());
}


bool UtilMsgHandler::recvAllMessages()
{
  return (p_received.size() == owner().ptNode().nbChildren());
}


void UtilMsgHandler::initialize()
//(util_table_t* table_ptr, Codec::sptr rows_code)
{
  // p_outgoing->setUtilTable( table_ptr );
  // p_outgoing->setCodec( rows_code );

  // oid_t xi = owner().boundaryVariableAt(0).id();
  
  // std::vector<oid_t> aux_vars = Utils::concat(xi, rows_code->variables());
  
  // p_query.resize(p_received.size());
  // p_msgvars2validx.resize( p_received.size() );

  // for (int i=0; i<p_received.size(); ++i)
  // {
  //   UtilMsg& msg = *p_received[ i ];
  //   std::vector<oid_t> msg_vars = msg.getVariables();
  //   p_query[ i ].resize( msg_vars.size() );
  //   p_msgvars2validx[ i ].resize( msg_vars.size() );

  //   // Construct the mapping UTIL values <-> solution values
  //   for (int j = 0; j < msg_vars.size(); ++j)
  //   {
  //     int idx = Utils::findIdx( aux_vars, msg_vars[ j ] );
  //     ASSERT(idx >= 0, 
  //     "Error in initializing the auxiliary DS for the util optimization phase");
  //     p_msgvars2validx[ i ][ j ] = idx;
  //   }
  // }

}


// b_e_vals = concatenation of b_values with e_values
cost_t UtilMsgHandler::msgCosts(std::vector<int> b_e_values)
{
  // cost_t sum_cost_ = 0;
  
  // for (int i=0; i<p_received.size(); ++i)
  // {
  //   UtilMsg& msg = *p_received[ i ];
  //   for (int j = 0; j < msg.nbVars(); ++j) 
  //   {
  //     int idx = p_msgvars2validx[ i ][ j ];
  //     p_query[ i ][ j ] = b_e_values[ idx ];
  //   }
  //   cost_t u = msg.getUtil( p_query[ i ], wid );
  //   if (!Constants::isFinite( u ))
  //     return u;
  //   sum_cost_ += u;
  // }

  // return sum_cost_;
  return 0;
}
