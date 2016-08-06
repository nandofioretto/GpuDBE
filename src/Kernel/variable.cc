#include "Kernel/globals.hh"
#include "Kernel/variable.hh"
#include "Kernel/constraint.hh"
#include "Utilities/constraint-utils.hh"
#include "Utilities/utils.hh"

Variable::Variable()
  :  weight_(-1), nb_int_hard_c_(0), nb_int_soft_c_(0),
     nb_ext_hard_c_(0), nb_ext_soft_c_(0)
{ }


size_t Variable::nbConstraints(Constraint::ConstraintType type) const
{
  switch (type)
    {
    case Constraint::kIntHard:
      return nb_int_hard_c_;
    case Constraint::kIntSoft:
      return nb_int_soft_c_;
    case Constraint::kExtHard:
      return nb_ext_hard_c_;
    case Constraint::kExtSoft:
      return nb_ext_soft_c_;
    case Constraint::kAny:      
    default:
      return nb_int_hard_c_ + nb_int_soft_c_ + nb_ext_hard_c_ + nb_ext_soft_c_;
    }
}


// Sorts the constraints so that they appear in the following order:
// 1. intensional hard constraints
// 2. intensional soft constraints
// 3. extensional hard constraints
// 4. extensional soft constraints
void Variable::registerConstraint( Constraint& c )
{
  std::vector<Constraint*>::iterator it = constraints_.begin();
  
  if(Utils::find(c.id(), ConstraintUtils::getID(constraints_)))
    return;
  
  int off = 0;
  switch( c.type() )
    {
    case Constraint::kAny:
      ASSERT(false, "Cannot register a constraint with no specific type");
    case Constraint::kIntHard:
      off = nb_int_hard_c_;
      constraints_.insert( it + off, &c );
      nb_int_hard_c_++;
      break;
    case Constraint::kIntSoft:
      off = nb_int_hard_c_ + nb_int_soft_c_;
      constraints_.insert( it + off, &c );
      nb_int_soft_c_++;
      break;
    case Constraint::kExtHard:
      off = nb_int_hard_c_ + nb_int_soft_c_ + nb_ext_hard_c_;
      constraints_.insert( it + off, &c );
      nb_ext_hard_c_++;
      break;
    case Constraint::kExtSoft:
      off = nb_int_hard_c_ + nb_int_soft_c_ + nb_ext_hard_c_ + nb_ext_soft_c_;
      constraints_.insert( it + off, &c );
      nb_ext_soft_c_++;
      break;
    }
}

