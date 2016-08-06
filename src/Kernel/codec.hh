#ifndef ULYSSES__KERNEL__CODEC_H_
#define ULYSSES__KERNEL__CODEC_H_

#include <vector>

#include "Kernel/globals.hh"
#include "Kernel/variable-factory.hh"
#include "Kernel/int-variable.hh"
#include "Utilities/utils.hh"
#include "Problem/dcop-instance.hh"

class Codec
{
public:
  typedef std::unique_ptr<Codec> uptr;
  typedef std::shared_ptr<Codec> sptr;

  // It constructs the codec for the set of variables given as a parameter.
  // It itnitializes the auxiliary structures for fast value retrival.
  Codec(std::vector<oid_t> vars)
    : variables_(vars), size_(1)
  {
    if( vars.empty() ) return;

    auxvalues_.resize( vars.size(), 0 );
    domsize_.resize( vars.size(), 0 );
    psum_domsize_.resize( vars.size(), 1 );
    offsets_.resize( vars.size(), 0 );
    size_ = 1;

    for (int i = 0; i < vars.size(); ++i)
    {
      domsize_[ i ] = g_dcop->variable(vars[ i ]).size(); 
      offsets_[ i ] = g_dcop->variable(vars[ i ]).min(); 
      size_ *= domsize_[ i ];
      for (int ii = i+1; ii<vars.size(); ++ii)
        psum_domsize_[ i ] *= g_dcop->variable(vars[ ii ]).size();
    }
  }


  Codec(std::vector<IntVariable*> vars)
  {
    if( vars.empty() ) return;
    for (IntVariable* v : vars)
      variables_.push_back( v->id() );

    auxvalues_.resize( vars.size(), 0 );
    domsize_.resize( vars.size(), 0 );
    psum_domsize_.resize( vars.size(), 1 );
    offsets_.resize( vars.size(), 0 );
    size_ = 1;

    for (int i = 0; i < vars.size(); ++i) 
    {
      domsize_[ i ] = vars[ i ]->size(); 
      offsets_[ i ] = vars[ i ]->min(); 
      size_ *= domsize_[ i ];
      for (int ii = i+1; ii<vars.size(); ++ii)
        psum_domsize_[ i ] *= vars[ ii ]->size();
    }
  }

  // Returns the size of the cartesian product of the variables' domain.  
  size_t size() const
  {
    return size_;
  }
  
  // It encodes the sequence of value elements given as a parmater to a large 
  // number.
  // @note: the values given as a paramter correspond to the values of the 
  // variables domains (and not their translated positions)
  // Complexity: O(k) - wiht k = number of variables
  size_t encode(std::vector<int> values)
  {
    size_t pos = 0;
    for (int i=0; i<variables_.size(); ++i)
      pos += ( (values[ i ] - offsets_[ i ]) * psum_domsize_[ i ]);
    return pos;
  }

  // It returns the set of values whose variable given as input, has value 
  // given as input, in the original encoding.
  // NOTE: inefficient!
  std::vector<int> encode_subject_to(oid_t vid, int value)
  {
    std::vector<int> res;
    int idx = Utils::findIdx(variables_, vid);
    for (size_t encoding = 0; encoding < size_; ++encoding)
    {
      if( decode(encoding)[idx] == value )
        res.push_back(encoding);
    }
    return res;
  }

  // It decodes the code given as a parameter into a sequence of values
  // for the variables of the object.
  // @note: it returns the sequence of values of the variables domains 
  // (as opposed at their translated positions which is stored here)
  // Complexity: O(k) - wiht k = number of variables
  std::vector<int>& decode(size_t code)
  {
    for (int i=0; i<variables_.size(); ++i) {
      auxvalues_[ i ] = (int)( code / psum_domsize_[ i ] ) % domsize_[ i ];
      auxvalues_[ i ] += offsets_[ i ];
    }
    return auxvalues_;
  }

  // It returns the list of variables of the object.
  std::vector<oid_t>& variables()
  {
    return variables_;
  }

private:
  // The variables whose values are encoded/decoded.
  std::vector<oid_t> variables_;

  // The domain of the variables in the encoder.
  std::vector<size_t> domsize_;

  // The offsets necessary to translate each domain element to 0.
  // E.g., if a variables has domain [-2, 8] the associated offset 
  // will hold the value '2'.  
  std::vector<int> offsets_;
  
  // The vector of partial sums defined as:
  // psums_domsize_[ i ] = \sum_j={i+1}^n D[ j ]
  // where:
  //  n = number of variables in variables
  //  D[ j ] = the size of the j-th variable's domain
  std::vector<size_t> psum_domsize_;
  
  // The size of the cartesian product of the variables' domain.  
  size_t size_;

  // Auxiliary value combinations used to temporary save the values
  // while decoding. It avoids to allocate a new vector at each decoding.
  std::vector<int> auxvalues_;
};


#endif
