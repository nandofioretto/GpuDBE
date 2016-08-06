#ifndef ULYSSES_UTILITIES__UTILS_H_
#define ULYSSES_UTILITIES__UTILS_H_

#include <vector>
#include <set>
#include <algorithm>
#include <string>
#include <cmath>
#include <sstream>
#include <cstdarg>
#include <iostream>

#include "Kernel/globals.hh"
#include "Kernel/object-info.hh"
#include "Kernel/int-variable.hh"
#include "Kernel/constraint.hh"
#include "Kernel/agent.hh"

namespace Utils
{
  // Merges Two vectors: It returs the union of the sets given as a parameters.
  template<typename T>
  std::vector<T> merge(const std::vector<T> _A, const std::vector<T> _B)
  {
    std::vector<T> A = _A, B = _B; // copy 

    std::vector<T> res(A.size() + B.size());
    std::sort(A.begin(), A.end());
    std::sort(B.begin(), B.end());
    typename std::vector<T>::iterator it;
    it = std::set_union(A.begin(), A.end(), B.begin(), B.end(), res.begin());
    res.resize(it-res.begin());
    return res;
  }

  // Merges Two vectors: It returs the union of the sets given as a parameters
  // in the first set given as parameter.
  template<typename T>
  void merge_emplace(std::vector<T>& A, const std::vector<T> B)
  {
    std::vector<T> res = merge(A,B);
    A.swap(res);
  }

 // Intersects two vectors: It returs the intersection of the sets given as 
  // a parameters.
  template<typename T>
  std::vector<T> intersect(const std::vector<T> _A, const std::vector<T> _B)
  {
    std::vector<T> A = _A, B = _B; // copy 

    std::vector<T> res( std::max(A.size(), B.size()));
    std::sort(A.begin(), A.end());
    std::sort(B.begin(), B.end());
    typename std::vector<T>::iterator it;
    it = std::set_intersection(A.begin(), A.end(), B.begin(), B.end(), res.begin());
    res.resize(it-res.begin());
    return res;
  }

  // Intersects two vectors: It returs the intersection of the sets given as 
  // a parameters.
  // @return The result of the intersection is saved in the vector A. 
  template<typename T>
  void intersect_emplace(std::vector<T>& A, const std::vector<T> B)
  {
    std::vector<T> res = intersect(A,B);
    A.swap(res);
  }

  // It excludes the vector B from A: 
  // It Returns the set difference of the two sets given as a parameters.
  template<typename T>
  std::vector<T> exclude(std::vector<T> _B, std::vector<T> _A)
  {
    std::vector<T> A = _A, B = _B; // copy 

    std::vector<T> res(A.size());
    std::sort(A.begin(), A.end());
    std::sort(B.begin(), B.end());
    typename std::vector<T>::iterator it;
    it = std::set_difference(A.begin(), A.end(), B.begin(), B.end(), res.begin());
    res.resize(it-res.begin());
    return res;
  }

  // It excludes the vector B from A: 
  // It Returns the set difference of the two sets given as a parameters
  // in second set given as a parameter.
  template<typename T>
  void exclude_emplace(const std::vector<T> B, std::vector<T>& A)
  {
    std::vector<T> res = exclude(B,A);
    A.swap(res);
  }

  // It returns the concatenation of the vectors A and B.
  template<typename T>
  std::vector<T> concat(T A, T B)
  {
    std::vector<T> out( 2 );
    out[ 0 ] = A; out[ 1 ] = B;
    return out;
  }

  // It returns the concatenation of the vectors A and B.
  template<typename T>
  std::vector<T> concat(const std::vector<T> A, T B)
  {
    std::vector<T> out( A.size() + 1 );
    int j=0;
    for (int i=0; i<A.size(); ++i) out[ j++ ] = A[ i ];
    out[ j ] = B;
    return out;
  }

  // It returns the concatenation of the vectors A and B.
  template<typename T>
  std::vector<T> concat(T A, const std::vector<T> B)
  {
    std::vector<T> out( B.size() + 1 );
    out[ 0 ] = A;
    for (int i=0; i<B.size(); ++i) out[ i+1 ] = B[ i ];
    return out;
  }
 
  // It returns the concatenation of the vectors A and B.
  template<typename T>
  std::vector<T> concat(const std::vector<T> A, const std::vector<T> B)
  {
    std::vector<T> out( A.size() + B.size() );
    int j=0;
    for (int i=0; i<A.size(); ++i) out[ j++ ] = A[ i ];
    for (int i=0; i<B.size(); ++i) out[ j++ ] = B[ i ];
    return out;
  }

  // It concatenates A and B into 'out'.
  // The vector out should be of size A+B
  template<typename T>
  void concat_emplace(std::vector<T>& out, const std::vector<T> A, const std::vector<T> B)
  {
    ASSERT( out.size() == A.size() + B.size(), 
	    "Error in concatenating arryas of different sizes" << A.size() 
	    << " and " << B.size() << " with " << out.size());

    int j=0;
    for (int i=0; i<A.size(); ++i) out[ j++ ] = A[ i ];
    for (int i=0; i<B.size(); ++i) out[ j++ ] = B[ i ];
  }

  template<typename T>
  std::vector<std::vector<T> > transpose(std::vector<T> vec)
  {
    std::vector<std::vector<T> > res(vec.size());
    for (int i=0; i<vec.size(); ++i)
      res[ i ].push_back(vec[ i ]);
    return res;
  }

  template<typename T>
  void cartesian_product(std::vector<std::vector<T> >& CP, std::vector<T> A)
  {
    if (CP.empty())
      { CP = Utils::transpose(A); return; }
    if (A.empty())
      { return; }
    
    std::vector<std::vector<T> > res;
    for(std::vector<T> vec : CP)
    {
      int e = vec.size();
      std::vector<T> new_vec(e+1);
      std::copy(vec.begin(), vec.end(), new_vec.begin());
      for(T t : A) {
        new_vec[ e ] = t;
        res.push_back( new_vec );
      }
    }
    CP.swap(res);
  }

  // It returns the description of the array content given as a parameter.
  template<typename T>
  std::string dump(const std::vector<T> array)
  {
    std::string res = "{";
    for (int i=0; i<array.size(); ++i) {
      res += std::to_string(array[i]);
      if( i < array.size()-1) res += ", ";
    }
    res += "}";
    return res;
  }

  template<typename T>
  std::string dump(const std::vector<std::vector<T> > matrix)
  {
    std::string res;
    for (int i=0; i<matrix.size(); ++i) {
      for (int j=0; j<matrix[i].size(); ++j) {
        res += std::to_string(matrix[i][j]);
        if( j < matrix[ i ].size()-1) res += ", ";   
      }
      res+="\n";
    }
    return res;
  }

  // It returns the description of the array content given as a parameter.
  template<typename T, typename U>
  std::string dump(const std::vector<std::pair<T,U> > array)
  {
    std::string res = "{";
    for (int i=0; i<array.size(); ++i) {
      res += "(" + std::to_string(array[i].first) + ", "
	+ std::to_string(array[i].second) + ")";
      if( i < array.size()-1) res += ", ";
    }
    res += "}";
    return res;
  }

  // It returns the description of the array content given as a parameter.
  template<typename T, typename U>
  std::vector<std::pair<T,U> > make_pairs
  (const std::vector<T> array_first, const std::vector<U> array_second)
  {
    ASSERT( array_first.size() == array_second.size(), "Error in making pairs");
    std::vector<std::pair<T,U> > res;
    for (int i=0; i<array_first.size(); ++i) {
      res.push_back(std::make_pair(array_first[ i ], array_second[ i ]));
    }
    return res;
  }

  template <typename T>
  std::string numberToString ( T Number )
  {
    std::stringstream ss;
    ss << Number;
    return ss.str();
  }

  template<typename T>
  T max(const std::vector<T> array)
  {
    int idx = 0;
    for (int i=1; i<array.size(); ++i)
      if (array[ i ] > array[ idx ]) idx = i;
    return array[ idx ];
  }

  template<typename T>
  T min(const std::vector<T> array)
  {
    int idx = 0;
    for (int i=1; i<array.size(); ++i)
      if (array[ i ] < array[ idx ]) idx = i;
    return array[ idx ];
  }

  template<typename T>
  bool find(T elem, const std::vector<T> array)
  { 
    return (std::find(array.begin(), array.end(), elem) != array.end());
  }

  template<typename T>
  int findIdx(const std::vector<T> A, T query)
  {
    for (int i=0; i<A.size(); ++i)
      if (A[i] == query)
	return i;
    return -1;
  }

  template<typename T, typename U>
  bool findFirst(const std::vector<std::pair<T,U> > A, T query)
  {
    for (int i=0; i<A.size(); ++i)
      if (A[i].first == query)
	return true;
    return false;
  }

  template<typename T, typename U>
  int findIdxFirst(const std::vector<std::pair<T,U> > A, T query)
  {
    for (int i=0; i<A.size(); ++i)
      if (A[i].first == query)
	return i;
    return -1;
  }

  template<typename T, typename U>
  bool findSecond(const std::vector<std::pair<T,U> > A, T query)
  {
    for (int i=0; i<A.size(); ++i)
      if (A[i].second == query)
	return true;
    return false;
  }

  template<typename T, typename U>
  int findIdxSecond(const std::vector<std::pair<T,U> > A, T query)
  {
    for (int i=0; i<A.size(); ++i)
      if (A[i].second == query)
	return i;
    return -1;
  }

  template<typename T, typename U>
  std::vector<T> extractFirst(const std::vector<std::pair<T,U> > array)
  {
    std::vector<T> res(array.size());
    for (int i=0; i<array.size(); ++i)
      res[ i ] = array[ i ].first;
    return res;
  }

  template<typename T, typename U>
  std::vector<U> extractSecond(const std::vector<std::pair<T,U> > array)
  {
    std::vector<U> res(array.size());
    for (int i=0; i<array.size(); ++i)
      res[ i ] = array[ i ].second;
    return res;
  }

  template<typename T>
  void findAndRemove(T elem, std::vector<T>& array)
  {
    typename std::vector<T>::iterator it =
      std::find(array.begin(), array.end(), elem);
    if( it != array.end())
      array.erase( it );
  }

  template<typename T>
  void insertOnce(T elem, std::vector<T>& array)
  {
    if(!Utils::find(elem, array))
      array.push_back(elem);
  }

  // Generates the next combination of k elements of a vector of size n >= k.  
  template <typename Iterator>
  bool next_combination(const Iterator first, Iterator k, const Iterator last)
  {
     /* Credits: Thomas Draper */
     if ((first == last) || (first == k) || (last == k))
        return false;
     Iterator itr1 = first;
     Iterator itr2 = last;
     ++itr1;
     if (last == itr1)
        return false;
     itr1 = last;
     --itr1;
     itr1 = k;
     --itr2;
     while (first != itr1)
     {
        if (*--itr1 < *itr2)
        {
           Iterator j = k;
           while (!(*itr1 < *j)) ++j;
           std::iter_swap(itr1,j);
           ++itr1;
           ++j;
           itr2 = k;
           std::rotate(itr1,j,last);
           while (last != j)
           {
              ++j;
              ++itr2;
           }
           std::rotate(k,itr2,last);
           return true;
        }
     }
     std::rotate(first,k,last);
     return false;
  }
  
  // It sums two cost values.
  // The costs are either finite number, or (+/-)Constants::infinity
  inline cost_t sum(cost_t a, cost_t b)
  {
    if( Constants::isFinite(a) && Constants::isFinite(b) ) return a + b;
    if( !Constants::isFinite(a) ) return a;
    return b;
  }

  inline void sum_emplace(cost_t& a, cost_t b)
  {
    a = Utils::sum(a,b);
  }

  // Returns whether a is better than b 
  inline bool isBetter(cost_t a, cost_t b)
  {
    // if (!Constants::isFinite(b)) return true;
    if (!Constants::isFinite(a)) return false;
    return (DCOPinfo::maximize()) ? (a > b) : (a < b);
  } 

  // Returns whether a is worse than b
  inline bool isWorse(cost_t a, cost_t b)
  {
    if (!Constants::isFinite(b)) return false;
    return (DCOPinfo::maximize()) ?  (a < b) : (a > b);
  }

  // Returns the worse bound between a and b.
  inline cost_t getWorst(cost_t a, cost_t b)
  {
    return (isWorse(a, b)) ? a : b;
  }
    
  // Returns the best bound between a and b.
  inline cost_t getBest(cost_t a, cost_t b)
  {
    return (isBetter(a, b)) ? a : b;    
  }

  inline std::vector<oid_t> getID(std::vector<ObjectInfo*> objects)
  {
    std::vector<oid_t> res(objects.size());
    for(int i=0; i<objects.size(); ++i)
      res[ i ] = objects[ i ]->id();
    return res;
  }
  
  // trim from start
  inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
  }

  // trim from end
  inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
  }

  // trim from both ends
  inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
  }
  
  inline int stringToInt(const std::string &Text)
  {                               //character array as argument
    std::stringstream ss(Text);
    int result;
    return ss >> result ? result : 0;
  }

  inline double stringToDouble(const std::string &Text)
  {                               //character array as argument
    std::stringstream ss(Text);
    double result;
    return ss >> result ? result : 0;
  }
  
  inline std::vector<std::string> stringSplit(const std::string &Text)
  {
    std::vector<std::string> res;
    std::string param;
    std::stringstream ss(Text);
    while(ss.good()) {
      ss >> param; 
      res.push_back(param);
    }
    return res;
  }
  
};

#endif // ULYSSES_UTILITIES__UTILS_H_
