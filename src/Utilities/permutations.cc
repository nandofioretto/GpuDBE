#include <iostream>
#include <cmath>
#include "permutations.hh"

using namespace std;

// Creates permutation of k elements among range [0 n-1] 
Permutations::Permutations (int _n, int _k) 
  : n(_n), k(_k), level( 0 ), perm_counter( 0 )
{
  size_t ntuples = pow( (double)_n, (double)_k );
  T = new int[ _k ];
  _size = ntuples;
  permutations = new int[ _size * _k ];
  generate( );
}//-

Permutations::~Permutations()
{
  delete[] permutations;
  delete[] T;
}

void Permutations::generate( )
{
  // static int level = 0;
  // static int perm_counter = 0;
  if( level == k )
  {
    for( int i = 0; i < k; i++ ) {
      permutations[ ( perm_counter * k ) + i ] = T[ i ];
    }
    perm_counter++;
    return;
  }

  for( int i = 0; i < n; i++ )
  {
    T[ level++ ] = i;
    generate();
    level--; 
  }
}//-

int* Permutations::get_permutations()
{
  return permutations;
}

size_t
Permutations::size() const {
  return _size;
}

int 
Permutations::get_n() const {
  return n;
}//-

int 
Permutations::get_k() const {
  return k;
}//-

void 
Permutations::dump() 
{
  for (int i = 0; i < _size; i++) {
    for (int j = 0; j < k; j++) {
      std::cout << permutations[ (i * k) + j ] << " ";
    }
    std::cout << std::endl;
  }
}//-
