/* Permutations 
 * This class is used to generate the describe the possible node 
 * permutation of K nodes for a network of N nodes.
 */

#ifndef PERMUTATIONS_H
#define PERMUTATIONS_H

class Permutations 
{
 private:
  int n, k;
  int* permutations;
  size_t _size;
  int* T;
  void generate();
  int level;
  size_t perm_counter;

 public:
  Permutations(int n, int k);
  ~Permutations();

  int* get_permutations();
  size_t size() const;
  int get_k() const;
  int get_n() const;
  void dump();
};

#endif
