//#define ULYSSES_GPU_DEV_GLOBALS_H
#include "GPU/gpu_globals.hh"

__device__ int gdev_nb_constraints;
__device__ int gdev_nb_variables;
__device__ int gdev_nb_agents;
__device__ int gdev_dom_size;
__device__ int gdev_opt_type; // The optimization type (maximize / minimize [default])
__device__ int gdev_infty;    // The infinity cost

__device__ struct dev_class_Constraint* gdev_Constraints;
__device__ struct dev_class_Variable*  gdev_Variables;
__device__ struct dev_class_DPOP_Agent*  gdev_DPOP_Agents;

size_t CUDA::used_memory::global = 0;
size_t CUDA::used_memory::constant = 0;
size_t CUDA::used_memory::texture = 0;
size_t CUDA::used_memory::shared = 0;
size_t CUDA::info::global_memory = 0;
size_t CUDA::info::shared_memory = 0;
size_t CUDA::info::max_dim_grid = 0;

cudaStream_t* agtStream;


//host_class_DevMirror*  host_class_DevMirrors;

// DEVICE: Efficient computation of the integer pow(base, exp)
inline __device__ int ipow(int base, int exp)
{
  int result = 1;
  while (exp)
  {
    if (exp & 1)
      result *= base;
    exp >>= 1;
    base *= base;
  }
  return result;
}

// HOST: computes the index in the UTIL table associated to a given tuple T
//       given the domain size d and the constraint arity a.
size_t g_hash_util( int d, int a, int T[])
{
  size_t ofs=0; size_t pos_i=0; int i;
  for(i=0; i<a; i++)
  {
    pos_i = std::pow(d, (double)(a-i-1));
    ofs  += T[i]*pos_i;
  }
  return ofs;
}

// HOST: computes the tuple associated to a given index in a UTIL table
void g_hash_tuple(int* T, int d, int a, size_t idx )
{
  for( int i = a-1; i >= 0; i--)
  {      
    T[i] = idx % d;
    idx /= d;
  }
}

// DEVICE: device compuation of the hash_util
inline __device__ int cuda_hash_util( int d, int scope[], int a, int T[])
{
  int ofs=0; int pos_i=0; int i;
  for(i=0; i<a; i++)
  {
    pos_i = ipow(d, a-i-1);
    ofs  += T[scope[i]]*pos_i; // error here;
  }
  return ofs;
}

// var, val are the variable and value to be sobstiutited in T
inline __device__ int cuda_hash_util2( int d, int scope[], int a, int T[],
				   int i_sobst, int i_val)
{
  int ofs=0; int pos_i=0; int i;
  for(i=0; i<a; i++)
  {
    pos_i = ipow(d, a-(i+1));
    ofs  += (i_sobst == scope[i]) ?
      i_val * pos_i 
      : T[scope[i]] * pos_i;
  }
  return ofs;
}

inline __device__ void cuda_hash_tuple(int* T, int d, int a, size_t idx )
{
  for( int i = a-1; i >= 0; i--)
  {      
    T[i] = idx % d;
    idx /= d;
  }
}
