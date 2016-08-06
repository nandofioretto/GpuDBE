/**
 * cudaDBE usage:
 * cudaDBE fileName root=(int)[0..|n-1|] memlimit=(int)MB device=(int)
 *
 */
#include <iostream>
#include <string>

#include "Utilities/statistics.hh"
#include "Problem/dcop-model.hh"
#include "Problem/dcop-instance.hh"
#include "Problem/dcop-instance-factory.hh"
#include "Problem/IO/input-settings.hh"
#include "Problem/IO/input-problem.hh"
#include "GPU/cuda_allocator.hh"
#include "GPU/cuda_utils.hh"

#include "preferences.hh"


int main(int argc, char* argv[])
{
  // -------------------------------------- //
  // Register timers
  // -------------------------------------- //
  Statistics::registerTimer("wallclock");
  Statistics::registerTimer("init");
  Statistics::registerTimer("gpu-alloc");


  Statistics::startTimer("wallclock");
  Statistics::startTimer("init");

  InputProblem problem(argc, argv);
  InputSettings settings(argc, argv);

  if(preferences::verbose) {
	  std::cout << problem.dump() << std::endl;
	  std::cout << settings.dump() << std::endl;
  }

  // -------------------------------------- //
  // Retrieve Device Info and Set CUDA Memory Limits
  // -------------------------------------- //
  CUDA::CUDAutils::initializeCUDA(argc, argv);

  // -------------------------------------- //
  // Read Model and create the DCOP
  // -------------------------------------- //
  DCOPmodel model(problem);
  g_dcop = DCOPinstanceFactory::create(model, settings);
  Statistics::stopwatch("init");

  if(preferences::verbose) {
	  std::cout << "Allocating Global data on Device..." << std::flush;
  }

  // -------------------------------------- //
  // Copy data structures onto GPU device
  // -------------------------------------- //
  Statistics::startTimer("gpu-alloc");

  CUDA::Allocator::allocate_data();

  Statistics::stopwatch("gpu-alloc");

  if(preferences::verbose) {
	  std::cout << "done\n";
	  std::cout << "Solving...\n";
  }

  // -------------------------------------- //
  // Start the DCOP
  // -------------------------------------- //
  g_dcop->solve();

  Statistics::stopwatch("wallclock");


  // -------------------------------------- //
  // Prints statistics
  // -------------------------------------- //

  //std::cout << Statistics::dumpCSV() << std::endl;
  std::cout << Statistics::dump() << std::endl;
  std::cout << g_dcop->dump() << std::endl;
  // CUDA::CUDAutils::dump_worlds_solution();

  return 0;
}
