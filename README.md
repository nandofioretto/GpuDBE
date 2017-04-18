# GpuBE 

This repository contains the sources of a GPU Distributed Bucket Elimination (GpuBE). For details, please refer to our paper:

Ferdinando Fioretto, Tiep Le, Enrico Pontelli, William Yeoh, Tran Cao Son
[Exploiting GPUs in Solving (Distributed) Constraint Optimization Problems with Dynamic Programming](http://link.springer.com/chapter/10.1007%2F978-3-319-23219-5_9), In proceeding of CP 2015.


### Compiling 
GpuDBE has been tested on MAC-OS-X and Linux operating systems. Prior compiling, you need to set up the following parameters in the Makefile:

	DEVICE_CC		(The device compute capability of the GPU)
	CUDA_PATH   	(The path to the CUDA libraries) 
	cudaDBE_PATH	(The path to GpuBE)

Then, from the GpuBE folder execute:

	make 


### Execution
	./cudaDBE fileIn.xml [device=]devId [root=]rootId [memlimitMB=|memlimitGB=]mem

	The parameters in brackets are optionals.
	- fileIn.xml is the input file, complete with its path. cudaDBE accepts FRODO style xml file formats, with the following restrictions:
	- a problem with N variables must have N agents, each of which controls one variable.
	- agents ID are in the range [0, N-1], in a N agents problem.   
	- the framework has been tested for maximization problems only, and handles exclusively binary constraints.
	- devId  is the GPU device ID
	- rootId is the ID of the agent selected as the pseudo-tree root. 
	- mem    is the maximal amount of global memory (in MB or GB) allowed for storage onto the device. 
         (this paramter can also be set through the preference.hh file with the maxDevMemory option).


### Preferences
	Print and Report preferences (boolean)
	- verbose:        Verbose report
	- verboseDevInit: Reports device intialization procedures   
	- silent:         Silent execution          

	// CUDA Memory preferences
	// default: when singleAgent is not select (i.e., multi-agent mode), we suggest to use the option usePinned.
	// default: when singleAgent is selected we suggest to *disable* the use of usePinned memory.
	- usePinnedMemory:   If enabled allows to concurrent CPU/GPU execution. It does so by breaking the UTIL tables in
                      smaller chuncks (whose size is defined by the 'streamSizeMB' option) and by concurrently
                      transferring and operating on the UTIL tables   
	- streamSizeMB:      Defines the maximum stream size to be transfered (in host-device transactions) 
	- singleAgent:       This option defines the single agent use (classical Bucket Elimination for COPs) vs
                      the multi-agent use (DPOP for DCOPs) 

	// Host and Device Memory Limits (in bytes)
	- maxHostMemory:        Defines the maximum memory allowed on the host
	- maxDevMemory:         Defines the maximum global memory allowed on the device. Leave '0' for using the 
                         maximal device global memory capabilities.



## References
- [1] Ferdinando Fioretto and Tiep Le and William Yeoh and Enrico Pontelli and Tran Cao Son. "[Exploiting {GPU}s in Solving (Distributed) Constraint Optimization Problems with Dynamic Programming](http://www-personal.umich.edu/~fioretto/files/papers/cp15.pdf)". In Proceedings of the International Conference on Principles and Practice of Constraint Programming (CP), pages 121-139, 2017. 


## Citing
```
@inproceedings{fioretto:CP-15,
  author    = "Ferdinando Fioretto and Tiep Le and William Yeoh and Enrico Pontelli and Tran Cao Son",
  title     = "Exploiting {GPU}s in Solving (Distributed) Constraint Optimization Problems with Dynamic Programming",
  booktitle = "Proceedings of the International Conference on Principles and Practice of Constraint Programming {(CP)}",
  year      = "2015",
  pages     = "121--139",
  doi       = "10.1007/978-3-319-23219-5_9"
}
```

## Contacts
- Ferdinando Fioretto: fioretto@umich.edu
- William Yeoh: wyeoh@cs.nmsu.edu
- Enrico Pontelli: epontell@cs.nmsu.edu
