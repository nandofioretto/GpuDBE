=================
Execution
================
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


=================
Preferences
================
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

 
