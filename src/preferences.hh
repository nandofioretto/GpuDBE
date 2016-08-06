#ifndef CUDA_DBE_PREFERENCES_H_
#define CUDA_DBE_PREFERENCES_H_

class preferences {
	public:

	// Print and Report preferences
	static constexpr bool verbose = false;
	static constexpr bool verboseDevInit = false;
	static constexpr bool silent = false;

	// CUDA Memory preferences
	// default: when singleAgent=F, usePinned=T
	// default: when singleAgent=T, usePinned=F
	static constexpr bool usePinnedMemory = false;
	static constexpr bool singleAgent     = true;
	static constexpr float streamSizeMB = 25;

	static constexpr bool importPseudoTree = true;

	// Host and Device Memory Limits (in bytes)
	static constexpr float maxHostMemory = 120/*GB*/ * 1e+9;
    static constexpr float maxDevMemory  = 0 /*GB*/ * 1e+9; // 0 for unbounded
};

#endif
