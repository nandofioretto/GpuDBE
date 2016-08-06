# ---------------------------------------------------------------------
# Includes all the files to be compiled.
# To add a new file, modify the make.inc file.
# ---------------------------------------------------------------------
include sources.inc

# ---------------------------------------------------------------------
# Includes MACROs 
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Paths 
# ---------------------------------------------------------------------
# cudaDBE
cudaDBE_PATH="/home/grad12/ffiorett/git/CUDA/cudaDBE/"
LIB_PATH="$(cudaDBE_PATH)/lib"

# Set your Rapid-xml library path here
RAPID_XML_PATH="$(LIB_PATH)/rapidxml-1.13"
# Set your CUDA paths here
CUDA_PATH ="/usr/local/cuda/"
CUDA_SAMPLES_PATH="$(CUDA_PATH)/samples/common/inc"


# ---------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------
H_SOURCES = /src/cudaDBE.cc $(H_HEADERS:%.hh=%.cc)
D_SOURCES = $(D_HEADERS:%.hh=%.cu)
H_OBJECTS = $(H_SOURCES:%.cc=%.o) 
D_OBJECTS = $(D_SOURCES:%.cu=%.o)
OBJECTS   = $(H_OBJECTS) $(D_OBJECTS)


# ---------------------------------------------------------------------
# Compiler options 
# ---------------------------------------------------------------------
UNAME_S := $(shell uname -s)
# MAC-OS-X options
ifeq ($(UNAME_S),Darwin)
  CC = clang++
  NVCC=$(CUDA_PATH)/bin/nvcc 
  DEPEND = -std=c++11 -stdlib=libc++
endif
# LINUX optinos
ifeq ($(UNAME_S),Linux)
  CC = g++
  NVCC=$(CUDA_PATH)/bin/nvcc
  DEPEND = -std=c++11
endif

# Set your device acompute cabability here
DEVICE_CC=35

OPTIONS = -O3 -w -gencode arch=compute_$(DEVICE_CC),code=sm_$(DEVICE_CC) 
## Debug info
OPTIONS += -G -g -lineinfo

#LINKOPT=-lm -lpthread

vpath %.o ./.obj
## lib dirs -L...
CCLNDIRS= 
## include dirs -I...
INCLDIRS=-I$(cudaDBE_PATH)/src -I$(RAPID_XML_PATH) -I$(CUDA_SAMPLES_PATH=)

#Directives
DFLAGS=-D__cplusplus=201103L

## Compiler Flags
OPTIONS+= $(INCLDIRS) $(CCLNDIRS) $(LINKOPT) $(DFLAGS) $(DEPEND)

DIR_GUARD=@mkdir -p $(cudaDBE_PATH)/.obj/$(@D)

all:	cudaDBE


cudaDBE: $(OBJECTS)
	$(NVCC) $(OPTIONS) -o cudaDBE $(OBJECTS:%=$(cudaDBE_PATH)/.obj/%)

$(H_OBJECTS): %.o: %.cc
	$(DIR_GUARD)
	$(CC) $(CCOPT) $< -c -o $(cudaDBE_PATH)/.obj/$@

$(D_OBJECTS): %.o: %.cu
	$(DIR_GUARD)
	$(NVCC) $(NVCCOPT) $< -dc -o $(cudaDBE_PATH)/.obj/$@

clean-gpu:
	rm -f $(D_OBJECTS:%=.obj/%)

clean-cpu:
	rm -f $(H_OBJECTS:%=.obj/%)

clean: clean-cpu clean-gpu
