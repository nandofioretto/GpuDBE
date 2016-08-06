

H_HEADERS=\
src/Algorithms/algorithm-factory.hh algorithm.hh \
src/Algorithms/Orderings/ordering.hh Orderings/linear-ordering.hh \
src/Algorithms/Orderings/pseudo-tree-ordering.hh \
src/Algorithms/DPOP/pseudo-tree-phase.hh \
src/Algorithms/DPOP/dpop.hh \
src/Algorithms/DPOP/util-propagation.hh \
src/Algorithms/DPOP/util-msg-handler.hh \
src/Algorithms/DPOP/util-msg.hh \
src/Algorithms/DPOP/value-propagation.hh \
src/Algorithms/DPOP/value-msg-handler.hh \
src/Algorithms/DPOP/value-msg.hh \
src/Algorithms/DPOP/regret-propagation.hh \
src/Algorithms/DPOP/regret-msg-handler.hh \
src/Algorithms/DPOP/regret-msg.hh \
src/Communication/scheduler.hh \
src/Communication/message.hh \
src/Communication/mailbox.hh \
src/GPU/cuda_allocator.hh \
src/GPU/cuda_dpop_state.hh \
src/Kernel/globals.hh \
src/Kernel/solution.hh \
src/Kernel/domain-factory.hh \
src/Kernel/int-domain.hh \
src/Kernel/bound-domain.hh \
src/Kernel/bound-value-iterator.hh \
src/Kernel/variable-factory.hh \
src/Kernel/variable.hh \
src/Kernel/int-variable.hh \
src/Kernel/constraint-factory.hh \
src/Kernel/constraint.hh prob-constraint.hh \
src/Kernel/agent-factory.hh \
src/Kernel/agent.hh \
src/Problem/dcop-model.hh \
src/Problem/dcop-instance.hh \
src/Problem/dcop-instance-factory.hh \
src/Problem/dcop-standard.hh \
src/Problem/IO/input-settings.hh \
src/Utilities/statistics.hh \
src/Utilities/utils.hh \
src/Utilities/constraint-utils.hh \
src/Utilities/variable-utils.hh \
src/Utilities/Statistics/metrics.hh \
src/Utilities/Statistics/local-statistics.hh \
src/Utilities/Statistics/message-statistics.hh \
src/Utilities/permutations.hh

D_HEADERS=\
src/GPU/gpu_globals.hh \
src/GPU/gpu_data_allocator.hh \
src/GPU/gpu_dpop_util_phase.hh \
src/GPU/gpu_dpop_value_phase.hh \
src/GPU/gpu_dpop_regret_phase.hh \
src/GPU/cuda_utils.hh 





################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cudaDBE.cpp 

OBJS += \
./src/cudaDBE.o 

CPP_DEPS += \
./src/cudaDBE.d 

CC_SRCS += \
../src/Algorithms/algorithm-factory.cc \
../src/Algorithms/algorithm.cc 

OBJS += \
./src/Algorithms/algorithm-factory.o \
./src/Algorithms/algorithm.o 

CC_DEPS += \
./src/Algorithms/algorithm-factory.d \
./src/Algorithms/algorithm.d 


CC_SRCS += \
../src/Algorithms/DPOP/dpop-state.cc \
../src/Algorithms/DPOP/dpop.cc \
../src/Algorithms/DPOP/pseudo-tree-phase.cc \
../src/Algorithms/DPOP/util-msg-handler.cc \
../src/Algorithms/DPOP/util-msg.cc \
../src/Algorithms/DPOP/util-propagation.cc \
../src/Algorithms/DPOP/value-msg-handler.cc \
../src/Algorithms/DPOP/value-msg.cc \
../src/Algorithms/DPOP/value-propagation.cc 

OBJS += \
./src/Algorithms/DPOP/dpop-state.o \
./src/Algorithms/DPOP/dpop.o \
./src/Algorithms/DPOP/pseudo-tree-phase.o \
./src/Algorithms/DPOP/util-msg-handler.o \
./src/Algorithms/DPOP/util-msg.o \
./src/Algorithms/DPOP/util-propagation.o \
./src/Algorithms/DPOP/value-msg-handler.o \
./src/Algorithms/DPOP/value-msg.o \
./src/Algorithms/DPOP/value-propagation.o 

CC_DEPS += \
./src/Algorithms/DPOP/dpop-state.d \
./src/Algorithms/DPOP/dpop.d \
./src/Algorithms/DPOP/pseudo-tree-phase.d \
./src/Algorithms/DPOP/util-msg-handler.d \
./src/Algorithms/DPOP/util-msg.d \
./src/Algorithms/DPOP/util-propagation.d \
./src/Algorithms/DPOP/value-msg-handler.d \
./src/Algorithms/DPOP/value-msg.d \
./src/Algorithms/DPOP/value-propagation.d 

CC_SRCS += \
../src/Algorithms/Orderings/linear-ordering.cc \
../src/Algorithms/Orderings/ordering.cc \
../src/Algorithms/Orderings/pseudo-tree-ordering.cc 

OBJS += \
./src/Algorithms/Orderings/linear-ordering.o \
./src/Algorithms/Orderings/ordering.o \
./src/Algorithms/Orderings/pseudo-tree-ordering.o 

CC_DEPS += \
./src/Algorithms/Orderings/linear-ordering.d \
./src/Algorithms/Orderings/ordering.d \
./src/Algorithms/Orderings/pseudo-tree-ordering.d 

CC_SRCS += \
../src/Communication/mailbox.cc \
../src/Communication/message.cc \
../src/Communication/scheduler.cc 

OBJS += \
./src/Communication/mailbox.o \
./src/Communication/message.o \
./src/Communication/scheduler.o 

CC_DEPS += \
./src/Communication/mailbox.d \
./src/Communication/message.d \
./src/Communication/scheduler.d 

CU_SRCS += \
../src/GPU/cuda_utils.cu \
../src/GPU/gpu_data_allocator.cu \
../src/GPU/gpu_dpop_util_phase.cu \
../src/GPU/gpu_dpop_value_phase.cu \
../src/GPU/gpu_globals.cu 

CC_SRCS += \
../src/GPU/cuda_allocator.cc \
../src/GPU/cuda_dpop_state.cc 

CU_DEPS += \
./src/GPU/cuda_utils.d \
./src/GPU/gpu_data_allocator.d \
./src/GPU/gpu_dpop_util_phase.d \
./src/GPU/gpu_dpop_value_phase.d \
./src/GPU/gpu_globals.d 

OBJS += \
./src/GPU/cuda_allocator.o \
./src/GPU/cuda_dpop_state.o \
./src/GPU/cuda_utils.o \
./src/GPU/gpu_data_allocator.o \
./src/GPU/gpu_dpop_util_phase.o \
./src/GPU/gpu_dpop_value_phase.o \
./src/GPU/gpu_globals.o 

CC_DEPS += \
./src/GPU/cuda_allocator.d \
./src/GPU/cuda_dpop_state.d 

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../src/Kernel/agent-factory.cc \
../src/Kernel/agent.cc \
../src/Kernel/bound-domain.cc \
../src/Kernel/bound-value-iterator.cc \
../src/Kernel/constraint-factory.cc \
../src/Kernel/constraint.cc \
../src/Kernel/domain-factory.cc \
../src/Kernel/globals.cc \
../src/Kernel/int-domain.cc \
../src/Kernel/int-variable.cc \
../src/Kernel/solution.cc \
../src/Kernel/table-constraint.cc \
../src/Kernel/variable-factory.cc \
../src/Kernel/variable.cc 

OBJS += \
./src/Kernel/agent-factory.o \
./src/Kernel/agent.o \
./src/Kernel/bound-domain.o \
./src/Kernel/bound-value-iterator.o \
./src/Kernel/constraint-factory.o \
./src/Kernel/constraint.o \
./src/Kernel/domain-factory.o \
./src/Kernel/globals.o \
./src/Kernel/int-domain.o \
./src/Kernel/int-variable.o \
./src/Kernel/solution.o \
./src/Kernel/table-constraint.o \
./src/Kernel/variable-factory.o \
./src/Kernel/variable.o 

CC_DEPS += \
./src/Kernel/agent-factory.d \
./src/Kernel/agent.d \
./src/Kernel/bound-domain.d \
./src/Kernel/bound-value-iterator.d \
./src/Kernel/constraint-factory.d \
./src/Kernel/constraint.d \
./src/Kernel/domain-factory.d \
./src/Kernel/globals.d \
./src/Kernel/int-domain.d \
./src/Kernel/int-variable.d \
./src/Kernel/solution.d \
./src/Kernel/table-constraint.d \
./src/Kernel/variable-factory.d \
./src/Kernel/variable.d 

CC_SRCS += \
../src/Problem/dcop-instance-factory.cc \
../src/Problem/dcop-instance.cc \
../src/Problem/dcop-model.cc \
../src/Problem/dcop-standard.cc 

OBJS += \
./src/Problem/dcop-instance-factory.o \
./src/Problem/dcop-instance.o \
./src/Problem/dcop-model.o \
./src/Problem/dcop-standard.o 

CC_DEPS += \
./src/Problem/dcop-instance-factory.d \
./src/Problem/dcop-instance.d \
./src/Problem/dcop-model.d \
./src/Problem/dcop-standard.d 

CC_SRCS += \
../src/Problem/IO/input-settings.cc 

OBJS += \
./src/Problem/IO/input-settings.o 

CC_DEPS += \
./src/Problem/IO/input-settings.d 

CC_SRCS += \
../src/Utilities/constraint-utils.cc \
../src/Utilities/permutations.cc \
../src/Utilities/statistics.cc \
../src/Utilities/utils.cc \
../src/Utilities/variable-utils.cc 

OBJS += \
./src/Utilities/constraint-utils.o \
./src/Utilities/permutations.o \
./src/Utilities/statistics.o \
./src/Utilities/utils.o \
./src/Utilities/variable-utils.o 

CC_DEPS += \
./src/Utilities/constraint-utils.d \
./src/Utilities/permutations.d \
./src/Utilities/statistics.d \
./src/Utilities/utils.d \
./src/Utilities/variable-utils.d 

CC_SRCS += \
../src/Utilities/Statistics/local-statistics.cc \
../src/Utilities/Statistics/message-statistics.cc \
../src/Utilities/Statistics/metrics.cc 

OBJS += \
./src/Utilities/Statistics/local-statistics.o \
./src/Utilities/Statistics/message-statistics.o \
./src/Utilities/Statistics/metrics.o 

CC_DEPS += \
./src/Utilities/Statistics/local-statistics.d \
./src/Utilities/Statistics/message-statistics.d \
./src/Utilities/Statistics/metrics.d 
