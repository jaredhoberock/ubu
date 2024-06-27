CUDACXX := circle
# this discovers the top-level directory of ubu which is also its include path 
UBU_INC_PATH := $(shell git rev-parse --show-toplevel)
# this discovers the top-level CUDA directory, which is the grandparent directory of nvcc
CUDA_PATH := $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_INC_PATH := $(CUDA_PATH)/include
CUDA_LIB_PATH := $(CUDA_PATH)/lib64
INCLUDES := -I$(UBU_INC_PATH) -I$(CUDA_INC_PATH)
CCFLAGS := -std=c++20 -Wall -Werror -ferror-limit=3 $(INCLUDES)
CUDACXXFLAGS := --cuda-path=$(CUDA_PATH) $(CCFLAGS) -sm_80

two_pass_reduce.out: two_pass_reduce.cpp measure_bandwidth_of_invocation.hpp
	$(CUDACXX) $(CUDACXXFLAGS) two_pass_reduce.cpp -L$(CUDA_LIB_PATH) -lcudart -lfmt -o $@

# this runs reduce_after_at.out sliently and quickly
test: two_pass_reduce.out
	./$< quick > /dev/null

all: test

clean:
	rm -rf *.o *.out

all_clean: clean

