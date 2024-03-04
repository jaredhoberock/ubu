CXX := clang
CUDACXX := circle
# this discovers the top-level directory of ubu which is also its include path 
UBU_INC_PATH := $(shell git rev-parse --show-toplevel)
# this discovers the top-level CUDA directory, which is the grandparent directory of nvcc
CUDA_PATH := $(shell dirname $(shell dirname $(shell which nvcc)))
CUDA_INC_PATH := $(CUDA_PATH)/include
CUDA_LIB_PATH := $(CUDA_PATH)/lib64
INCLUDES := -I$(UBU_INC_PATH) -I$(CUDA_INC_PATH)
CCFLAGS := -std=c++20 -Wall -Werror -ferror-limit=3 $(INCLUDES)
CUDACXXFLAGS := -libstdc++=11 --cuda-path=$(CUDA_PATH) $(CCFLAGS) -sm_60

# this gets a list of all directories in this directory
DIRECTORIES = $(sort $(wildcard */))

# this first builds all the tests, then runs them, then prints if everything ran successfully
test : all-test-sources all-cpp-test-programs all-cuda-test-programs all-test-results

# this turns a .hpp file into a .cpp file
%.cpp : %.hpp
	@echo '#include <iostream>\n#include "'$<'"\nint main(){ test_'$(basename $<)'(); std::cout << "OK" << std::endl; return 0;}' > $@

# this turns a .hpp file into a .cu file
%.cu : %.hpp
	@echo '#include <iostream>\n#include "'$<'"\nint main(){ test_'$(basename $<)'(); std::cout << "OK" << std::endl; return 0;}' > $@

# this compiles a .cpp file into an executable file (without a file extension)
% : %.cpp
	$(CXX) $(CCFLAGS) $< -lstdc++ -lpthread -L$(CUDA_LIB_PATH) -lcudart -o $@

# this compiles a .cpp file into a CUDA executable file (without a file extension)
%-cuda : %.cpp
	$(CUDACXX) $(CUDACXXFLAGS) $< -lstdc++ -lpthread -o $@

# this runs an executable silently
test.% : %
	./$< > /dev/null

# this is a list of test program c++ sources
all-test-sources: $(addsuffix .cpp, $(basename $(wildcard *.hpp)))

# this is a list of test program names
all-cpp-test-programs : $(basename $(wildcard *.hpp))

# this is a list of CUDA test program names
all-cuda-test-programs : $(addsuffix -cuda, $(basename $(wildcard *.hpp)))

# this is a list of test program names prefixed by test
all-test-results : $(addprefix test., $(basename $(wildcard *.hpp))) $(addsuffix -cuda, $(addprefix test., $(basename $(wildcard *.hpp))))

clean:
	rm -f *.cpp *.cu *-cuda $(basename $(wildcard *.hpp))

.PHONY : test clean $(all-test-results) $(DIRECTORIES)

# when the name of a directory is a target, make recursively descends into that directory
$(DIRECTORIES):
	$(MAKE) --directory=$@ $(TARGET)

# the all target builds and tests absolutely everything
# || exit 1 causes our build to fail if any subdirectory's build fails
all: test
	@for dir in $(DIRECTORIES); do \
	  $(MAKE) --directory=$$dir $@ || exit 1; \
        done
	@echo "$(CURDIR) OK."

# the all_clean target cleans absolutely everything
all_clean: clean
	@for dir in $(DIRECTORIES); do \
	  $(MAKE) --directory=$$dir $@; \
	 done

