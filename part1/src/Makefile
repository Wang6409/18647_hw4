CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -fopenmp -Wall -Wextra
magic_bf: magic_bf.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<
clean:
	rm -f magic_bf
