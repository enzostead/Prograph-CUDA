CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS1=`pkg-config --libs opencv`
LDLIBS2=-lm -lIL

all: sobel sobel-cu sobel-stream-cu blur blur-cu edge edge-cu laplaciang laplaciang-cu line line-cu

blur: blur.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS1)

edge: edge.cpp
	 $(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS1)
	 
laplaciang: laplaciang.cpp
	 $(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS1)
	 
line: line.cpp
	 $(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS1)

sobel: sobel.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS1)

blur-cu: blur.cu
	nvcc -o $@ $<  $(LDLIBS1)
	
edge-cu: edge.cu
	nvcc -o $@ $<  $(LDLIBS1)

laplaciang-cu: laplaciang.cu
	nvcc -o $@ $<  $(LDLIBS1)

line-cu: line.cu
	nvcc -o $@ $<  $(LDLIBS1)
	
sobel-cu: sobel.cu
	nvcc -o $@ $<  $(LDLIBS1)

sobel-stream-cu: sobel-stream.cu
	nvcc -o $@ $<  $(LDLIBS1)

.PHONY: clean

clean:
	rm sobel sobel-cu sobel-stream-cu blur blur-cu edge edge-cu laplaciang laplaciang-cu line line-cu out_*.jpg
