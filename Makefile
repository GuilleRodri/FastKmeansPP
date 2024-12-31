TARGET = kmeanspp kmeanspp_tie kmeanspp_tie_norm
CXXFLAGS = -std=c++20 -Wall -O3 -I/usr/include/eigen3  
CXX = g++


all: ${TARGET}

kmeanspp: kmeanspp.cpp
	${CXX} ${CXXFLAGS} $^ -o $@ -lstdc++fs
	
kmeanspp_tie: kmeanspp_tie.cpp
	${CXX} ${CXXFLAGS} $^ -o $@ -lstdc++fs

kmeanspp_tie_norm: kmeanspp_tie_norm.cpp
	${CXX} ${CXXFLAGS} $^ -o $@ -lstdc++fs


clean:
	@rm -f *~ *.o ${TARGET} core
