CXX := g++
CXXFLAGS := -O3 -Wall -Wextra -std=c++17 -I/usr/include/eigen3 -I..

BIN_DIR := bin

.PHONY: all update test clean

all: test

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

update: $(BIN_DIR)/update_golden
	./$(BIN_DIR)/update_golden

test: $(BIN_DIR)/regression_test
	./$(BIN_DIR)/regression_test

$(BIN_DIR)/update_golden: update_golden.cpp ../fastkmeanspp.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) update_golden.cpp -o $(BIN_DIR)/update_golden

$(BIN_DIR)/regression_test: regression_test.cpp golden_generated.hpp ../fastkmeanspp.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) regression_test.cpp -o $(BIN_DIR)/regression_test

clean:
	rm -rf $(BIN_DIR) golden.json golden_generated.hpp
