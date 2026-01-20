#include <iostream>
#include <string>
#include <cstdlib>
#include "fastkmeanspp.hpp"

using namespace std;

int main(int argc, char* argv[]) {
  string inputFile;
  int k = 32;
  unsigned seed = 0;
  int acc = 0;
  bool scoreOnly = false;

  // Argument parser
  for (int i = 1; i < argc; ++i) {
    string arg = argv[i];
    if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
      inputFile = argv[++i];
    } else if ((arg == "-k" || arg == "--clusters") && i + 1 < argc) {
      k = stoi(argv[++i]);
    } else if ((arg == "-s" || arg == "--seed") && i + 1 < argc) {
      seed = static_cast<unsigned>(stoul(argv[++i]));
    } else if ((arg == "-a" || arg == "--acceleration") && i + 1 < argc) {
      acc = stoi(argv[++i]);
    } else if (arg == "--score-only") {
      scoreOnly = true;
    } else {
      cerr << "Unknown or incomplete argument: " << arg << endl;
      return 1;
    }
  }
  
  // Argument validation
  if (inputFile.empty() || k <= 0 || acc < 0 || acc > 2) {
   cerr << "Usage: ./main -i [input_file] -k [clusters] -seed [seed] -acc [0|1|2]" << endl;
   return 1;
  }

  try {
    FastKMeansPP::Matrix centers;
    double score = FastKMeansPP::run_kmeanspp(inputFile, centers, k, seed, acc);
    
    if (scoreOnly) {
      cout << score << "\n";
      return 0;
    }

    cout << "KMeans++ Initialization Complete!\n";
    cout << "\tFile:      " << inputFile << "\n";
    cout << "\tClusters:  " << k << "\n";
    cout << "\tSeed:      " << seed << "\n";
    cout << "\tMethod:    " << (acc == 0 ? "Standard" : acc == 1 ? "TIE" : "TIE + Norm") << "\n";
    cout << "\tScore:     " << score << "\n";

  } catch (const exception& ex) {
    cerr << "Error: " << ex.what() << endl;
    return 1;
  }

  return 0;
}

