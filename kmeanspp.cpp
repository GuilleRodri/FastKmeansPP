#include <iostream>
#include <string>
#include <filesystem>
#include <utility>
#include <numeric>
#include <cmath>
#include <limits>
#include <random>
#include <set>
#include <queue>
#include <iterator>
#include <chrono>
#include <fstream>
#include <deque>
#include <cstring>
#include <Eigen/Dense>
#include <ctime>
#include <time.h>
#include <sys/resource.h>

using namespace std;
using Eigen::VectorXd;
using Eigen::Map;

namespace fs = filesystem;
int inputNClusters = 4096;
int inputSeed = 0;

string inputDataset;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix;


void read_parameters(int argc, char **argv) {
  int iarg = 1;
  while (iarg < argc) {
    if (strcmp(argv[iarg],"-input")==0) inputDataset = (argv[++iarg]);
    if (strcmp(argv[iarg],"-n_clusters")==0) inputNClusters = atoi(argv[++iarg]);
    if (strcmp(argv[iarg],"-seed")==0) inputSeed = atoi(argv[++iarg]);
    iarg++;
  }
}


template <typename T>
void printVector(const vector<T>& vec) {
  for (const auto& element : vec) {
    cout << element << " ";
  }
  cout << endl;
}


template <typename T>
void printMatrix(const vector<vector<T>>& mat) {
  for (const auto& vec : mat) {
    printVector(vec);
  }
}


matrix readFile(const string& filename) {
  vector<vector<double>> data;
  ifstream file(filename);

  string line;
  while (getline(file, line)) {
    vector<double> row;
    stringstream lineStream(line);
    string cell;

    while (getline(lineStream, cell, ',')) {
      row.push_back(stod(cell));
    }

    data.push_back(row);
  }
  
  int nSamples = data.size();
  int nDimensions = data[0].size();
  matrix M(nSamples, nDimensions);

  for (int i = 0; i < nSamples; ++i) {
    M.row(i) = Eigen::Map<Eigen::VectorXd>(data[i].data(), data[i].size());
  }
  
  return M;
}



// Select next centroid by Roulette Wheel
int selectCentroid(const vector<double>& weights, double randomValue) {
  double cumulativeSum = 0.0;
  for (size_t i = 0; i < weights.size(); ++i) {
    cumulativeSum += weights[i];
    if (randomValue <= cumulativeSum) {
      return i;
    }
  }
  return weights.size() - 1;
}




// Standard
void KmeansPlusPlus(const matrix& M, int nClusters, int seed) {

  struct rusage usageStart, usageEnd;
  getrusage(RUSAGE_SELF, &usageStart);

  int nSamples = M.rows();
  int nDimensions = M.cols();
  
  vector<double> d(nSamples);
  matrix centers(nClusters, nDimensions);
  vector<bool> isCenter(nSamples,false);
  
  // Random number generation setup
  random_device rd;
  mt19937 gen(seed);
  uniform_real_distribution<> distrib_double(0.0, 1.0);
  uniform_int_distribution<> distrib_int(0,nSamples-1);
  
  // Select first centroid randomly
  int newCenterID = distrib_int(gen);
  centers.row(0) = M.row(newCenterID);
  isCenter[newCenterID] = true;
  d[newCenterID] = 0.0;
  
  double totalSum = 0.0;
  for (int i = 0; i < nSamples; ++i) {
    if (i == newCenterID) continue;
    d[i] = (centers.row(0) - M.row(i)).squaredNorm();
    totalSum += d[i];	
  }
  
  for (int j = 1; j < nClusters; ++j) {
  
    // Select next centroid by Roulette Wheel
    newCenterID = selectCentroid(d, distrib_double(gen)*totalSum);
    centers.row(j) = M.row(newCenterID);
    isCenter[newCenterID] = true;
    d[newCenterID] = 0.0;
  
    totalSum = 0.0;
    for (int i = 0; i < nSamples; ++i) {
      if (isCenter[i]) continue;
      double currentSED = (centers.row(j) - M.row(i)).squaredNorm();
      if (currentSED < d[i]) {
        d[i] = currentSED;
      }
      totalSum += d[i];	
    }
    
  }
  
  // End timing
  getrusage(RUSAGE_SELF, &usageEnd);

  // Calculate and return the difference in user and system time
  double startTime = (double)usageStart.ru_utime.tv_sec + (double)usageStart.ru_utime.tv_usec * 1.0E-6 +
                       (double)usageStart.ru_stime.tv_sec + (double)usageStart.ru_stime.tv_usec * 1.0E-6;
  double endTime = (double)usageEnd.ru_utime.tv_sec + (double)usageEnd.ru_utime.tv_usec * 1.0E-6 +
                     (double)usageEnd.ru_stime.tv_sec + (double)usageEnd.ru_stime.tv_usec * 1.0E-6;

  cout << "time " << endTime - startTime << endl;
  cout << "score " << totalSum << endl;
  cout << centers << endl;
}



/* ---------- MAIN ---------- */

int main(int argc, char **argv) {

  if (argc < 3) {
    cout << "Use: main -input <input dataset> ..." << endl;
    exit(1);
  }
  else read_parameters(argc,argv);

  auto M = readFile(inputDataset);
  
  cout << fixed << setprecision(6) << "n " << M.rows() << " d " << M.cols() << " k " << inputNClusters << " s " << inputSeed << endl;
  KmeansPlusPlus(M, inputNClusters, inputSeed);
  
  return 0;
}


