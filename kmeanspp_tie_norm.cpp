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



struct PointN {
  double ub;
  double norm;
  int id;
  
};

struct ClusterN {
  double radius;
  double upperNorm;
  double lowerNorm;
  double sum;
};


inline double updatePartition(ClusterN& P, vector<PointN>& points){
  double radius = 0.0;
  double upperNorm = 0.0;
  double lowerNorm = numeric_limits<double>::max();
  double sum = 0.0;
  for (auto& point : points) {
    double currentED = sqrt(point.ub); 
    if (point.ub > radius) radius = point.ub;
    double tmp = point.norm + currentED;
    if (tmp > upperNorm) upperNorm = tmp;
    tmp = point.norm - currentED;
    if (tmp < lowerNorm) lowerNorm = tmp;
    sum += point.ub;
    point.ub = sum;
  }
  P.radius = radius;
  P.upperNorm = upperNorm;
  P.lowerNorm = lowerNorm;
  P.sum = sum;
  return sum;
}


// Accelerated KMeans++
void acceleratedKmeansPlusPlus_TIE_Norm(const matrix& M, int nClusters, int seed) {
  // Start timing
  struct rusage usageStart, usageEnd;
  getrusage(RUSAGE_SELF, &usageStart);
  
  int nSamples = M.rows();
  int nDimensions = M.cols();
  int nPartitions = nClusters << 1;

  matrix centers(nClusters, nDimensions);
  vector<ClusterN> partitions(nPartitions);
  vector<vector<PointN>> points(nPartitions);

  // Random number generation setup
  random_device rd;
  mt19937 gen(seed);
  uniform_real_distribution<> distrib_double(0.0, 1.0);
  uniform_int_distribution<> distrib_int(0, nSamples - 1);
    
  // Select first centroid randomly
  int newCenterID = distrib_int(gen);
  centers.row(0) = M.row(newCenterID);
  double centerNorm = centers.row(0).norm();
  
  
  auto& points_0 = points[0];
  points_0.reserve(nSamples-1);
  auto& points_1 = points[1];
  points_1.reserve(nSamples-1);
  
  for (int i = 0; i < nSamples; ++i) {
    if (i == newCenterID) continue;
    const auto& point = M.row(i);
    double currentNorm = point.norm();
    double currentSED = (centers.row(0) - point).squaredNorm();
    if (currentNorm <= centerNorm) points_0.emplace_back(currentSED, currentNorm, i);  
    else points_1.emplace_back(currentSED, currentNorm, i);  
  }

  vector<double> cumSum(nPartitions, 0.0);
  double totalSum = updatePartition(partitions[0], points_0);
  cumSum[0] = totalSum;
  totalSum += updatePartition(partitions[1], points_1);
  cumSum[1] = totalSum;

  int pj = 2;
  for (int j = 1; j < nClusters; ++j) {
  
    // Select next center
    double randomValue = distrib_double(gen) * totalSum;
    auto it = lower_bound(cumSum.begin(), cumSum.begin() + pj, randomValue);
    int nextPartition = distance(cumSum.begin(), it);
    if (nextPartition > 0) randomValue -=  cumSum[nextPartition-1];
    
    const auto& selectedPoints = points[nextPartition];
    auto itp = lower_bound(selectedPoints.begin(), selectedPoints.end(), randomValue, 
               [](const PointN& point, double value) {return point.ub < value;});
    newCenterID = itp->id;
    centerNorm = itp->norm;
    centers.row(j) = M.row(newCenterID);

    // Calculate Pairwise Distances between centers
    const auto squaredDistances = (centers.topRows(j).rowwise() - centers.row(j)).rowwise().squaredNorm()/4.0;
    
    auto& points_pj0 = points[pj];
    points_pj0.reserve(nSamples);
    auto& points_pj1 = points[pj+1];
    points_pj1.reserve(nSamples);
    
    int pk = 0;
    totalSum = 0.0;
    for (int k = 0;  k < j; ++k) {
    
      double dist = squaredDistances(k);
     
      // Lower Partition
      auto& partition_pk0 = partitions[pk];
      
      if  (partition_pk0.radius > dist  && centerNorm < partition_pk0.upperNorm && centerNorm > partition_pk0.lowerNorm) {
      
        auto& points_pk0 = points[pk];
        vector<PointN> newPointList;
        newPointList.reserve(points_pk0.size());
        double cumSED = 0.0;
        for (const auto& point : points_pk0) {    
          int pointID = point.id;
          double bestSED = point.ub - cumSED;
          cumSED = point.ub;          
          if (pointID == newCenterID) continue;         
          double currentNorm = point.norm;
          double currentDiff = currentNorm - centerNorm;   
          if (bestSED > dist && bestSED > currentDiff * currentDiff) {  
            double currentSED =  (centers.row(j) - M.row(pointID)).squaredNorm();                 
            if (currentSED < bestSED) {   
              if (currentNorm <= centerNorm) points_pj0.emplace_back(currentSED, currentNorm, pointID);  
              else points_pj1.emplace_back(currentSED, currentNorm, pointID);                                        
            }
            else newPointList.emplace_back(bestSED, currentNorm, pointID);     
          }
          else newPointList.emplace_back(bestSED, currentNorm, pointID);  
        }
        points_pk0 = move(newPointList);
        totalSum += updatePartition(partition_pk0, points_pk0);
      }
      else totalSum += partition_pk0.sum;
      
      cumSum[pk] = totalSum;
      pk++;
      
      // Upper Partition
      auto& partition_pk1 = partitions[pk];
      
      if  (partition_pk1.radius > dist  && centerNorm < partition_pk1.upperNorm && centerNorm > partition_pk1.lowerNorm) {
      
        auto& points_pk1 = points[pk];
        vector<PointN> newPointList;
        newPointList.reserve(points_pk1.size());
        double cumSED = 0.0;
        for (const auto& point : points_pk1) {    
          int pointID = point.id;
          double bestSED = point.ub - cumSED;
          cumSED = point.ub;          
          if (pointID == newCenterID) continue;         
          double currentNorm = point.norm; 
          double currentDiff = currentNorm - centerNorm; 
          if (bestSED > dist && bestSED > currentDiff * currentDiff) {                  
            double currentSED =  (centers.row(j) - M.row(pointID)).squaredNorm();                    
            if (currentSED < bestSED) {
              if (currentNorm <= centerNorm) points_pj0.emplace_back(currentSED, currentNorm, pointID);  
              else points_pj1.emplace_back(currentSED, currentNorm, pointID);                                        
            }
            else newPointList.emplace_back(bestSED, currentNorm, pointID);     
          }
          else newPointList.emplace_back(bestSED, currentNorm, pointID);  
        }
        
        points_pk1 = move(newPointList);
        totalSum += updatePartition(partition_pk1, points_pk1);
      }
      else totalSum += partition_pk1.sum;
      cumSum[pk] = totalSum;
      pk++;
    }
    
    totalSum += updatePartition(partitions[pj], points_pj0);
    cumSum[pj] = totalSum;
    pj++;
    totalSum += updatePartition(partitions[pj], points_pj1);
    cumSum[pj] = totalSum;
    pj++;
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
  acceleratedKmeansPlusPlus_TIE_Norm(M, inputNClusters, inputSeed);
  
  return 0;
}


