#include "svm-pegasos-optimizer.h"
#include "libsvm-reader.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
  const auto dataset = LibSVMReader::SparseLoader("train.dat", true);
  SVMPegasosOptimizer svm_optimizer(dataset.second, dataset.first,
      0.001);
  cout << dataset.first.size() << endl;
  svm_optimizer.TrainSGD(10000);
  return 0;
}
