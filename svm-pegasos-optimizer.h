#ifndef SVM_PEGASOS_OPTIMIZER_H_
#define SVM_PEGASOS_OPTIMIZER_H_

#include <iostream>
#include <memory>
#include <random>

#include <Eigen/Sparse>
#include <Eigen/Dense>

float HingLoss(const float l, const float p) {
  return std::max(0.f, 1.f - l * p);
}

class SVMPegasosOptimizer {
 public:
  SVMPegasosOptimizer(Eigen::SparseMatrix<float> examples,
                      Eigen::VectorXf predictions, const float regularization)
      : examples_(examples),
        predictions_(predictions),
        regularization_(regularization),
        parameters_(
            new Eigen::VectorXf(Eigen::VectorXf::Random(examples_.rows()))),
        gen_(rand_()) {}
  // Runs Pegasos optimizer for number of iterations and returns the
  // loss in each iteration.
  double TrainSGD(int iterations) {
    std::cout << "Initial loss : " << ComputeLoss() << std::endl;
    const float learning_rate = 0.2;
    std::uniform_int_distribution<> rand_int(0, examples_.cols() - 1);
    for (int i = 1; i <= iterations; ++i) {
      const int random_index = rand_int(gen_);
      const float scale_i = 1.0f - (1.0f / i);
      const float p =
          (parameters_->transpose() * examples_.col(random_index))(0);
      const float eta_i = (1.0f / (regularization_ * i));
      *parameters_ *= scale_i;
      if (p * predictions_(random_index) < 1.0f) {
        *parameters_ +=
            eta_i* predictions_(random_index) * examples_.col(random_index);
      }
      std::cout << "Last Loss : " << ComputeLoss() << std::endl;
    }
  }

  double TrainMiniBatch(int iterations, int batch_size) {}

 private:
  float ComputeLoss() {
    float emperical_loss = 0.0;
    for (int i = 0; i < examples_.cols(); ++i) {
      const float p = (parameters_->transpose() * examples_.col(i))(0);
      emperical_loss += HingLoss(predictions_(i), p);
    }
    return 0.5 * regularization_ * parameters_->transpose() * *parameters_ +
           (1.0f / examples_.cols()) * emperical_loss;
  }
  const Eigen::SparseMatrix<float> examples_;
  const Eigen::VectorXf predictions_;
  std::unique_ptr<Eigen::VectorXf> parameters_;
  const float regularization_;
  std::random_device rand_;
  std::mt19937 gen_;
};

#endif
