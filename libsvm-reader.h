#ifndef LIBSVM_READER_H_
#define LIBSVM_READER_H_

#include <string>
#include <iostream>
#include <fstream>
#include <utility>
#include <limits>
#include <unordered_map>

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "boost/algorithm/string.hpp"

// Class to read SVMLight format datasets.
class LibSVMReader {
 public:
  using SparseFeaturesMatrix = Eigen::SparseMatrix<float>;

  // Returns the labels and the sparse features. The data is a column-major
  // sparse that can be accessed by columns so if you want to load the data
  // where examples are columns and rows are features set transpose=true.
  // otherwise rows are examples and columns are features.
  static std::pair<Eigen::VectorXf, SparseFeaturesMatrix> SparseLoader(
      const std::string& filename, const bool transpose = false) {
    std::ifstream input_file(filename);
    int example_index = 0;
    int min_feature_index = std::numeric_limits<int>::max();
    int max_feature_index = std::numeric_limits<int>::min();
    std::unordered_map<int, int> feature_count;
    std::vector<Eigen::Triplet<float>> triplets;
    std::vector<float> labels;
    if (input_file.is_open()) {
      for (std::string current_line; std::getline(input_file, current_line);) {
        std::vector<std::string> tokens;
        boost::split(tokens, current_line, boost::is_any_of(" "));
        labels.push_back(std::stoi(tokens[0]));
        for (int i = 1; i < tokens.size(); ++i) {
          std::vector<std::string> sparse_feature;
          boost::split(sparse_feature, tokens[i], boost::is_any_of(":"));
          const int feature_index = std::stoi(sparse_feature[0]) - 1;
          const float feature_value = std::stof(sparse_feature[1]);
          if (!transpose) {
            triplets.emplace_back(example_index, feature_index, feature_value);
          } else {
            triplets.emplace_back(feature_index, example_index, feature_value);
          }
          if (feature_index > max_feature_index) {
            max_feature_index = feature_index;
          }
          if (feature_index < min_feature_index) {
            min_feature_index = feature_index;
          }

          feature_count[feature_index]++;
        }
        example_index++;
      }
    }
    const int num_examples = example_index;
    const int num_features = max_feature_index + 1;
    const int rows = transpose ? num_features : num_examples;
    const int columns = transpose ? num_examples : num_features;
    Eigen::VectorXf output_labels =
        Eigen::Map<Eigen::VectorXf>(labels.data(), labels.size());
    SparseFeaturesMatrix examples(rows, columns);
    examples.setFromTriplets(triplets.begin(), triplets.end());
    input_file.close();
    return {std::move(output_labels), std::move(examples)};
  }
};

#endif
