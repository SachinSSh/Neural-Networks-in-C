```c++
#include <iostream>
#include <vector>

using namespace std;

class NeuralNetwork {
public:
  NeuralNetwork(int num_inputs, int num_hidden, int num_outputs) {
    // Initialize the weights and biases for the input layer
    for (int i = 0; i < num_inputs; i++) {
      for (int j = 0; j < num_hidden; j++) {
        weights_ih[i][j] = random_weight();
      }
    }

    // Initialize the weights and biases for the hidden layer
    for (int i = 0; i < num_hidden; i++) {
      for (int j = 0; j < num_outputs; j++) {
        weights_ho[i][j] = random_weight();
      }
    }
  }

  void train(vector<vector<double>> inputs, vector<vector<double>> outputs, int num_epochs) {
    // Iterate over the training epochs
    for (int epoch = 0; epoch < num_epochs; epoch++) {
      // Iterate over the training examples
      for (int i = 0; i < inputs.size(); i++) {
        // Forward pass
        vector<double> hidden_layer_output = forward_pass(inputs[i]);

        // Backward pass
        vector<double> output_error = backward_pass(outputs[i], hidden_layer_output);

        // Update the weights and biases
        update_weights_biases(inputs[i], hidden_layer_output, output_error);
      }
    }
  }

  vector<double> predict(vector<double> input) {
    // Forward pass
    vector<double> hidden_layer_output = forward_pass(input);
    vector<double> output_layer_output = forward_pass(hidden_layer_output);

    return output_layer_output;
  }

private:
  vector<vector<double>> weights_ih;  // Weights from input layer to hidden layer
  vector<vector<double>> weights_ho;  // Weights from hidden layer to output layer
  vector<double> biases_h;            // Biases for hidden layer
  vector<double> biases_o;            // Biases for output layer

  double random_weight() {
    return (rand() / double(RAND_MAX)) * 2 - 1;
  }

  vector<double> forward_pass(vector<double> input) {
    // Calculate the dot product between the input and the weights from the input layer to the hidden layer
    vector<double> hidden_layer_output;
    for (int i = 0; i < weights_ih.size(); i++) {
      double sum = 0;
      for (int j = 0; j < input.size(); j++) {
        sum += input[j] * weights_ih[i][j];
      }
      hidden_layer_output.push_back(sigmoid(sum + biases_h[i]));
    }

    // Calculate the dot product between the output from the hidden layer and the weights from the hidden layer to the output layer
    vector<double> output_layer_output;
    for (int i = 0; i < weights_ho.size(); i++) {
      double sum = 0;
      for (int j = 0; j < hidden_layer_output.size(); j++) {
        sum += hidden_layer_output[j] * weights_ho[i][j];
      }
      output_layer_output.push_back(sigmoid(sum + biases_o[i]));
    }

    return output_layer_output;
  }

  vector<double> backward_pass(vector<double> expected_output, vector<double> actual_output) {
    // Calculate the error at the output layer
    vector<double> output_error;
    for (int i = 0; i < expected_output.size(); i++) {
      output_error.push_back(expected_output[i] - actual_output[i]);
    }

    // Calculate the error at the hidden layer
    vector<double> hidden_error;
    for (int i = 0; i < weights_ho.size(); i++) {
      double sum = 0;
      for (int j = 0; j < output_error.size(); j++) {
        sum += output_error[j] * weights_ho[i][j];
      }
      hidden_error.push_back(sigmoid_prime(sum + biases_h[i]));
    }

    return hidden_error;