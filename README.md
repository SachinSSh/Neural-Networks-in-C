```
#include <iostream>
#include <vector>
using namespace std;

// Define the neuron class
class Neuron {
public:
    // Constructor
    Neuron(int numInputs) {
        // Initialize the weights and bias
        for (int i = 0; i < numInputs; i++) {
            weights.push_back(0.0);
        }
        bias = 0.0;
    }

    // Activation function
    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    // Output function
    double output(vector<double> inputs) {
        // Calculate the weighted sum of the inputs
        double weightedSum = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            weightedSum += weights[i] * inputs[i];
        }

        // Add the bias
        weightedSum += bias;

        // Apply the activation function
        return sigmoid(weightedSum);
    }

    // Update weights and bias
    void update(vector<double> inputs, double error, double learningRate) {
        // Update the weights
        for (int i = 0; i < weights.size(); i++) {
            weights[i] += learningRate * error * inputs[i];
        }

        // Update the bias
        bias += learningRate * error;
    }

private:
    // Weights
    vector<double> weights;

    // Bias
    double bias;
};

// Define the neural network class
class NeuralNetwork {
public:
    // Constructor
    NeuralNetwork(int numInputs, int numOutputs) {
        // Create a vector of neurons for the hidden layer
        for (int i = 0; i < numInputs; i++) {
            hiddenLayer.push_back(Neuron(numInputs));
        }

        // Create a vector of neurons for the output layer
        for (int i = 0; i < numOutputs; i++) {
            outputLayer.push_back(Neuron(numInputs));
        }
    }

    // Feedforward function
    vector<double> feedforward(vector<double> inputs) {
        // Calculate the output of the hidden layer
        vector<double> hiddenOutputs;
        for (int i = 0; i < hiddenLayer.size(); i++) {
            hiddenOutputs.push_back(hiddenLayer[i].output(inputs));
        }

        // Calculate the output of the output layer
        vector<double> output;
        for (int i = 0; i < outputLayer.size(); i++) {
            output.push_back(outputLayer[i].output(hiddenOutputs));
        }

        // Return the output
        return output;
    }

    // Backpropagation function
    void backpropagate(vector<double> inputs, vector<double> expectedOutputs, double learningRate) {
        // Calculate the error for the output layer
        vector<double> outputErrors;
        for (int i = 0; i < outputLayer.size(); i++) {
            outputErrors.push_back(expectedOutputs[i] - outputLayer[i].output(inputs));
        }

        // Update the weights and biases of the output layer
        for (int i = 0; i < outputLayer.size(); i++) {
            outputLayer[i].update(inputs, outputErrors[i], learningRate);
        }

        // Calculate the error for the hidden layer
        vector<double> hiddenErrors;
        for (int i = 0; i < hiddenLayer.size(); i++) {
            double weightedError = 0.0;
            for (int j = 0; j < outputLayer.size(); j++) {
                weightedError += outputErrors[j] * outputLayer[j].weights[i];
            }
            hiddenErrors.push_back(weightedError);
        }

        // Update the weights and biases of the hidden layer
        for (int i = 0; i < hiddenLayer.size(); i++) {
            hiddenLayer[i].update(inputs, hiddenErrors[i], learningRate);
        }
    }

    // Train function
    void train(vector<vector<double>> inputs, vector<vector<double>> expectedOutputs, double learningRate, int numEpochs) {
        // Iterate over the epochs
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            // Iterate over the training data
            for (int i = 0; i < inputs.size(); i++) {
                // Feedforward the network
                vector<double> output = feedforward(inputs[i]);

                // Backpropagate the error
                backpropagate(inputs[i], expectedOutputs[i], learningRate);
            }
        }
    }

private:
    // Hidden layer
    vector<Neuron> hiddenLayer;

    // Output layer
    vector<Neuron> outputLayer;
};

int main() {
    // Create a neural network with 2 inputs and 1 output
    NeuralNetwork network(2, 1);

    // Train the network on the XOR dataset
    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> expectedOutputs = {{0}, {1}, {1}, {0}};
    network.train(inputs, expectedOutputs, 0.1, 1000);

    // Test the network on the XOR dataset
    for (int i = 0; i < inputs.size(); i++) {
        vector<double> output = network.feedforward(inputs[i]);
        cout << "Input: " << inputs[i][0] << ", " << inputs[i][1] << " Output: " << output[0] << endl;
    }

    return 0;
}
```