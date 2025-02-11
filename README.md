```cpp
#include <iostream>
#include <vector>

using namespace std;

class NeuralNetwork {
public:
  NeuralNetwork(vector<int> layers) {
    // ...
  }

  void train(vector<vector<double>> inputs, vector<vector<double>> outputs) {
    // ...
  }

  vector<double> predict(vector<double> input) {
    // ...
  }
};

int main() {
  // Create a neural network with two hidden layers, each with 100 neurons.
  NeuralNetwork nn({784, 100, 100, 10});

  // Train the neural network on the MNIST dataset.
  nn.train(mnist_inputs, mnist_outputs);

  // Predict the digit for a new input image.
  vector<double> prediction = nn.predict({...});

  // Print the predicted digit.
  cout << prediction << endl;

  return 0;
}
```