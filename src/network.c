// src/network.c
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../include/network.h"
#include "../include/matrix.h"
#include "../include/error.h"
#include "../include/activation.h"


typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    ActivationType hidden_activation;
    ActivationType output_activation;
    OptimizerType optimizer;
    double learning_rate;
} ModelConfig;

// Improved Xavier/Glorot initialization for weights
Matrix* initialize_weights(int rows, int cols) {
    Matrix* weights = matrix_create(rows, cols);
    if (!weights) return NULL;
    
    // Xavier initialization: std = sqrt(2.0 / (input_dim + output_dim))
    double std = sqrt(2.0 / (rows + cols));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Generate random number between -1 and 1
            double r = ((double)rand() / RAND_MAX) * 2 - 1;
            weights->data[i][j] = r * std;
        }
    }
    return weights;
}


NeuralNetwork* nn_create(int input_size, int hidden_size, int output_size,
                        ActivationType hidden_activation,
                        ActivationType output_activation,
                        OptimizerType optimizer,
                        double learning_rate) {
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    if (!nn) return NULL;

    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;
    
    // Use improved weight initialization
    nn->hidden_weights = initialize_weights(hidden_size, input_size);
    nn->output_weights = initialize_weights(output_size, hidden_size);
    
    // Initialize biases to small values close to zero
    nn->hidden_biases = matrix_create(hidden_size, 1);
    nn->output_biases = matrix_create(output_size, 1);
    
    if (!nn->hidden_weights || !nn->hidden_biases || 
        !nn->output_weights || !nn->output_biases) {
        nn_free(nn);
        return NULL;
    }
    
    // Initialize biases to small random values
    for (int i = 0; i < hidden_size; i++) {
        nn->hidden_biases->data[i][0] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
    }
    for (int i = 0; i < output_size; i++) {
        nn->output_biases->data[i][0] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
    }
    
    nn->hidden_activation = hidden_activation;
    nn->output_activation = OPTIMIZER_SOFTMAX;  // Change to softmax for classification
    nn->optimizer = optimizer;
    nn->learning_rate = learning_rate;
    
    return nn;
}

Matrix* nn_feedforward(NeuralNetwork* nn, Matrix* input) {
    if (!nn || !input) return NULL;
    
    // Hidden layer
    Matrix* hidden = matrix_create(nn->hidden_size, 1);
    if (!hidden) return NULL;
    
    // Calculate hidden layer outputs
    for (int i = 0; i < nn->hidden_size; i++) {
        double sum = 0;
        for (int j = 0; j < nn->input_size; j++) {
            sum += nn->hidden_weights->data[i][j] * input->data[j][0];
        }
        sum += nn->hidden_biases->data[i][0];
        hidden->data[i][0] = activation_function(sum, nn->hidden_activation);
    }
    
    // Output layer
    Matrix* output = matrix_create(nn->output_size, 1);
    if (!output) {
        matrix_free(hidden);
        return NULL;
    }
    
    // Calculate output layer outputs
    for (int i = 0; i < nn->output_size; i++) {
        double sum = 0;
        for (int j = 0; j < nn->hidden_size; j++) {
            sum += nn->output_weights->data[i][j] * hidden->data[j][0];
        }
        sum += nn->output_biases->data[i][0];
        output->data[i][0] = activation_function(sum, nn->output_activation);
    }
    
    matrix_free(hidden);
    return output;
}



Error* nn_train_batch(NeuralNetwork* nn, Matrix** inputs, Matrix** targets, int batch_size) {
    if (!nn || !inputs || !targets || batch_size <= 0) {
        return error_create(ERROR_NULL_POINTER, "Invalid parameters for training");
    }

    // Accumulated gradients for batch
    Matrix* hidden_weights_grad = matrix_create(nn->hidden_size, nn->input_size);
    Matrix* hidden_biases_grad = matrix_create(nn->hidden_size, 1);
    Matrix* output_weights_grad = matrix_create(nn->output_size, nn->hidden_size);
    Matrix* output_biases_grad = matrix_create(nn->output_size, 1);
    
    if (!hidden_weights_grad || !hidden_biases_grad || 
        !output_weights_grad || !output_biases_grad) {
        // Clean up and return error
        matrix_free(hidden_weights_grad);
        matrix_free(hidden_biases_grad);
        matrix_free(output_weights_grad);
        matrix_free(output_biases_grad);
        return error_create(ERROR_MEMORY, "Failed to allocate memory for gradients");
    }

    double batch_loss = 0.0;
    
    // Process each sample in the batch
    for (int b = 0; b < batch_size; b++) {
        // Forward pass
        Matrix* hidden_pre = matrix_multiply(nn->hidden_weights, inputs[b]);
        matrix_add_inplace(hidden_pre, nn->hidden_biases);
        Matrix* hidden = activate(hidden_pre, nn->hidden_activation);
        
        Matrix* output_pre = matrix_multiply(nn->output_weights, hidden);
        matrix_add_inplace(output_pre, nn->output_biases);
        Matrix* output = activate(output_pre, nn->output_activation);
        
        // Compute loss and gradients
        Matrix* output_error = matrix_subtract(output, targets[b]);
        batch_loss += matrix_sum_squared(output_error) / 2.0;
        
        // Backward pass
        Matrix* output_delta = matrix_hadamard(output_error, 
                                             activate_derivative(output_pre, nn->output_activation));
        
        Matrix* hidden_error = matrix_multiply_transpose(nn->output_weights, output_delta);
        Matrix* hidden_delta = matrix_hadamard(hidden_error, 
                                             activate_derivative(hidden_pre, nn->hidden_activation));
        
        // Accumulate gradients
        Matrix* output_weights_grad_batch = matrix_multiply(output_delta, matrix_transpose(hidden));
        Matrix* hidden_weights_grad_batch = matrix_multiply(hidden_delta, matrix_transpose(inputs[b]));
        
        matrix_add_inplace(output_weights_grad, output_weights_grad_batch);
        matrix_add_inplace(hidden_weights_grad, hidden_weights_grad_batch);
        matrix_add_inplace(output_biases_grad, output_delta);
        matrix_add_inplace(hidden_biases_grad, hidden_delta);
        
        // Clean up temporary matrices
        matrix_free(hidden_pre);
        matrix_free(hidden);
        matrix_free(output_pre);
        matrix_free(output);
        matrix_free(output_error);
        matrix_free(output_delta);
        matrix_free(hidden_error);
        matrix_free(hidden_delta);
        matrix_free(output_weights_grad_batch);
        matrix_free(hidden_weights_grad_batch);
    }
    
    // Apply average gradients with learning rate
    double scale = nn->learning_rate / batch_size;
    matrix_scale(hidden_weights_grad, scale);
    matrix_scale(hidden_biases_grad, scale);
    matrix_scale(output_weights_grad, scale);
    matrix_scale(output_biases_grad, scale);
    
    // Update weights and biases
    matrix_subtract_inplace(nn->hidden_weights, hidden_weights_grad);
    matrix_subtract_inplace(nn->hidden_biases, hidden_biases_grad);
    matrix_subtract_inplace(nn->output_weights, output_weights_grad);
    matrix_subtract_inplace(nn->output_biases, output_biases_grad);
    
    // Clean up
    matrix_free(hidden_weights_grad);
    matrix_free(hidden_biases_grad);
    matrix_free(output_weights_grad);
    matrix_free(output_biases_grad);
    
    return NULL;
}

void nn_free(NeuralNetwork* nn) {
    if (nn == NULL) return;
    
    matrix_free(nn->hidden_weights);
    matrix_free(nn->hidden_biases);
    matrix_free(nn->output_weights);
    matrix_free(nn->output_biases);
    free(nn);
}

Error* nn_save(NeuralNetwork* nn, const char* filename) {
    if (!nn || !filename) {
        return error_create(ERROR_NULL_POINTER, "Invalid parameters for saving");
    }

    FILE* file = fopen(filename, "wb");
    if (!file) {
        return error_create(ERROR_FILE_IO, "Could not open file for writing");
    }

    // Save configuration
    ModelConfig config = {
        .input_size = nn->input_size,
        .hidden_size = nn->hidden_size,
        .output_size = nn->output_size,
        .hidden_activation = nn->hidden_activation,
        .output_activation = nn->output_activation,
        .optimizer = nn->optimizer,
        .learning_rate = nn->learning_rate
    };
    fwrite(&config, sizeof(ModelConfig), 1, file);

    // Save weights and biases
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->input_size; j++) {
            fwrite(&nn->hidden_weights->data[i][j], sizeof(double), 1, file);
        }
    }

    for (int i = 0; i < nn->hidden_size; i++) {
        fwrite(&nn->hidden_biases->data[i][0], sizeof(double), 1, file);
    }

    for (int i = 0; i < nn->output_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            fwrite(&nn->output_weights->data[i][j], sizeof(double), 1, file);
        }
    }

    for (int i = 0; i < nn->output_size; i++) {
        fwrite(&nn->output_biases->data[i][0], sizeof(double), 1, file);
    }

    fclose(file);
    return NULL;
}
