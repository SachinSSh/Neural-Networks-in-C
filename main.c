// src/main.c
#include <stdio.h>
#include "../include/error.h"
#include <stdlib.h>
#include <time.h>
#define _USE_MATH_DEFINES  // Must come before math.h
#include <math.h>
#define PI 3.14159265358979323846
#include "../include/network.h"
#include "../include/matrix.h"
#include "../include/activation.h"
#include "../include/optimizer.h"



// Helper function to print matrix
void print_matrix(Matrix* m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%.4f ", m->data[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Example 1: XOR problem
void run_xor_example() {
    printf("\n=== XOR Problem Example ===\n");
    
    // Create network: 2 inputs, 4 hidden, 1 output
    NeuralNetwork* nn = nn_create(2, 4, 1, 
                                 ACTIVATION_RELU,    // Hidden layer activation
                                 ACTIVATION_SIGMOID, // Output layer activation
                                 OPTIMIZER_ADAM,     // Optimizer
                                 0.01);             // Learning rate
    
    // Training data
    double inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double targets[4] = {0, 1, 1, 0};
    
    // Convert to matrices
    Matrix* input_matrices[4];
    Matrix* target_matrices[4];
    for (int i = 0; i < 4; i++) {
        input_matrices[i] = matrix_create(2, 1);
        input_matrices[i]->data[0][0] = inputs[i][0];
        input_matrices[i]->data[1][0] = inputs[i][1];
        
        target_matrices[i] = matrix_create(1, 1);
        target_matrices[i]->data[0][0] = targets[i];
    }
    
    // Training loop
    printf("Training XOR network...\n");
    for (int epoch = 0; epoch < 1000; epoch++) {
        Error* save_error = nn_train_batch(nn, input_matrices, target_matrices, 4);
        if (save_error) {
            error_print(save_error);
            error_free(save_error);
            break;
        }
        
        // Print progress every 100 epochs
        if ((epoch + 1) % 100 == 0) {
            double total_error = 0;
            for (int i = 0; i < 4; i++) {
                Matrix* output = nn_feedforward(nn, input_matrices[i]);
                total_error += fabs(output->data[0][0] - targets[i]);
                matrix_free(output);
            }
            printf("Epoch %d, Average error: %.4f\n", epoch + 1, total_error / 4);
        }
    }
    
    // Test the network
    printf("\nTesting XOR network:\n");
    for (int i = 0; i < 4; i++) {
        Matrix* output = nn_feedforward(nn, input_matrices[i]);
        printf("Input: %.0f %.0f, Output: %.4f, Expected: %.0f\n",
               inputs[i][0], inputs[i][1], output->data[0][0], targets[i]);
        matrix_free(output);
    }
    
    // Save the trained network
    Error* save_error = nn_save(nn, "xor_model.nn");
    if (save_error) {
        error_print(save_error);
        error_free(save_error);
    } else {
        printf("\nSaved XOR model to 'xor_model.nn'\n");
    }
    
    // Cleanup
    for (int i = 0; i < 4; i++) {
        matrix_free(input_matrices[i]);
        matrix_free(target_matrices[i]);
    }
    nn_free(nn);
}



int main() {
    // Seed random number generator
    srand(time(NULL));
    
    // Run examples
    run_xor_example();
    //run_function_approximation();
    
    printf("\nAll examples completed successfully!\n");
    return 0;
}