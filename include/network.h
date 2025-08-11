#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"
#include "activation.h"
#include "optimizer.h"
#include "error.h"

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    ActivationType hidden_activation;
    ActivationType output_activation;
    OptimizerType optimizer;
    double learning_rate;
} ModelConfig;

typedef struct NeuralNetwork {
    int input_size;
    int hidden_size;
    int output_size;
    Matrix* hidden_weights;
    Matrix* hidden_biases;
    Matrix* output_weights;
    Matrix* output_biases;
    ActivationType hidden_activation;
    ActivationType output_activation;
    OptimizerType optimizer;
    double learning_rate;
} NeuralNetwork;

NeuralNetwork* nn_create(int input_size, int hidden_size, int output_size,
                        ActivationType hidden_activation,
                        ActivationType output_activation,
                        OptimizerType optimizer,
                        double learning_rate);
void nn_free(NeuralNetwork* nn);
Matrix* nn_feedforward(NeuralNetwork* nn, Matrix* input);
Error* nn_train_batch(NeuralNetwork* nn, Matrix** inputs, Matrix** targets, int batch_size);
Error* nn_save(NeuralNetwork* nn, const char* filename);
NeuralNetwork* nn_load(const char* filename);

#endif
