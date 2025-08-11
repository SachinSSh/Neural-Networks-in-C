// src/optimizer.c
#include <stdlib.h>
#include <math.h>
#include "../include/optimizer.h"
#include "../include/matrix.h"

struct Optimizer {
    OptimizerType type;
    double learning_rate;
    Matrix* momentum;  // For momentum and Adam
    Matrix* velocity;  // For Adam
    double beta1;      // For Adam
    double beta2;      // For Adam
    double epsilon;    // For Adam
    int t;            // Time step
};

Optimizer* optimizer_create(OptimizerType type, double learning_rate) {
    Optimizer* opt = malloc(sizeof(Optimizer));
    if (!opt) return NULL;
    
    opt->type = type;
    opt->learning_rate = learning_rate;
    opt->momentum = NULL;
    opt->velocity = NULL;
    opt->beta1 = 0.9;    // Default Adam parameters
    opt->beta2 = 0.999;
    opt->epsilon = 1e-8;
    opt->t = 0;
    
    return opt;
}

void optimizer_free(Optimizer* opt) {
    if (!opt) return;
    matrix_free(opt->momentum);
    matrix_free(opt->velocity);
    free(opt);
}

void optimizer_update(Optimizer* opt, Matrix* weights, Matrix* gradients) {
    if (!opt || !weights || !gradients) return;
    
    switch (opt->type) {
        case OPTIMIZER_SGD:
            // Simple SGD update
            for (int i = 0; i < weights->rows; i++) {
                for (int j = 0; j < weights->cols; j++) {
                    weights->data[i][j] -= opt->learning_rate * gradients->data[i][j];
                }
            }
            break;
            
        case OPTIMIZER_MOMENTUM:
            // Initialize momentum if needed
            if (!opt->momentum) {
                opt->momentum = matrix_create(weights->rows, weights->cols);
                if (!opt->momentum) return;
            }
            // Momentum update
            for (int i = 0; i < weights->rows; i++) {
                for (int j = 0; j < weights->cols; j++) {
                    opt->momentum->data[i][j] = 0.9 * opt->momentum->data[i][j] + 
                                              opt->learning_rate * gradients->data[i][j];
                    weights->data[i][j] -= opt->momentum->data[i][j];
                }
            }
            break;
            
        case OPTIMIZER_ADAM:
            break;

        case OPTIMIZER_SOFTMAX:
            for (int i = 0; i < weights->rows; i++) {
                double row_sum_exp = 0.0;
                for (int j = 0; j < weights->cols; j++) {
                    weights->data[i][j] = exp(weights->data[i][j]);
                    row_sum_exp += weights->data[i][j];
                }
                for (int j = 0; j < weights->cols; j++) {
                    weights->data[i][j] /= row_sum_exp;
                }
            }
            break;
            //
            
    }
}
