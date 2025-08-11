// include/optimizer.h
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "matrix.h"

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_ADAM,
    OPTIMIZER_SOFTMAX,
} OptimizerType;

typedef struct Optimizer Optimizer;

// Function declarations
Optimizer* optimizer_create(OptimizerType type, double learning_rate);
void optimizer_free(Optimizer* opt);
void optimizer_update(Optimizer* opt, Matrix* weights, Matrix* gradients);

#endif
