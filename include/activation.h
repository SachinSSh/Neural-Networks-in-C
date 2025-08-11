// include/activation.h
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "matrix.h"

typedef enum {
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_RELU
} ActivationType;

typedef struct {
    ActivationType type;
} Activation;


Activation* activation_create(ActivationType type);
void activation_free(Activation* act);
Matrix* apply_activation(Activation* act, Matrix* m);
Matrix* apply_activation_derivative(Activation* act, Matrix* m);
double activation_function(double x, ActivationType type);
double activation_derivative(double x, ActivationType type);

#endif
