// src/activation.c
#include <math.h>
#include <stdlib.h>
#include "../include/activation.h"

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

double tanh_activate(double x) {
    return tanh(x);
}

double tanh_derivative(double x) {
    double t = tanh(x);
    return 1 - t * t;
}

Activation* activation_create(ActivationType type) {
    Activation* act = malloc(sizeof(Activation));
    act->type = type;
    
    switch(type) {
        case ACTIVATION_SIGMOID:
            act->activate = sigmoid;
            act->derivative = sigmoid_derivative;
            break;
        case ACTIVATION_RELU:
            act->activate = relu;
            act->derivative = relu_derivative;
            break;
        case ACTIVATION_TANH:
            act->activate = tanh_activate;
            act->derivative = tanh_derivative;
            break;
    }
    
    return act;
}