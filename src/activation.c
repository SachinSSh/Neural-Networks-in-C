// src/activation.c
#include <stdlib.h>
#include <math.h>
#include "../include/activation.h"
#include "../include/matrix.h"

Activation* activation_create(ActivationType type) {
    Activation* act = malloc(sizeof(Activation));
    if (act == NULL) return NULL;
    act->type = type;
    return act;
}

void activation_free(Activation* act) {
    free(act);
}

double activation_function(double x, ActivationType type) {
    switch (type) {
        case ACTIVATION_SIGMOID:
            return 1.0 / (1.0 + exp(-x));
        case ACTIVATION_TANH:
            return tanh(x);
        case ACTIVATION_RELU:
            return x > 0 ? x : 0;
        default:
            return x;
    }
}

double activation_derivative(double x, ActivationType type) {
    switch (type) {
        case ACTIVATION_SIGMOID:
            {
                double s = activation_function(x, ACTIVATION_SIGMOID);
                return s * (1 - s);
            }
        case ACTIVATION_TANH:
            {
                double t = tanh(x);
                return 1 - t * t;
            }
        case ACTIVATION_RELU:
            return x > 0 ? 1 : 0;
        default:
            return 1;
    }
}

Matrix* apply_activation(Activation* act, Matrix* m) {
    if (!act || !m) return NULL;
    
    Matrix* result = matrix_create(m->rows, m->cols);
    if (!result) return NULL;
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result->data[i][j] = activation_function(m->data[i][j], act->type);
        }
    }
    
    return result;
}

Matrix* apply_activation_derivative(Activation* act, Matrix* m) {
    if (!act || !m) return NULL;
    
    Matrix* result = matrix_create(m->rows, m->cols);
    if (!result) return NULL;
    
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            result->data[i][j] = activation_derivative(m->data[i][j], act->type);
        }
    }
    
    return result;
}
