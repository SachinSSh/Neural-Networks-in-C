// src/matrix.c
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h> // For strcmp
#include "../include/matrix.h"
#include "../include/error.h"

Matrix* matrix_create(int rows, int cols) {
    Matrix* m = malloc(sizeof(Matrix));
    if (m == NULL) {
        return NULL;
    }
    
    m->rows = rows;
    m->cols = cols;
    
    // Allocate rows
    m->data = malloc(rows * sizeof(double*));
    if (m->data == NULL) {
        free(m);
        return NULL;
    }
    

    for (int i = 0; i < rows; i++) {
        m->data[i] = malloc(cols * sizeof(double));
        if (m->data[i] == NULL) {
            // Free previously allocated memory
            for (int j = 0; j < i; j++) {
                free(m->data[j]);
            }
            free(m->data);
            free(m);
            return NULL;
        }
        // Initialize to zeros
        for (int j = 0; j < cols; j++) {
            m->data[i][j] = 0.0;
        }
    }
    
    return m;
}

void matrix_free(Matrix* m) {
    if (m == NULL) return;
    
    if (m->data != NULL) {
        for (int i = 0; i < m->rows; i++) {
            free(m->data[i]);
        }
        free(m->data);
    }
    free(m);
}

Matrix* matrix_multiply(Matrix* a, Matrix* b) {
    if (a->cols != b->rows) {
        return NULL; // Dimension mismatch
    }

    Matrix* result = matrix_create(a->rows, b->cols);
    if (result == NULL) return NULL;

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i][k] * b->data[k][j];
            }
            result->data[i][j] = sum;
        }
    }

    return result;
}

void matrix_add_inplace(Matrix* a, Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return; // Dimension mismatch
    }

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            a->data[i][j] += b->data[i][j];
        }
    }
}

Matrix* matrix_subtract(Matrix* a, Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return NULL; // Dimension mismatch
    }

    Matrix* result = matrix_create(a->rows, a->cols);
    if (result == NULL) return NULL;

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }

    return result;
}

double matrix_sum_squared(Matrix* a) {
    double sum = 0.0;
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            sum += a->data[i][j] * a->data[i][j];
        }
    }
    return sum;
}

Matrix* matrix_hadamard(Matrix* a, Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return NULL; // Dimension mismatch
    }

    Matrix* result = matrix_create(a->rows, a->cols);
    if (result == NULL) return NULL;

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] * b->data[i][j];
        }
    }

    return result;
}

Matrix* matrix_multiply_transpose(Matrix* a, Matrix* b) {
    if (a->cols != b->cols) {
        return NULL; // Dimension mismatch
    }

    Matrix* result = matrix_create(a->rows, b->rows);
    if (result == NULL) return NULL;

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->rows; j++) {
            double sum = 0.0;
            for (int k = 0; k < a->cols; k++) {
                sum += a->data[i][k] * b->data[j][k];
            }
            result->data[i][j] = sum;
        }
    }

    return result;
}

Matrix* matrix_transpose(Matrix* a) {
    Matrix* result = matrix_create(a->cols, a->rows);
    if (result == NULL) return NULL;

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[j][i] = a->data[i][j];
        }
    }

    return result;
}

void matrix_scale(Matrix* a, double scale) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            a->data[i][j] *= scale;
        }
    }
}

void matrix_subtract_inplace(Matrix* a, Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return; // Dimension mismatch
    }

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            a->data[i][j] -= b->data[i][j];
        }
    }
}



Matrix* activate(Matrix* input, const char* activation) {
    Matrix* result = matrix_create(input->rows, input->cols);
    if (result == NULL) return NULL;

    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            double value = input->data[i][j];
            if (strcmp(activation, "sigmoid") == 0) {
                result->data[i][j] = 1.0 / (1.0 + exp(-value));
            } else if (strcmp(activation, "relu") == 0) {
                result->data[i][j] = fmax(0.0, value);
            } else if (strcmp(activation, "tanh") == 0) {
                result->data[i][j] = tanh(value);
            } else {
                // Handle unsupported activation functions
                matrix_free(result);
                return NULL;
            }
        }
    }
    return result;
}

Matrix* activate_derivative(Matrix* input, const char* activation) {
    Matrix* result = matrix_create(input->rows, input->cols);
    if (result == NULL) return NULL;

    for (int i = 0; i < input->rows; i++) {
        for (int j = 0; j < input->cols; j++) {
            double value = input->data[i][j];
            if (strcmp(activation, "sigmoid") == 0) {
                double sigmoid = 1.0 / (1.0 + exp(-value));
                result->data[i][j] = sigmoid * (1.0 - sigmoid);
            } else if (strcmp(activation, "relu") == 0) {
                result->data[i][j] = (value > 0.0) ? 1.0 : 0.0;
            } else if (strcmp(activation, "tanh") == 0) {
                double tanh_val = tanh(value);
                result->data[i][j] = 1.0 - tanh_val * tanh_val;
            } else {
                // Handle unsupported activation functions
                matrix_free(result);
                return NULL;
            }
        }
    }
    return result;
}
