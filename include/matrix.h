#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int rows;
    int cols;
    double** data;
} Matrix;

// Function to create a new matrix
Matrix* matrix_create(int rows, int cols);

// Function to free a matrix
void matrix_free(Matrix* m);

// Matrix multiplication: result = a * b
Matrix* matrix_multiply(Matrix* a, Matrix* b);

// In-place matrix addition: a += b
void matrix_add_inplace(Matrix* a, Matrix* b);

// Matrix subtraction: result = a - b
Matrix* matrix_subtract(Matrix* a, Matrix* b);

// Sum of squared elements of a matrix
double matrix_sum_squared(Matrix* a);

// Element-wise (Hadamard) multiplication: result = a ⊙ b
Matrix* matrix_hadamard(Matrix* a, Matrix* b);

// Matrix multiplication with transpose: result = a * b^T
Matrix* matrix_multiply_transpose(Matrix* a, Matrix* b);

// Transpose of a matrix: result = a^T
Matrix* matrix_transpose(Matrix* a);
Matrix* activate(Matrix* input, const char* activation);
Matrix* activate_derivative(Matrix* input, const char* activation);

// Scale a matrix in-place: a *= scale
void matrix_scale(Matrix* a, double scale);

// In-place matrix subtraction: a -= b
void matrix_subtract_inplace(Matrix* a, Matrix* b);

#endif
