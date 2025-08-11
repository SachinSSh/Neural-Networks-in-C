#include "../include/mnist_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/matrix.h"

// Helper function to reverse bytes for big-endian to little-endian conversion
static int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

unsigned char* read_mnist_images(const char* filename, int* num_images) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return NULL;
    }

    int magic_number = 0;
    int n_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    // Read file header
    fread(&magic_number, sizeof(int), 1, file);
    fread(&n_images, sizeof(int), 1, file);
    fread(&n_rows, sizeof(int), 1, file);
    fread(&n_cols, sizeof(int), 1, file);

    // Convert from big-endian to little-endian
    magic_number = reverse_int(magic_number);
    n_images = reverse_int(n_images);
    n_rows = reverse_int(n_rows);
    n_cols = reverse_int(n_cols);

    *num_images = n_images;

    // Allocate memory for image data
    unsigned char* images = malloc(n_images * n_rows * n_cols);
    if (!images) {
        fclose(file);
        return NULL;
    }

    // Read image data
    fread(images, 1, n_images * n_rows * n_cols, file);
    fclose(file);

    return images;
}

unsigned char* read_mnist_labels(const char* filename, int* num_labels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return NULL;
    }

    int magic_number = 0;
    int n_labels = 0;

    // Read file header
    fread(&magic_number, sizeof(int), 1, file);
    fread(&n_labels, sizeof(int), 1, file);

    // Convert from big-endian to little-endian
    magic_number = reverse_int(magic_number);
    n_labels = reverse_int(n_labels);

    *num_labels = n_labels;

    // Allocate memory for labels
    unsigned char* labels = malloc(n_labels);
    if (!labels) {
        fclose(file);
        return NULL;
    }

    // Read label data
    fread(labels, 1, n_labels, file);
    fclose(file);

    return labels;
}

Matrix* image_to_matrix(unsigned char* image_data) {
    Matrix* matrix = matrix_create(784, 1);  // 28x28 = 784
    if (!matrix) return NULL;

    for (int i = 0; i < 784; i++) {
        matrix->data[i][0] = (double)image_data[i] / 255.0;  // Normalize to [0,1]
    }

    return matrix;
}

Matrix* label_to_matrix(unsigned char label) {
    Matrix* matrix = matrix_create(10, 1);  // One-hot encoding for digits 0-9
    if (!matrix) return NULL;

    // Initialize all elements to 0
    for (int i = 0; i < 10; i++) {
        matrix->data[i][0] = 0.0;
    }

    // Set the corresponding digit position to 1
    if (label < 10) {
        matrix->data[label][0] = 1.0;
    }

    return matrix;
}
