#ifndef MNIST_UTILS_H
#define MNIST_UTILS_H

#include "matrix.h"

unsigned char* read_mnist_images(const char* filename, int* num_images);
unsigned char* read_mnist_labels(const char* filename, int* num_labels);
Matrix* image_to_matrix(unsigned char* image_data);
Matrix* label_to_matrix(unsigned char label);

#endif
