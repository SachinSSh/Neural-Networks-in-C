#include <stdio.h>
#include "../include/error.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "../include/network.h"
#include "../include/matrix.h"
#include "../include/activation.h"
#include "../include/optimizer.h"
#include "../include/mnist_utils.h"

#define MNIST_IMAGE_SIZE 784  // 28x28 pixels
#define NUM_CLASSES 10        // digits 0-9
#define HIDDEN_SIZE 128       // size of hidden layer
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28

// Structure to hold model configuration
typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    ActivationType hidden_activation;
    ActivationType output_activation;
    OptimizerType optimizer;
    double learning_rate;
} ModelConfig;

// Function to load a saved model
NeuralNetwork* nn_load(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open model file %s\n", filename);
        return NULL;
    }

    // Read model configuration
    ModelConfig config;
    fread(&config, sizeof(ModelConfig), 1, file);

    // Create network with loaded configuration
    NeuralNetwork* nn = nn_create(config.input_size, config.hidden_size, 
                                 config.output_size, config.hidden_activation,
                                 config.output_activation, config.optimizer,
                                 config.learning_rate);
    if (!nn) {
        fclose(file);
        return NULL;
    }

    // Load weights and biases
    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->input_size; j++) {
            fread(&nn->hidden_weights->data[i][j], sizeof(double), 1, file);
        }
    }

    for (int i = 0; i < nn->hidden_size; i++) {
        fread(&nn->hidden_biases->data[i][0], sizeof(double), 1, file);
    }

    for (int i = 0; i < nn->output_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            fread(&nn->output_weights->data[i][j], sizeof(double), 1, file);
        }
    }

    for (int i = 0; i < nn->output_size; i++) {
        fread(&nn->output_biases->data[i][0], sizeof(double), 1, file);
    }

    fclose(file);
    return nn;
}

// Function to load and process a PGM image
Matrix* load_pgm_image(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Could not open image file %s\n", filename);
        return NULL;
    }

    // Read PGM header
    char magic[3];
    int width, height, maxval;
    fscanf(file, "%s\n%d %d\n%d\n", magic, &width, &height, &maxval);

    if (width != IMAGE_WIDTH || height != IMAGE_HEIGHT) {
        printf("Error: Image dimensions must be 28x28\n");
        fclose(file);
        return NULL;
    }

    // Create matrix for image data
    Matrix* image = matrix_create(MNIST_IMAGE_SIZE, 1);
    if (!image) {
        fclose(file);
        return NULL;
    }

    // Read image data
    unsigned char pixel;
    for (int i = 0; i < MNIST_IMAGE_SIZE; i++) {
        fread(&pixel, 1, 1, file);
        image->data[i][0] = (double)pixel / 255.0;  // Normalize to [0,1]
    }

    fclose(file);
    return image;
}

// Function to calculate accuracy on validation set
double calculate_accuracy(NeuralNetwork* nn, Matrix** inputs, unsigned char* labels, int num_samples) {
    int correct = 0;
    
    for (int i = 0; i < num_samples; i++) {
        Matrix* output = nn_feedforward(nn, inputs[i]);
        if (!output) continue;

        // Find predicted digit
        int predicted = 0;
        double max_val = output->data[0][0];
        for (int j = 1; j < NUM_CLASSES; j++) {
            if (output->data[j][0] > max_val) {
                max_val = output->data[j][0];
                predicted = j;
            }
        }

        if (predicted == labels[i]) {
            correct++;
        }

        matrix_free(output);
    }

    return (double)correct / num_samples;
}

// Function to predict digit from custom image
int predict_from_file(NeuralNetwork* nn, const char* image_path) {
    Matrix* input = load_pgm_image(image_path);
    if (!input) {
        return -1;
    }

    Matrix* output = nn_feedforward(nn, input);
    if (!output) {
        matrix_free(input);
        return -1;
    }

    // Find predicted digit
    int predicted = 0;
    double max_val = output->data[0][0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output->data[i][0] > max_val) {
            max_val = output->data[i][0];
            predicted = i;
        }
    }

    matrix_free(input);
    matrix_free(output);
    return predicted;
}

int main() {
    srand(time(NULL));

    // Create neural network
    NeuralNetwork* nn = nn_create(MNIST_IMAGE_SIZE, HIDDEN_SIZE, NUM_CLASSES,
                                 ACTIVATION_RELU,
                                 ACTIVATION_SIGMOID,
                                 OPTIMIZER_ADAM,
                                 0.001);

    // Load MNIST training data
    int num_train_images = 0;
    unsigned char* train_images = read_mnist_images("../mnist/train-images.idx3-ubyte", &num_train_images);
    int num_train_labels = 0;
    unsigned char* train_labels = read_mnist_labels("../mnist/train-labels.idx1-ubyte", &num_train_labels);

    // Load MNIST validation data
    int num_val_images = 0;
    unsigned char* val_images = read_mnist_images("../mnist/t10k-images.idx3-ubyte", &num_val_images);
    int num_val_labels = 0;
    unsigned char* val_labels = read_mnist_labels("../mnist/t10k-labels.idx1-ubyte", &num_val_labels);

    if (!train_images || !train_labels || !val_images || !val_labels) {
        printf("Failed to load MNIST data\n");
        return 1;
    }

    // Convert validation images to matrix format
    Matrix** val_matrices = malloc(num_val_images * sizeof(Matrix*));
    for (int i = 0; i < num_val_images; i++) {
        val_matrices[i] = image_to_matrix(&val_images[i * MNIST_IMAGE_SIZE]);
    }

    // Training
    printf("Training on MNIST dataset...\n");
    const int batch_size = 32;
    const int num_epochs = 10;
    const int validation_frequency = 1000;  // Validate every 1000 batches
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        int batch_count = 0;
        for (int i = 0; i < num_train_images; i += batch_size) {
            int current_batch_size = (i + batch_size > num_train_images) ? 
                                   (num_train_images - i) : batch_size;

            Matrix* input_batch[batch_size];
            Matrix* target_batch[batch_size];

            // Prepare batch
            for (int j = 0; j < current_batch_size; j++) {
                input_batch[j] = image_to_matrix(&train_images[(i + j) * MNIST_IMAGE_SIZE]);
                target_batch[j] = label_to_matrix(train_labels[i + j]);
            }

            // Train on batch
            Error* train_error = nn_train_batch(nn, input_batch, target_batch, current_batch_size);
            if (train_error) {
                error_print(train_error);
                error_free(train_error);
            }

            // Cleanup batch
            for (int j = 0; j < current_batch_size; j++) {
                matrix_free(input_batch[j]);
                matrix_free(target_batch[j]);
            }

            // Validate periodically
            batch_count++;
            if (batch_count % validation_frequency == 0) {
                double accuracy = calculate_accuracy(nn, val_matrices, val_labels, num_val_images);
                printf("Epoch %d, Batch %d: Validation accuracy = %.2f%%\n", 
                       epoch + 1, batch_count, accuracy * 100);
            }
        }

        // Validate at end of epoch
        double accuracy = calculate_accuracy(nn, val_matrices, val_labels, num_val_images);
        printf("Epoch %d completed. Validation accuracy = %.2f%%\n", 
               epoch + 1, accuracy * 100);
    }

    Error* save_error = nn_save(nn, "mnist_model.nn");
    if (save_error) {
        error_print(save_error);
        error_free(save_error);
    } else {
        printf("Saved model to 'mnist_model.nn'\n");
    }

    printf("\nTesting custom image prediction...\n");
    NeuralNetwork* loaded_nn = nn_load("mnist_model.nn");
    if (loaded_nn) {
        const char* test_image_path = "../mnist/test_digit.pgm";
        int predicted = predict_from_file(loaded_nn, test_image_path);
        if (predicted >= 0) {
            printf("Predicted digit: %d\n", predicted);
        }
        nn_free(loaded_nn);
    }

   
    for (int i = 0; i < num_val_images; i++) {
        matrix_free(val_matrices[i]);
    }
    free(val_matrices);
    free(train_images);
    free(train_labels);
    free(val_images);
    free(val_labels);
    nn_free(nn);

    return 0;
}
