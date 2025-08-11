// src/error.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/error.h"

Error* error_create(ErrorCode code, const char* message) {
    Error* error = malloc(sizeof(Error));
    if (error == NULL) {
        return NULL;
    }
   
    // Allocate memory for the message string
    char* msg_copy = strdup(message);
    if (msg_copy == NULL) {
        free(error);
        return NULL;
    }
   
    error->message = msg_copy;
    error->code = code;
    return error;
}

void error_print(Error* error) {
    if (error != NULL) {
        fprintf(stderr, "Error %d: %s\n", error->code, error->message);
    }
}

void error_free(Error* error) {
    if (error != NULL) {
        free((void*)error->message);  
        free(error);
    }
}

ErrorCode save_error(Error* error) {
    if (error == NULL) {
        return ERROR_NULL_POINTER;
    }
    return ERROR_SUCCESS; 
}
