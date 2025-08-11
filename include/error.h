#ifndef ERROR_H
#define ERROR_H

#include <stddef.h>

typedef enum {
    ERROR_SUCCESS = 0,
    ERROR_NULL_POINTER,
    ERROR_FILE_IO,
    ERROR_MEMORY = 1,
    ERROR_INVALID_ARGUMENT =3,
} ErrorCode;

typedef struct {
    ErrorCode code;
    const char* message;
} Error;

Error* error_create(ErrorCode code, const char* message);
void error_print(Error* error);
void error_free(Error* error);
ErrorCode save_error(Error* error);

#endif
