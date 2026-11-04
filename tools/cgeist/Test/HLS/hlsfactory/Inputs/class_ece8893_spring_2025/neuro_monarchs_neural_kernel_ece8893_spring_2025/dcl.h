#ifndef LOAD_BINARY_H
#define LOAD_BINARY_H

// Loads a 4D binary file into a dynamically allocated 4D array of floats.
// The dimensions (C1, C2, C3, C4) must be provided.
// Returns a pointer to the allocated 4D array, or nullptr on error.
float**** loadBinary(const char* filename, int C1, int C2, int C3, int C4);

#endif // LOAD_BINARY_H
