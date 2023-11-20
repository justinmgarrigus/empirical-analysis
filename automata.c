// Compile with "-DDRAW" if an output gif should be created.
// This collects a set of statistics each frame of execution, including
// distance, entropy, and density.
// Also include either "-DCONWAYLIFE", "-DLIFEWITHOUTDEATH", or 
// "-DDAYANDNIGHT".

#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
#include <limits.h> 
#include <stdint.h> 
#include <assert.h> 

int test_conwaylife(int, int); 
int test_lifewithoutdeath(int, int); 
int test_dayandnight(int, int);

// The automata to choose from. 
#if defined(CONWAYLIFE)
    int (*test_func)(int, int) = &test_dayandnight;
#elif defined(LIFEWITHOUTDEATH) 
    int (*test_func)(int, int) = &test_lifewithoutdeath; 
#elif defined(DAYANDNIGHT)
    int (*test_func)(int, int) = &test_dayandnight;
#else 
    #error Define either CONWAYLIFE, LIFEWITHOUTDEATH, or DAYANDNIGHT. 
#endif

#ifndef ROWS
    #define ROWS 48
#endif
#ifndef COLUMNS 
    #define COLUMNS 48
#endif

#define PADDING 16 
#ifndef ITERATIONS
    #define ITERATIONS 25
#endif 
#ifndef EXECUTIONS 
    #define EXECUTIONS 1000
#endif
#define MILLIS_PER_ITER 100 

#define OUTPUT_WIDTH 500 
#define OUTPUT_HEIGHT 400
#define CELL_WIDTH (OUTPUT_WIDTH / COLUMNS) 
#define CELL_HEIGHT (OUTPUT_HEIGHT / ROWS) 

typedef unsigned data_t; 
#define DATA_SIZE_BITS (sizeof(unsigned) * 8)

// If we want to create gifs for each iteration, we can use the DRAW_FUNC
// macro which either points to a gif-creating function or does nothing (NOP).
#define NOP do {} while(0)
#ifdef DRAW 
    #include "gifenc.h" 
    ge_GIF* create_gif(); 
    void draw(ge_GIF*, data_t*);
    void close_gif(ge_GIF*);
    #define CREATE_GIF_FUNC() (create_gif())
    #define DRAW_FUNC(a, b) (draw(a, b))
    #define CLOSE_GIF_FUNC(a) (close_gif(a))
#else
    typedef void *ge_GIF; // Unused;
    #define CREATE_GIF_FUNC() NULL
    #define DRAW_FUNC(a, b) NOP
    #define CLOSE_GIF_FUNC(a) NOP 
#endif 

// Conway's Game of Life: B3/S23
int test_conwaylife(int alive, int neighbors) {
    if (alive) return neighbors == 2 || neighbors == 3; 
    else return neighbors == 3; 
}

// Life Without Death: B3/S012345678
int test_lifewithoutdeath(int alive, int neighbors) {
    if (alive) return 1;
    else return neighbors == 3;
}

// Day and Night: B3678/S34678
int test_dayandnight(int alive, int neighbors) {
    if (alive) return neighbors == 3 || neighbors >= 6;
    else return neighbors == 3 || neighbors == 4 || neighbors >= 6; 
}

// "data" is the pointer to the current frame, while "buffer" is used during 
// the iteration process to create the next frame.
data_t *data;
data_t *buffer; 

// Sets the kth bit in the data. 
int set_bit(data_t* data, int k) {
    data[k / DATA_SIZE_BITS] |= 1 << (k % DATA_SIZE_BITS); 
}

// Unsets the kth bit in the data. 
int unset_bit(data_t* data, int k) {
    data[k / DATA_SIZE_BITS] &= ~(1 << (k % DATA_SIZE_BITS)); 
}

// Tests if the kth bit in the data is set or unset. 
int test_bit(data_t* data, int k) {
    return (data[k / DATA_SIZE_BITS] & (1 << (k % DATA_SIZE_BITS))) != 0; 
}

// Simulates one iteration.
void iterate() {
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLUMNS; c++) {
            // How many neighbors does this cell have? 
            int neighbors = 0; 
            for (int i = -1; i <= 1; i++) 
                for (int j = -1; j <= 1; j++) 
                    if (i != 0 || j != 0) 
                        if (r + i >= 0 && r + i < ROWS && 
                            c + j >= 0 && c + j < COLUMNS && 
                            test_bit(data, (r + i) * COLUMNS + c + j))
                                neighbors++; 
           
            int alive = test_bit(data, r * COLUMNS + c); 
            if (test_func(alive, neighbors)) set_bit(buffer, r * COLUMNS + c); 
            else unset_bit(buffer, r * COLUMNS + c);
        }
    }

    // Swap buffer and data. 
    data_t *temp = data; 
    data = buffer; 
    buffer = temp;
}

// Returns the area of the set points on the grid. Essentially found by finding
// the bounds (min r, max r, min c, max c) of the set cells on the grid and 
// squaring the distance.
uint64_t area(data_t* data) {
    uint64_t min_r = ROWS, max_r = 0, min_c = COLUMNS, max_c = 0; 
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLUMNS; c++) {
            if (test_bit(data, r * COLUMNS + c)) {
                if (r < min_r) min_r = r; 
                if (r > max_r) max_r = r; 
                if (c < min_c) min_c = c; 
                if (c > max_c) max_c = c;
            }
        }
    }
    uint64_t area = (max_r - min_r) * (max_c - min_c); 
    assert(area < ROWS * COLUMNS); 
    return area; 
}

// Returns the entropy of the grid. This is how many of the cells changed 
// values between the current and previous iterations.
uint64_t entropy(data_t* current_frame, data_t* previous_frame) {
    uint64_t entropy = 0; 
    for (int r = 0; r < ROWS; r++) 
        for (int c = 0; c < COLUMNS; c++) 
            entropy += 
                test_bit(current_frame,  r * COLUMNS + c) != 
                test_bit(previous_frame, r * COLUMNS + c); 
    return entropy;
}

// Returns the density of the grid. This is the number of cells that are set 
// divided by the area() of the grid.
double density(data_t* data) {
    int set = 0; 
    for (int r = 0; r < ROWS; r++) 
        for (int c = 0; c < COLUMNS; c++) 
            set += test_bit(data, r * COLUMNS + c); 
    
    uint64_t a = area(data); 
    if (a == 0) return 0; 
    else return (float)set / a; 
}

#ifdef DRAW
ge_GIF* create_gif() {
    return ge_new_gif(
        "automata.gif", OUTPUT_WIDTH, OUTPUT_HEIGHT, 
        (uint8_t []) {
            0x00, 0x00, 0x00, // black
            0xFF, 0xFF, 0xFF  // white
        }, 
        2, -1, 0
    );
}

void draw(ge_GIF* gif, data_t* data) {
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLUMNS; c++) {
            int k = r * COLUMNS + c;
            int color = test_bit(data, k); 
            int r0 = r * CELL_HEIGHT;
            int c0 = c * CELL_WIDTH;
            for (int i = 0; i < CELL_HEIGHT; i++)
                for (int j = 0; j < CELL_WIDTH; j++)
                    gif->frame[(r0 + i) * OUTPUT_WIDTH + c0 + j] = color;
        }
    }
    ge_add_frame(gif, MILLIS_PER_ITER / 10); 
}

void close_gif(ge_GIF* gif) {
    ge_close_gif(gif);
}
#endif 

void execute(
    data_t* data, data_t* buffer, int seed, 
    uint64_t* areas, uint64_t* entropies, double* densities) 
{
    ge_GIF *gif = CREATE_GIF_FUNC();

    // Get a good initial state for the generator. The seed for the state will
    // be a 32 bit random string. 
    srand(seed); 

    // Set the initial state in the automata.  
    for (int i = PADDING; i < ROWS - PADDING; i++) 
        for (int j = PADDING; j < COLUMNS - PADDING; j++)
            if (rand() % 2) 
                set_bit(data, i * COLUMNS + j); 
            else
                unset_bit(data, i * COLUMNS + j); 

    DRAW_FUNC(gif, data); 
    for (int i = 0; i < ITERATIONS; i++) {
        iterate(); 
        DRAW_FUNC(gif, data);

        areas[i] = area(data); 
        entropies[i] = entropy(data, buffer);
        densities[i] = density(data);
    }

    CLOSE_GIF_FUNC(gif);
}

int main(int argc, const char** argv) { 
    // Allocate the size nearest to an unsigned int. 
    int s = sizeof(data_t);
    int multiple = ((ROWS * COLUMNS + s - 1) / s) * s; 
    data = malloc(multiple);
    buffer = malloc(multiple);
    
    uint64_t *areas = malloc(sizeof(uint64_t) * ITERATIONS); 
    uint64_t *entropies = malloc(sizeof(uint64_t) * ITERATIONS); 
    double *densities = malloc(sizeof(double) * ITERATIONS); 
    
    // Assert that the maximum value for the area, entropy, and density will
    // fit within their containers. 
    assert((uint64_t)ROWS * COLUMNS < 0xFFFFFFFF); 

    // This seed will be overwritten each iteration. 
    srand(time(NULL));

    // If the file name is specified as an argument, then write to that as the 
    // log file. Otherwise, set it to default "log.txt".
    const char *fname = (argc == 2) ? argv[1] : "log.txt"; 

    FILE *log = fopen(fname, "wb");
    for (int i = 0; i < EXECUTIONS; i++) {
        unsigned int seed = 
            ((rand() & 0xFF)      ) | 
            ((rand() & 0xFF) <<  8) |
            ((rand() & 0xFF) << 16) | 
            ((rand() & 0xFF) << 24);
        execute(data, buffer, seed, areas, entropies, densities);
        fwrite(&seed, sizeof(int), 1, log); 
        fwrite(areas, sizeof(uint64_t), ITERATIONS, log); 
        fwrite(entropies, sizeof(uint64_t), ITERATIONS, log); 
        fwrite(densities, sizeof(double), ITERATIONS, log);
    } 
    fclose(log);  
    free(data); 
}
