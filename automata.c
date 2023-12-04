// Compile with "-DDRAW" if an output gif should be created.
// This collects a set of statistics each frame of execution, including
// distance, entropy, and density.
// Usage: 
//   ./automata (rule) (seed)
// Parameters: 
//   (rule): string in the format B123/S456, where 123 and 456 can be any 
//     combination of digits between 1 and 8, inclusive. This defines a ruleset
//     the automaton will follow. For example, Conway's Game of Life is B3/S23.
//   (seed): optional integer between 0 and UINT_MAX defining the range of 
//     initial states we should test. We start with an initial state of 0 and 
//     do states up to the number of executions. The size of the bitmask is 
//     equal to this variable in bits. The least-significant bit is the top-
//     left corner of the automata, the next bit is the cell to the right, etc.

#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
#include <limits.h> 
#include <float.h>
#include <stdint.h> 
#include <assert.h> 
#include <string.h> 
#include <stdbool.h> 
#include <math.h> 
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h> 
#include "hashset.h" 

#ifndef ROWS
    #define ROWS 48
#endif
#ifndef COLUMNS 
    #define COLUMNS 48
#endif

#define PADDING 16 
#ifndef ITERATIONS
    #define ITERATIONS 1
#endif 
#ifndef EXECUTIONS 
    #define EXECUTIONS 1
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
    ge_GIF* create_gif(char*, unsigned int); 
    void draw(ge_GIF*, data_t*);
    void close_gif(ge_GIF*);
    #define CREATE_GIF_FUNC(a, b) (create_gif(a, b))
    #define DRAW_FUNC(a, b) (draw(a, b))
    #define CLOSE_GIF_FUNC(a) (close_gif(a))
#else
    typedef void *ge_GIF; // Unused;
    #define CREATE_GIF_FUNC(a) NULL
    #define DRAW_FUNC(a, b) NOP
    #define CLOSE_GIF_FUNC(a) NOP 
#endif 

// "data" is the pointer to the current frame, while "buffer" is used during 
// the iteration process to create the next frame.
data_t *data;
data_t *buffer; 

bool rule_birth[9] = { false };
bool rule_stay[9]  = { false };

// Tests if a cell is alive given the ruleset passed as a command-line 
// argument.
int test_alive(int alive, int neighbors) {
    if (alive) return rule_stay[neighbors]; 
    else       return rule_birth[neighbors];
}

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
            if (test_alive(alive, neighbors)) set_bit(buffer, r * COLUMNS + c); 
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
    uint64_t area = (max_r - min_r + 1) * (max_c - min_c + 1); 
    if (max_r - min_r < 0) return 0; // True if there are no cells in the grid.
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
    else return (double)set / a; 
}

// Returns the distance between two points on a grid such that the diagonal 
// distance between two adjacent points is 1 (instead of sqrt(2)).
double diagonal_distance(int x1, int y1, int x2, int y2) {
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1); 
    int min = dx < dy ? dx : dy; 
    int max = dx > dy ? dx : dy; 
    int diagonal_steps = min; 
    int straight_steps = max - min; 
    return sqrt(2.0) * diagonal_steps + straight_steps;
}

// Returns the clumpiness of the grid, defined as the average smallest distance
// between each cell. High clumpiness has a low value, indicating cells are 
// placed very close to each other.
double clumpiness(data_t* data) {
    double dist_sum = 0; 
    int num_cells = 0; 
    for (int rs = 0; rs < ROWS; rs++) {
        for (int cs = 0; cs < COLUMNS; cs++) {
            // If the source row/column is set.
            if (test_bit(data, rs * COLUMNS + cs)) {
                // Find the smallest distance from this cell to another.
                unsigned int min_dist = UINT_MAX;
                for (int rd = 0; rd < ROWS; rd++) {
                    for (int cd = 0; cd < COLUMNS; cd++) {
                        if ((rd != rs || cd != cs) && 
                            test_bit(data, rd * COLUMNS + cd)) 
                        {
                            unsigned int dist = 
                                diagonal_distance(rs, cs, rd, cd); 
                            if (dist < min_dist) 
                                min_dist = dist;
                        }
                    }
                }
                dist_sum += min_dist; 
                num_cells++; 
            }
        }
    }
    if (num_cells != 0) 
        return dist_sum / num_cells;
    else 
        return 0;
}

#ifdef DRAW
ge_GIF* create_gif(char* rule, unsigned int seed) {
    // Create the "gifs" directory if it doesn't exist. 
    struct stat st = {0}; 
    if (stat("gif/", &st) == -1)
        mkdir("gif/", 0700);

    char fname[64];
    sprintf(fname, "gifs/%s-%u.gif", rule, seed);
    return ge_new_gif(
        fname, OUTPUT_WIDTH, OUTPUT_HEIGHT, 
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
    data_t* data, data_t* buffer, int seed, char* rule, 
    double* initial_clumpiness, uint64_t* initial_count, 
    uint64_t* areas, uint64_t* entropies, double* densities, 
    unsigned int initial_state, const unsigned int initial_state_size_bits) 
{
    // The name of the gif is dependent on the seed.
    ge_GIF *gif; 

    // Set the initial state in the automata.  
    uint64_t counter = 0;
    if (initial_state_size_bits > 0) {
        gif = CREATE_GIF_FUNC(rule, initial_state);

        // We want to center the cells in the middle of the grid.
        int max_size = initial_state_size_bits;
        int root_max_size = (int)sqrt(max_size);
        const int row_start = ROWS / 2 - root_max_size / 2, 
            col_start = COLUMNS / 2 - root_max_size / 2; 
        for (int i = 0; i < root_max_size; i++) {
            for (int j = 0; j < root_max_size; j++) {
                int pos = (row_start + i) * COLUMNS + (col_start + j);
                if (initial_state & (1 << (i * root_max_size + j))) {
                    set_bit(data, pos); 
                    counter++; 
                }
            }
        }
    }
    else {
        gif = CREATE_GIF_FUNC(rule, seed);

        // Get a good initial state for the generator. The seed for the state
        // will be a 32 bit random string. 
        srand(seed);
        for (int i = PADDING; i < ROWS - PADDING; i++) {
            for (int j = PADDING; j < COLUMNS - PADDING; j++) {
                if (rand() % 2) { 
                    set_bit(data, i * COLUMNS + j); 
                    counter++;
                } 
            }
        }
    }
    
    *initial_clumpiness = clumpiness(data);
    *initial_count = counter;

    DRAW_FUNC(gif, data); 
    int iter; 
    for (iter = 0; iter < ITERATIONS; iter++) {
        iterate(); 
        DRAW_FUNC(gif, data);

        areas[iter] = area(data); 
        entropies[iter] = entropy(data, buffer);
        densities[iter] = density(data);

        if (areas[iter] == 0) 
            break; // Prevents having to simulate no cells on the grid. 
    }

    // If we stopped early (due to there being no cells left on the grid), then
    // fill the rest of the data with empty values.
    for (; iter < ITERATIONS; iter++) {
        areas[iter] = 0; 
        entropies[iter] = 0; 
        densities[iter] = 0; 
    }

    CLOSE_GIF_FUNC(gif);
}

int main(int argc, char** argv) { 
    if (argc < 2) {
        fprintf(stderr, "Error: you must pass in an argument containing the " \
            "ruleset for the automaton. Example: B2S23\n");
        exit(1); 
    }

    // Parse the ruleset.
    char *rule = argv[1]; 
    char *birth_str = strstr(rule, "B") + 1; 
    char *stay_str = strstr(rule, "S")  + 1;
    char *end_str = rule + strlen(argv[1]);
    if (!(birth_str < stay_str && 
          stay_str <= end_str))
    {
        fprintf(stderr, "Error: automaton ruleset is improperly formed.\n"); 
        exit(1);
    }

    for (char* cdx = birth_str; cdx < stay_str - 1; cdx++) {
        if (*cdx < 48 || *cdx > 56) // Allowed: 0, 1, 2, 3, 4, 5, 6, 7, 8
            fprintf(stderr, "Error: ruleset contains illegal characters.\n");
        rule_birth[*cdx - 48] = true; 
    }
    for (char* cdx = stay_str; cdx < end_str; cdx++) {
        if (*cdx < 48 || *cdx > 56) // Allowed: 0, 1, 2, 3, 4, 5, 6, 7, 8
            fprintf(stderr, "Error: ruleset contains illegal characters.\n");
        rule_stay[*cdx - 48] = true;
    }

    // Allocate the size nearest to an unsigned int. 
    int s = sizeof(data_t);
    int multiple = ((ROWS * COLUMNS + s - 1) / s) * s; 
    data = malloc(multiple);
    buffer = malloc(multiple);
    
    double initial_clumpiness;
    uint64_t initial_count;
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
    const char *fname = (argc >= 3) ? argv[2] : "log.txt";  
    FILE *log = fopen(fname, "wb+");

    // The beginning of the file contains the number of executions performed.
    // We won't execute simulations that are identical to executions already 
    // performed, so the number of simulations may not equal EXECUTIONS. 
    // Although, we won't know how many executions are performed until we 
    // actually run the program, so we put a placeholder at the beginning
    // and then replace that placeholder after we finish the simulations. 
    uint64_t num_executions_performed = 0; 
    fwrite(&num_executions_performed, sizeof(uint64_t), 1, log);

    unsigned int initial_state_size_bits = 0;
    hashset_t set = NULL; 
    if (argc >= 4) {
        initial_state_size_bits = atoi(argv[3]);
        set = hashset_create(); 
    }
    
    for (uint64_t i = 0; i < EXECUTIONS; i++) {
        uint64_t start = i;
        if (set != NULL) {
            // Check the hashset to see if we have already performed a test 
            // that is identical to the one we are about to perform. If we 
            // have, then skip it.

            // The seed ("start") can be visualized as a (size*size) grid, 
            // where "size" is equal to sqrt(initial_state). We should push
            // the activated cells in that grid as far left and as far up as
            // possible, then compare it to cells that already exist in the 
            // hashset.
            unsigned int root_size = 
                (unsigned int)sqrt(initial_state_size_bits);
            
            // To push the cells left, we find the bit that is activated the 
            // furthest to the right. Pretend the number is split into (root)
            // groups of (root) bits each. We can right-shift the number that
            // many times. To push up cells, we find the bit that is activated
            // the closest to the top, and right-shift the number (root) times 
            // that value. 
            int rightmost = root_size;
            int topmost = root_size; 
            for (int r = 0; r < root_size; r++) {
                for (int c = 0; c < root_size; c++) {
                    int x = r * root_size + c;
                    if (start & (1 << x)) {
                        if (c < rightmost) 
                            rightmost = c; 
                        if (r < topmost)
                            topmost = r; 
                    }
                }
            }
            start >>= rightmost;
            start >>= root_size * topmost;

            // If that number already exists in our set, then we've done a 
            // simulation identical to this one before. We will gain no new 
            // information from performing this automata.
            if (hashset_is_member(set, (void*)start)) 
                continue; 
            else 
                hashset_add(set, (void*)start); 
        }
        
        // Clear the buffers. 
        memset(data, 0, multiple); 
        memset(buffer, 0, multiple);
        
        uint64_t seed; 
        if (initial_state_size_bits != 0) 
            seed = start; 
        else 
            seed = 
                ((rand() & 0xFF)      ) | 
                ((rand() & 0xFF) <<  8) |
                ((rand() & 0xFF) << 16) | 
                ((rand() & 0xFF) << 24);
        
        execute(
            data, buffer, seed, rule,  
            &initial_clumpiness, &initial_count, 
            areas, entropies, densities,
            start, initial_state_size_bits);
        
        fwrite(&seed, sizeof(uint64_t), 1, log); 
        fwrite(&initial_clumpiness, sizeof(double), 1, log);
        fwrite(&initial_count, sizeof(uint64_t), 1, log);
        fwrite(areas, sizeof(uint64_t), ITERATIONS, log); 
        fwrite(entropies, sizeof(uint64_t), ITERATIONS, log); 
        fwrite(densities, sizeof(double), ITERATIONS, log);
        num_executions_performed++; 
    } 
    
    // Rewrite the placeholder at the beginning with the number of executions
    // actually performed.
    fseek(log, 0, SEEK_SET); 
    fwrite(&num_executions_performed, sizeof(uint64_t), 1, log); 

    fclose(log);  
    free(data); 
}
