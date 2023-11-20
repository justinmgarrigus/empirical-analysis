// Parses a logfile for data about the executions of an automata.
// Usage: ./parse (file.log) (iterations per execution) (executions)

#include <stdio.h> 
#include <assert.h> 
#include <stdlib.h>

int main(int argc, const char** argv) {
    assert(argc == 4); 

    int iterations = atoi(argv[2]);
    int executions = atoi(argv[3]);

    int *areas = malloc(sizeof(int) * iterations);
    int *entropies = malloc(sizeof(int) * iterations); 
    float *densities = malloc(sizeof(float) * iterations); 

    float average_area = 0; 
    float average_entropy = 0;
    float average_density = 0; 

    FILE *log = fopen(argv[1], "rb"); 
    for (int ex = 0; ex < executions; ex++) {
        // Read one execution from the file. 
        int seed; 
        fread(&seed, sizeof(int), 1, log); 
        fread(areas, sizeof(int), iterations, log); 
        fread(entropies, sizeof(int), iterations, log); 
        fread(densities, sizeof(float), iterations, log); 

        // Add those values to the statistic.
        for (int i = 0; i < iterations; i++) {
            average_area += (float)areas[i] / (iterations * executions); 
            average_entropy += (float)entropies[i] / (iterations * executions); 
            average_density += densities[i] / (iterations * executions); 
        }
    }
    fclose(log); 

    printf("Average area: %f\n", average_area); 
    printf("Average entropy: %f\n", average_entropy); 
    printf("Average density: %f\n", average_density);
}
