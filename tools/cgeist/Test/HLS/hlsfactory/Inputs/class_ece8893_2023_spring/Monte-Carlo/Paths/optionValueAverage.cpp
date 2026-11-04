#include "./optionValueAverage.hpp"

void generatePath(double path[STEPS], double S0,  double f1, double f2){
    #pragma HLS INLINE
    path[0] = S0*exp(f1+f2*path[0]); //S0
    for(int i = 1; i < STEPS; i++) {
        path[i] = path[i-1]*exp(f1+f2*path[i]);
    }
}

void loadPaths(double paths_from_mem[NUM_PATHS][STEPS], double paths_buff[N_BUFF][STEPS], int starting_idx) {
    #pragma HLS INLINE OFF
    FROM_MEM_OUTER:
    for(int i = 0; i < N_BUFF; i++) {
        FROM_MEM_INNER :
        for(int j = 0; j < STEPS; j++) {
            paths_buff[i][j] = paths_from_mem[i+starting_idx][j];
        }
    }
}


void writePaths(double paths_to_mem[NUM_PATHS][STEPS], double paths_buff[N_BUFF][STEPS], int starting_idx) {
        #pragma HLS INLINE OFF
        TO_MEM_OUTER:
        for(int i = 0; i < N_BUFF; i++) {
            TO_MEM_INNER :
            for(int j = 0; j < STEPS; j++) {
                paths_to_mem[i+starting_idx][j] = paths_buff[i][j];
            }
        }
        
}
void optionValue(double paths[NUM_PATHS][STEPS], double params[4]){
    #pragma HLS INTERFACE m_axi depth=1  port=paths  bundle=pths
    #pragma HLS INTERFACE m_axi depth=1  port=params  bundle=prms

    double S0 = params[0];
    double T = params[1];
    double SIGMA = params[2];
    double MU  = params[3];

    double dt = T/STEPS;
    double path[N_BUFF][STEPS]; //This is the B-RAM Buffer.
    double f1 = (MU - (0.5*pow(SIGMA,2)))*dt;
    double f2 = SIGMA*sqrt(dt);


    //NUM_PATHS/N_BUFF is the number of times we increment.
    //Consequently, you will want NUM_PATHS to be divisible by N_BUFF
    for(int idx = 0; idx < NUM_PATHS/N_BUFF; idx++) {
        
        //Loads paths from DRAM to BRAM
        loadPaths(paths, path, idx*N_BUFF);

        //Operates on paths stored in BRAM
        #pragma HLS ARRAY_PARTITION variable=path cyclic dim=1 factor=2
        GENERATE_PATH: 
        for(int i = 0; i < N_BUFF; i++) {
            #pragma HLS pipeline
            generatePath(path[i], S0, f1, f2);
        }
     
        //Writes Results from BRAM back to DRAM
        writePaths(paths, path, idx*N_BUFF);
    }

}

