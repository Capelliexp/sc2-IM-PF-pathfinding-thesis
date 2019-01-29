#pragma once

#include "CUDA/cuda_header.cuh"

#include <stdio.h>
#include <iostream>
#include <string>
#include <time.h>
#include <math.h>

class CUDA_wrapper {
public:
    CUDA_wrapper();
    ~CUDA_wrapper();

    void Update(clock_t dt_ticks);
private:
    int* data;
};

CUDA_wrapper::CUDA_wrapper() {
    std::cout << "Initializing CUDA object" << std::endl;

    data = new int[THREADS_IN_GRID];
    for (int i = 0; i < THREADS_IN_GRID; ++i) data[i] = i;

    InitializeCUDA(data);
}

CUDA_wrapper::~CUDA_wrapper() {}

void CUDA_wrapper::Update(clock_t dt_ticks){
	//float dt = ((float)dt_ticks) / CLOCKS_PER_SEC;
}