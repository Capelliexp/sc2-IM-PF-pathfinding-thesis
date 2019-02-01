#pragma once

#include "CUDA/cuda_header.cuh"
#include "CUDA/map_storage.hpp"

#include <stdio.h>
#include <iostream>
#include <string>
#include <time.h>
#include <math.h>

class CUDAWrapper {
public:
	CUDAWrapper(MapStorage* maps);
    ~CUDAWrapper();

    void Update(clock_t dt_ticks);
private:
    int* data;
	MapStorage* map_storage;
};

CUDAWrapper::CUDAWrapper(MapStorage* maps) {
    std::cout << "Initializing CUDA object" << std::endl;

    data = new int[THREADS_IN_GRID];
    for (int i = 0; i < THREADS_IN_GRID; ++i) data[i] = i;

    InitializeCUDA(data);
}

CUDAWrapper::~CUDAWrapper() {}

void CUDAWrapper::Update(clock_t dt_ticks) {
	//float dt = ((float)dt_ticks) / CLOCKS_PER_SEC;	//get dt in seconds


}