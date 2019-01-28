#pragma once

#include "CUDA/host.cu"

#include <stdio.h>
#include <string>
#include <iostream>

class CUDA_wrapper {
public:
    CUDA_wrapper();
    ~CUDA_wrapper();

    void Update();
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

void CUDA_wrapper::Update(){}