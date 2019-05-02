#pragma once
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>

#define TEST_START_NR (0)

void PrintMemoryUsage(std::string location = "");
int GetMemoryUsage();
void PrintDataToFile(float* frame_data, int* RAM_data, int* VRAM_data, int length, std::string file_name, bool close);