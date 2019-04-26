#pragma once
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>

void PrintMemoryUsage(std::string location = "");
void PrintFrameTimesToFile(float* data, int length, std::string file_name);