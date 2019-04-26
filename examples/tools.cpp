#include "tools.h"

#include "windows.h"
#include "psapi.h"
#include <iostream>
#include <string>

void PrintMemoryUsage(std::string location) {
	if (location != "") {
		location = " at " + location;
	}

	MEMORYSTATUSEX memInfo;
	memInfo.dwLength = sizeof(MEMORYSTATUSEX);
	if (GlobalMemoryStatusEx(&memInfo) == 0)
		std::cout << "WARNING: unable to read virt memory info" << std::endl;

	PROCESS_MEMORY_COUNTERS_EX pmc;
	if (GetProcessMemoryInfo(GetCurrentProcess(), (PPROCESS_MEMORY_COUNTERS)&pmc, sizeof(pmc)) == 0)
		std::cout << "WARNING: unable to read phys memory info" << std::endl;

	DWORDLONG totalVirtualMem = memInfo.ullTotalPageFile;
	DWORDLONG virtualMemUsed = memInfo.ullTotalPageFile - memInfo.ullAvailPageFile;
	SIZE_T virtualMemUsedByMe = pmc.PrivateUsage;
	DWORDLONG totalPhysMem = memInfo.ullTotalPhys;
	DWORDLONG physMemUsed = memInfo.ullTotalPhys - memInfo.ullAvailPhys;
	SIZE_T physMemUsedByMe = pmc.WorkingSetSize;

	std::cout << "Host memory usage" << location << ":" << std::endl <<
		//"   Virtual memory size (swap file + RAM): " << totalVirtualMem << " bytes" << std::endl <<
		//"   Virtual memory currently used: " << virtualMemUsed << " bytes" << std::endl <<
		"   Virtual memory currently used by process: " << virtualMemUsedByMe << " bytes" << std::endl <<
		//"   Physical memory size: " << totalPhysMem << " bytes" << std::endl <<
		//"   Physical memory currently used: " << physMemUsed << " bytes" << std::endl <<
		"   Physical memory currently used by process: " << physMemUsedByMe << " bytes" << std::endl;
}

void PrintFrameTimesToFile(float* data, int length, std::string file_name) {
	std::ofstream file(file_name);

	if (!file.is_open()) std::cout << "FPS output FAILED" << std::endl;

	for (int i = 0; i < length; ++i) {
		file << data[i] << "\n";
	}

	file.close();
}