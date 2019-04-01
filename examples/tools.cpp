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

	std::cout << "Memory usage" << location << ":" << std::endl <<
		//"   Virtual memory size (swap file + RAM): " << totalVirtualMem << std::endl <<
		//"   Virtual memory currently used: " << virtualMemUsed << std::endl <<
		"   Virtual memory currently used by process: " << virtualMemUsedByMe << std::endl <<
		//"   Physical memory size: " << totalPhysMem << std::endl <<
		//"   Physical memory currently used: " << physMemUsed << std::endl <<
		"   Physical memory currently used by process: " << physMemUsedByMe << std::endl;
}