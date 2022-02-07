#include "cump.h"
#include "cump_impl.hpp"

static CumpImpl cump;

cudaError cumpMalloc(void** p, size_t s)
{
	return cump.Malloc(p, s);
}

cudaError cumpFree(void* devPtr)
{
	return cump	.Free(devPtr);
}


void cumpSetBlockSize(size_t s)
{
	cump.SetBockSize(s);
}

void cumpDestroy()
{
	cump.Destroy();
}

size_t cumpGetUsedSize()
{
	return cump.GetTotalUsed();
}

