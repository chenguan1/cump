/*
* cuda �Դ��
* ȫ���滻 cudaMalloc �� cudaFree
*/

#include <cuda_runtime.h>

cudaError cumpMalloc(void** p, size_t s);
cudaError cumpFree(void* devPtr);

void cumpSetBlockSize(size_t s);

size_t cumpGetUsedSize(); 