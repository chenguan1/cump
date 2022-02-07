
#include <cuda_runtime.h>
#include "cump.h"

#include <queue>
#include <iostream>
#include <chrono>
#include <Windows.h>

#define CHECK_CUDA(call)                                                        \
    {                                                                           \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess)                                                  \
        {                                                                       \
            std::cout << "CUDA Runtime failure: '#" << _e << "' at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
            exit(1);                                                            \
        }                                                                       \
    }

using namespace std;
const size_t block_size = 2448 * 2048 * 3;


void test_native()
{
	void* p = nullptr;
	queue<void*> buffers;
	while (buffers.size() < 300)
	{
		// malloc 10
		for (int i = 0; i < 10; i++) {
			CHECK_CUDA(cudaMalloc(&p, block_size));
			CHECK_CUDA(cudaMemset(p, 0x56 ,block_size));
			buffers.push(p);
		}

		// free 5
		for (int i = 0; i < 5; i++) {
			CHECK_CUDA(cudaFree(buffers.front()));
			buffers.pop();
		}
	}

	// free all
	while (!buffers.empty())
	{
		CHECK_CUDA(cudaFree(buffers.front()));
		buffers.pop();
	}
}

void test_pool()
{
	void* p = nullptr;
	queue<void*> buffers;
	while (buffers.size() < 300)
	{
		// malloc 10
		for (int i = 0; i < 10; i++) {
			CHECK_CUDA(cumpMalloc(&p, block_size));
			CHECK_CUDA(cudaMemset(p, 0x56, block_size));
			buffers.push(p);
		}

		// free 5
		for (int i = 0; i < 5; i++) {
			CHECK_CUDA(cumpFree(buffers.front()));
			buffers.pop();
		}
	}

	// free all
	while (!buffers.empty())
	{
		CHECK_CUDA(cumpFree(buffers.front()));
		buffers.pop();
	}
}


int main()
{
	auto start = GetTickCount64();
	test_pool();
	cout << "pool cost: " << GetTickCount64() - start << endl;
	cumpDestroy();

	start = GetTickCount64();
	test_native();
	cout << "native cost: " << GetTickCount64() - start << endl;

	cin.ignore();
	return 0;
}
