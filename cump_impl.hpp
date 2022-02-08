#include <cuda_runtime.h>
#include <exception>
#include <list>
#include <mutex>

namespace cump_impl {



static size_t g_pitch = 0;

/****************************************************************************
* 内存片：内存块中，分配出去的或空闲的小段
*/
struct CumPiece {
	size_t size;
	void* pdata;
	void* pend;
};

/****************************************************************************
* 内存块，每个内存块有两个队列：空闲链表和已分配链表
*/
struct CumBlock {
	size_t size;
	void* pdata;
	void* pend;

	std::list<CumPiece*> listFree; // 空闲
	std::list<CumPiece*> listUsed; // 已分配

	size_t maxFree;
	size_t totalFree;
	size_t totalUsed;

	//// 对齐后的尾部
	//void* PitchEnd() {
	//	return (void*)((unsigned char*)(pdata)+(size + g_pitch - 1) / g_pitch * g_pitch);
	//}

	// 判断是否拥有这段内存
	bool Own(void* p) {
		return (p >= pdata && p < (unsigned char*)pdata + size);
	}

	bool Free(void* p)
	{
		// 从已分配的列表中查找
		auto iter = listUsed.begin();
		while (iter != listUsed.end())
		{
			if ((*iter)->pdata == p) break;
			iter++;
		}
		if (iter == listUsed.end()) return false;

		auto pPiece = *iter;
		listUsed.erase(iter);

		size_t pieceSize = pPiece->size;

		// 在空闲列表中查找是否有相邻的空闲块，如果有则合并空闲块
		CumPiece* pHead = nullptr;
		CumPiece* pTail = nullptr;
		iter = listFree.begin();
		while (iter != listFree.end())
		{
			// 是否跟上一个的结尾相邻
			if ((*iter)->pend == pPiece->pdata) {
				pHead = (*iter);
			}
			// 跟后一个的开头相邻
			if ((*iter)->pdata == pPiece->pend) {
				pTail = (*iter);
			}
			if (pHead && pTail) break;
			iter++;
		}

		// 合并
		if (pHead) {
			pPiece->pdata = pHead->pdata;
			pPiece->size = (char*)pPiece->pend - (char*)pPiece->pdata;
		}
		if (pTail) {
			pPiece->pend = pTail->pend;
			pPiece->size = (char*)pPiece->pend - (char*)pPiece->pdata;
		}

		// 释放
		if (pHead) {
			iter = listFree.begin();
			while (iter != listFree.end()) {
				if ((*iter)->pdata == pHead->pdata) {
					listFree.erase(iter);
					break;
				}
				iter++;
			}
			delete pHead;
			pHead = nullptr;
		}
		if (pTail) {
			iter = listFree.begin();
			while (iter != listFree.end()) {
				if ((*iter)->pdata == pTail->pdata) {
					listFree.erase(iter);
					break;
				}
				iter++;
			}
			delete pTail;
			pTail = nullptr;
		}

		// 插入
		listFree.push_front(pPiece);

		// 更新属性
		totalFree += pieceSize;
		totalUsed -= pieceSize;
		if (pPiece->size > maxFree) {
			maxFree = pPiece->size;
		}

		return true;
	}

	bool Malloc(void** p, size_t s) {
		s = (s + g_pitch - 1) / g_pitch * g_pitch;
		// 从空闲的列表中查找满足大小的 piece
		auto iter = listFree.begin();
		while (iter != listFree.end())
		{
			if ((*iter)->size >= s) break;
			iter++;
		}
		if (iter == listFree.end()) return false;

		// 找到之后切出来需要的大小（尾部切，不用操作链表）
		auto pPiece = *iter;

		// 更新属性
		totalFree -= s;
		totalUsed += s;

		// 如果大小相等
		if (s == pPiece->size) {
			listFree.erase(iter);
			listUsed.push_back(pPiece);
			*p = pPiece->pdata;
			return true;
		}

		// 大小不等
		pPiece->size -= s;
		pPiece->pend = (char*)pPiece->pdata + pPiece->size;
		auto pPNew = new CumPiece();
		pPNew->pdata = pPiece->pend;
		pPNew->size = s;
		pPNew->pend = (char*)pPNew->pdata + s;
		listUsed.push_back(pPNew);
		*p = pPNew->pdata;
		return true;
	}
};

cudaError CreateCumBlock(CumBlock** ppb ,size_t s)
{
	void* pdata = nullptr;
	auto err = cudaMalloc(&pdata, s);
	if (err != cudaSuccess) return err;

	s = (s + g_pitch - 1) / g_pitch * g_pitch;

	auto pP = new CumPiece();
	pP->pdata = pdata;
	pP->size = s;
	pP->pend = (unsigned char*)pdata + s;

	*ppb = new CumBlock();
	(*ppb)->size = s;
	(*ppb)->pdata = pdata;
	(*ppb)->pend = pP->pend;
	(*ppb)->maxFree = s;
	(*ppb)->totalFree = s;
	(*ppb)->totalUsed = 0;

	(*ppb)->listFree.push_back(pP);

	return cudaSuccess;
}

void DestroyCumBlock(CumBlock* pb)
{
	cudaFree(pb->pdata);
	delete pb;
}


/****************************************************************************
* 内存池实现
* 内存池中由多个内存块组成
* 分配内存的时候依次从每个内存块中进行查找空闲的内存片
*/
class CumpImpl
{
public:
	CumpImpl();
	~CumpImpl();
	
public:
	cudaError Malloc(void** p, size_t s);
	cudaError Free(void* devPtr);

	void SetBockSize(size_t s);
	size_t GetTotalUsed();

	void Destroy();

private:
	size_t m_pitch;
	size_t m_block_size;
	
private:
	std::list<CumBlock*> m_listBlock; // 内存块
	std::mutex m_lck;
};

CumpImpl::CumpImpl()
	:m_block_size(0x10000000) // 256MB
{
	std::lock_guard<std::mutex> lck(m_lck);

	void* ptr = nullptr;
	auto cuer = cudaMallocPitch(&ptr, &m_pitch, 1, 1);
	if (cuer == cudaSuccess) cudaFree(ptr);
	else throw(std::exception("cump init failed."));
	g_pitch = m_pitch;
}

CumpImpl::~CumpImpl()
{

}

/* 分配内存，从每个内存块中分配内存，
* 如果没有满足的内存块，则申请新的内存块进行分配
*/
cudaError CumpImpl::Malloc(void** p, size_t s)
{
	std::lock_guard<std::mutex> lck(m_lck);

	// try each block
	for (auto& pb : m_listBlock) {
		if (pb->Malloc(p, s)) return cudaSuccess;
	}
	// create new block
	if (s > m_block_size) m_block_size = s * 2;
	CumBlock* pb = nullptr;
	auto err = CreateCumBlock(&pb, m_block_size);
	if (cudaSuccess != err) return err;
	m_listBlock.push_front(pb);

	if (pb->Malloc(p, s)) return cudaSuccess;

	return cudaErrorMemoryAllocation;
}

cudaError CumpImpl::Free(void* devPtr)
{
	std::lock_guard<std::mutex> lck(m_lck);

	// 地址没有对齐
	if ((uint64_t)devPtr % g_pitch != 0) return  cudaErrorIllegalAddress;

	// try each block
	for (auto& pb : m_listBlock) {
		if (pb->Own(devPtr)){
			if (pb->Free(devPtr)) {
				return cudaSuccess;
			}
			else {
				return cudaErrorInvalidDevicePointer;
			}
		}
	}
	// 没有成功释放，内存地址应该不属于内存池
	return cudaErrorInvalidValue;
}

void CumpImpl::SetBockSize(size_t s)
{
	if (s == 0) throw(std::exception("block size can not set to 0."));
	m_block_size = s;
}


size_t CumpImpl::GetTotalUsed()
{
	std::lock_guard<std::mutex> lck(m_lck);

	size_t total = 0;
	for (auto& pb : m_listBlock) {
		total += pb->totalUsed;
	}
	return total;
}

void CumpImpl::Destroy()
{
	std::lock_guard<std::mutex> lck(m_lck);
	for (auto& pb : m_listBlock) {
		DestroyCumBlock(pb);
	}
	m_listBlock.swap(std::list<CumBlock*>());
}


}