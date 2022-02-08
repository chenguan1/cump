#include <cuda_runtime.h>
#include <exception>
#include <list>
#include <mutex>

namespace cump_impl {



static size_t g_pitch = 0;

/****************************************************************************
* �ڴ�Ƭ���ڴ���У������ȥ�Ļ���е�С��
*/
struct CumPiece {
	size_t size;
	void* pdata;
	void* pend;
};

/****************************************************************************
* �ڴ�飬ÿ���ڴ�����������У�����������ѷ�������
*/
struct CumBlock {
	size_t size;
	void* pdata;
	void* pend;

	std::list<CumPiece*> listFree; // ����
	std::list<CumPiece*> listUsed; // �ѷ���

	size_t maxFree;
	size_t totalFree;
	size_t totalUsed;

	//// ������β��
	//void* PitchEnd() {
	//	return (void*)((unsigned char*)(pdata)+(size + g_pitch - 1) / g_pitch * g_pitch);
	//}

	// �ж��Ƿ�ӵ������ڴ�
	bool Own(void* p) {
		return (p >= pdata && p < (unsigned char*)pdata + size);
	}

	bool Free(void* p)
	{
		// ���ѷ�����б��в���
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

		// �ڿ����б��в����Ƿ������ڵĿ��п飬�������ϲ����п�
		CumPiece* pHead = nullptr;
		CumPiece* pTail = nullptr;
		iter = listFree.begin();
		while (iter != listFree.end())
		{
			// �Ƿ����һ���Ľ�β����
			if ((*iter)->pend == pPiece->pdata) {
				pHead = (*iter);
			}
			// ����һ���Ŀ�ͷ����
			if ((*iter)->pdata == pPiece->pend) {
				pTail = (*iter);
			}
			if (pHead && pTail) break;
			iter++;
		}

		// �ϲ�
		if (pHead) {
			pPiece->pdata = pHead->pdata;
			pPiece->size = (char*)pPiece->pend - (char*)pPiece->pdata;
		}
		if (pTail) {
			pPiece->pend = pTail->pend;
			pPiece->size = (char*)pPiece->pend - (char*)pPiece->pdata;
		}

		// �ͷ�
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

		// ����
		listFree.push_front(pPiece);

		// ��������
		totalFree += pieceSize;
		totalUsed -= pieceSize;
		if (pPiece->size > maxFree) {
			maxFree = pPiece->size;
		}

		return true;
	}

	bool Malloc(void** p, size_t s) {
		s = (s + g_pitch - 1) / g_pitch * g_pitch;
		// �ӿ��е��б��в��������С�� piece
		auto iter = listFree.begin();
		while (iter != listFree.end())
		{
			if ((*iter)->size >= s) break;
			iter++;
		}
		if (iter == listFree.end()) return false;

		// �ҵ�֮���г�����Ҫ�Ĵ�С��β���У����ò�������
		auto pPiece = *iter;

		// ��������
		totalFree -= s;
		totalUsed += s;

		// �����С���
		if (s == pPiece->size) {
			listFree.erase(iter);
			listUsed.push_back(pPiece);
			*p = pPiece->pdata;
			return true;
		}

		// ��С����
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
* �ڴ��ʵ��
* �ڴ�����ɶ���ڴ�����
* �����ڴ��ʱ�����δ�ÿ���ڴ���н��в��ҿ��е��ڴ�Ƭ
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
	std::list<CumBlock*> m_listBlock; // �ڴ��
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

/* �����ڴ棬��ÿ���ڴ���з����ڴ棬
* ���û��������ڴ�飬�������µ��ڴ����з���
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

	// ��ַû�ж���
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
	// û�гɹ��ͷţ��ڴ��ַӦ�ò������ڴ��
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