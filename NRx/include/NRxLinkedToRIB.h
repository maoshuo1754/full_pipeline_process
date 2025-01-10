#ifndef NRXLINKEDTORIB_H
#define NRXLINKEDTORIB_H

#include "NRxObj.h"

// 前置声明

// class NRxLinkedToRIB
class DLLAPI NRxLinkedToRIB : public NRxObj
{
public:
    // 构造函数
    NRxLinkedToRIB(const char *name, const NRxLinkedToRIB *parent)
        : NRxObj(name)
        , m_parent(nullptr)
    {
        m_parent = parent;
    }

    // 析构函数
    virtual ~NRxLinkedToRIB(void) {}

private:
    NRxLinkedToRIB(void) = delete;// forbid default constructor
    NRxLinkedToRIB(const NRxLinkedToRIB &) = delete;// forbid copy constructor
    NRxLinkedToRIB(NRxLinkedToRIB &&) = delete;// forbid copy constructor by right val
    NRxLinkedToRIB &&operator=(const NRxLinkedToRIB &) = delete;// forbid operator=()
    NRxLinkedToRIB &&operator=(NRxLinkedToRIB &&) = delete;// forbid operator=() by right val

public:

    // 缓存通知更新数据
    virtual void UpdateData(const char *buf, const int32 bytes) = 0;

    const NRxLinkedToRIB *GetParent(void)const
    {
        return m_parent;
    }
private:
    const NRxLinkedToRIB *m_parent;
};
#endif // NRXLINKEDTORIB_H
