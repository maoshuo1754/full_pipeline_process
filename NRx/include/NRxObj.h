#ifndef NRXOBJ_H
#define NRXOBJ_H

#include "NRxType.h"
#include "NRxUtile.h"

#ifdef BUILDBASEDLL
#ifdef WIN32
#define DLLAPI __declspec(dllexport)
#else
#define DLLAPI __attribute__((visibility("default")))
#endif
#else
#ifdef WIN32
#define DLLAPI __declspec(dllimport)
#else
#define DLLAPI __attribute__((visibility("default")))
#endif
#endif

// Leading events
typedef enum
{
    NRX_EVENT_INVALID   = 0, // invalid.
    NRX_EVENT_CLOSE     = 1, // close NRx SDK.
    NRX_EVENT_CLOSE_RADAR_SOURCE = 2, // close radar data source.
    NRX_EVENT_ACTIVE_RADAR_SOURCE = 3 // activate radar data source.
}NRxEvent;

// State of NRxSDK
typedef enum
{
    NRX_SDK_UNINI   = 0,  // Un-initialized.
    NRX_SDK_NOAUTH  = 1, // Run without authorized.
    NRX_SDK_AUTH    = 2,   // Run with authorized.
    NRX_SDK_EXIT    = 3   // Start to exit.
}NRxState;

// Base class for NRxSDK
class DLLAPI NRxObj
{
public:
    NRxObj(const char *name);// construct by name
    virtual ~NRxObj(void);// deconstructor
private:
    NRxObj(void) = delete;// forbid default constructor
    NRxObj(const NRxObj &) = delete;// forbid copy constructor
    NRxObj(NRxObj &&) = delete;// forbid copy constructor by right val
    NRxObj &&operator=(const NRxObj &) = delete;// forbid operator=()
    NRxObj &&operator=(NRxObj &&) = delete;// forbid operator=() by right val

public:
    const char *Name(void)const
    {
        return m_name.c_str();
    }

    // notify key event
    virtual void Notify(const NRxEvent event, void *param = nullptr) = 0;

    // 检查常量是否发生变化
    bool CheckProtectedConst(void)const;

    // destroy objs of derived from class NRxObj
    static void Destroy(void);

    // 退出其它数据源，激活指定数据源
    static int32 SelectSource(uint32 code);

    static NRxState State(void)
    {
        return m_state;
    }

    static void SetState(const NRxState &state)
    {
        m_state = state;
    }

    static bool IsRun(void)
    {
        return ((NRX_SDK_NOAUTH == m_state) || (NRX_SDK_AUTH == m_state));
    }
    static bool isTestTime()
    {
        return (1 == m_isTestTime);
    }
    static void SetIsTestTime(const int32 &isTestTime);

private:
    uint32 m_protectNo;// 类 NRxObj 中前4各字节
    std::string m_name;// 仅仅为了区分，可以不唯一

    static std::atomic<NRxState>    m_state;// state of NRxSDK
    static list<NRxObj *> m_objs;// 存储所有NRxObj对象的地址
    static int32 m_isTestTime;// 是否测试时间，由集成设置在需要测试时间的代码中获取
};
/* 全局函数声明 */
// 初始化NRxSDK
extern DLLAPI void NRxInit(void);

// 销毁 NRxObj子类的对象
extern DLLAPI void NRxDestroyTracker(void);

// select radar data source
extern DLLAPI int32 NRxSelectSource(uint32 code);

#endif // NRXOBJ_H
