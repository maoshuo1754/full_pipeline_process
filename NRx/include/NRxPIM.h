#ifndef NRXPIM_H
#define NRXPIM_H

//#include "NRxLinkedToRIB.h"
#include "NRxInterface.h"

using namespace NRxIf;

class NRx8BitPulse
{
public:
    NRx8BitPulse();//创建时需另外调用CreateData来给data分配内存
    NRx8BitPulse(const NRx8BitPulse& src);
    NRx8BitPulse(NRx8BitPulse&& src);
    ~NRx8BitPulse(void);
    NRx8BitPulse& operator=(const NRx8BitPulse& src);
    NRx8BitPulse& operator=(NRx8BitPulse&& src);
//    NRx8BitPulse& operator=(const NRx16BitPulse& src);
//    NRx8BitPulse& operator=(NRx16BitPulse&& src);

    void CreateData(int32 length);

    void SetIfHeader(NRxIfHeader header)
    {
        ifheader=header;
    }
    NRxIfHeader GetIfHeader()
    {
        return ifheader;
    }
    void SetVidInfo(NRxVidInfo vid)
    {
        vidinfo=vid;
    }
    NRxVidInfo GetVidInfo()
    {
        return vidinfo;
    }
    void SetData(uint8 * data_in)
    {
        data=data_in;
    }
    uint8 * GetData()
    {
        return data;
    }
    void SetIfEnd(NRxIfEnd end)
    {
        ifend=end;
    }
    NRxIfEnd GetIfEnd()
    {
        return ifend;
    }
    void SetPulseNo(int32 pulseNo)
    {
        m_pulseNo = pulseNo;
    }

public:
    NRxIfHeader ifheader;
    NRxVidInfo vidinfo;
    uint8* data;
    NRxIfEnd ifend;
    int32 m_pulseNo;
    int32 m_dataLen;
};

#endif // NRXPIM_H
