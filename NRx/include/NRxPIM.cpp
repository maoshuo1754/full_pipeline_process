#include <cstring>
#include "NRxPIM.h"

NRx8BitPulse::NRx8BitPulse()
{
    data = nullptr;
    m_pulseNo = -1; // invalid  20220325
    m_dataLen = 0;
}

NRx8BitPulse::NRx8BitPulse(const NRx8BitPulse& src)
    : ifheader(src.ifheader)
    , vidinfo(src.vidinfo)
    , data(src.data)
    , ifend(src.ifend)
    , m_pulseNo(src.m_pulseNo)
    , m_dataLen(src.m_dataLen)
{
}

NRx8BitPulse::NRx8BitPulse(NRx8BitPulse&& src)
    : ifheader(src.ifheader)
    , vidinfo(src.vidinfo)
    , data(std::move(src.data))
    , ifend(src.ifend)
    , m_pulseNo(src.m_pulseNo)
    , m_dataLen(src.m_dataLen)
{
}

NRx8BitPulse::~NRx8BitPulse(void)
{
    if (data != nullptr)
    {
        delete []data;
        data = nullptr;
    }
}

NRx8BitPulse& NRx8BitPulse::operator=(const NRx8BitPulse& src)
{
    if (&src != this)
    {
        ifheader = src.ifheader;
        vidinfo = src.vidinfo;
        if(data == nullptr)
            data = new uint8[src.m_dataLen];
        else if(m_dataLen < src.m_dataLen)
        {
            delete []data;
            data = new uint8[src.m_dataLen];
        }
        memcpy(data, src.data, sizeof(uint8) * src.m_dataLen);
        ifend = src.ifend;
        m_pulseNo = src.m_pulseNo;
        m_dataLen = src.m_dataLen;
    }
    return *this;
}

NRx8BitPulse& NRx8BitPulse::operator=(NRx8BitPulse&& src)
{
    if (&src != this)
    {
        ifheader = src.ifheader;
        vidinfo = src.vidinfo;
        if(data != nullptr)
            delete []data;
        data = std::move(src.data);
        ifend = src.ifend;
        m_pulseNo = src.m_pulseNo;
        m_dataLen = src.m_dataLen;
    }
    return *this;
}

//NRx8BitPulse &NRx8BitPulse::operator=(const NRx16BitPulse &src)
//{
//    ifheader = src.ifheader;
//    vidinfo = src.vidinfo;
//    for(uint32 i=0;i<src.vidinfo.cellNum;i++)
//    {
//        data[i]=src.data[i];
//    }
//    ifend = src.ifend;
//    m_pulseNo = src.m_pulseNo;
//}

//NRx8BitPulse &NRx8BitPulse::operator=(NRx16BitPulse &&src)
//{
//    ifheader = src.ifheader;
//    vidinfo = src.vidinfo;
//    for(uint32 i=0;i<src.vidinfo.cellNum;i++)
//    {
//        data[i]=src.data[i];
//    }
//    ifend = src.ifend;
//    m_pulseNo = src.m_pulseNo;
//}

void NRx8BitPulse::CreateData(int32 length)
{
    data = new uint8[length];
    memset(data, 0, sizeof(uint8) * length);
    m_dataLen = length;
}
