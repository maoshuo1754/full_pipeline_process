#include "NRxUdpSender.h"
#include "udpSenderManager.h"
#include <iostream>

int udpSender::addUdpSender(const std::string& name, const std::string& localIP,
                            unsigned short localPort, const int bufLen)
{
    if (isValidIP(localIP)) {
        return UdpSenderMgr::addSender(name, localIP, localPort, bufLen);
    } else {
        std::cout << "invalid localIP format in addUdpSender(): " << localIP << std::endl;
        return 3;
    }
}

int udpSender::setDstAddress(const std::string& name, const std::string& dstIP,
                             unsigned short dstPort)
{
    if (isValidIP(dstIP)) {
        return UdpSenderMgr::setSenderDstAddress(name, dstIP, dstPort);
    } else {
        std::cout << "invalid dstIP format in setDstAddress(): " << dstIP << std::endl;
        return 3;
    }
}

int udpSender::udpSendMsg(const void* msgBuf, const int msgLen, const std::string& dstIP,
                          unsigned short dstPort, const std::string& udpName)
{
    if (isValidIP(dstIP)) {
        return UdpSenderMgr::udpSendMsg(msgBuf, msgLen, dstIP, dstPort, udpName);
    } else {
        std::cout << "invalid dstIP format in udpSendMsg(): " << dstIP << std::endl;
        return 3;
    }
}

int udpSender::udpSendMsg(const void* msgBuf, const int msgLen, const std::string& udpName)
{
    return UdpSenderMgr::udpSendMsg(msgBuf, msgLen, udpName);
}

void udpSender::udpDestory(const string& name)
{
    UdpSenderMgr::destoryUdp();
}
