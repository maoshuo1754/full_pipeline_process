#include "udpSenderManager.h"
#include <cstring>
#include <sstream>
#include <iostream>
#include <regex>

UdpMapWrap UdpSenderMgr::udpMap;

///
/// \brief UdpSender::UdpSender, create udp sender with assigned localIP, localPort,
///     buffer length and name.
/// \param name, name of udp sender
/// \param localIP, local IP to bind
/// \param localPort, local port to bind
/// \param bufLen, buffer length of udp sender
///
UdpSender::UdpSender(const std::string &name, const std::string &localIP,
                     unsigned short localPort, int bufLen)
    : m_sockName(name)
{
    m_sender.socket = -1;
    memset(&m_sender.bindAddr, 0, sizeof(sockaddr_in));
    memset(&m_sender.remoteAddr, 0, sizeof(sockaddr_in));

    setLocalAddress(localIP, localPort);
    initUdp(bufLen);
}

///
/// \brief UdpSender::~UdpSender, release udp resource
///
UdpSender::~UdpSender()
{
    close();
}

///
/// \brief UdpSender::setDstAddress, set default destination address to use when
///     send msgs.
/// \param dstIP, destination IP
/// \param dstPort, destination port
///
void UdpSender::setDstAddress(const string& dstIP, unsigned short dstPort)
{
    m_sender.remoteAddr.sin_family = AF_INET;
    m_sender.remoteAddr.sin_addr.s_addr = inet_addr(dstIP.c_str());
    m_sender.remoteAddr.sin_port = htons(dstPort);
}

///
/// \brief UdpSender::sendTo, send out msg to default assigned IP address
/// \param msgBuf, message to send
/// \param msgLen, message length
/// \return
///     sendLen, send out msg length if socket is valid;
///     -1, if socket is invalid
///
int UdpSender::sendTo(const void* msgBuf, int msgLen)
{
    if(UdpSender::isValid()) {
        return sendto(m_sender.socket, (char*)msgBuf, msgLen, 0,
                      (sockaddr*)(&m_sender.remoteAddr), sizeof(sockaddr_in));
    } else {
        std::cout << "udp " << m_sockName << "'s' socket is invalid." << std::endl;
        return -1;
    }
}

///
/// \brief UdpSender::sendTo, send out msg with assigned IP address
/// \param msgBuf, message to send
/// \param msgLen, message length
/// \param dstIP, destination IP
/// \param dstPort, destination port
/// \return
///     sendLen, send out msg length if socket is valid;
///     -1, if socket is invalid
///
int UdpSender::sendTo(const void* msgBuf, int msgLen, const string& dstIP,
                      unsigned short dstPort)
{
    sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(dstIP.c_str());
    addr.sin_port = htons(dstPort);

    if(UdpSender::isValid()) {
        return sendto(m_sender.socket, (char*)msgBuf, msgLen, 0,
                      (sockaddr*)(&addr), sizeof(sockaddr_in));
    } else {
        std::cout << "udp " << m_sockName << "'s' socket is invalid." << std::endl;
        return -1;
    }
}

///
/// \brief UdpSender::printUdpInfo, print udp sender info
///
void UdpSender::printUdpInfo()
{
    std::stringstream msg;
    msg << "udp Name: " << m_sockName << ", local IP: " << (int)(m_sender.bindAddr.sin_addr.s_addr & 0xFF)
        << "." << (int)((m_sender.bindAddr.sin_addr.s_addr >> 8) & 0xFF)
        << "." << (int)((m_sender.bindAddr.sin_addr.s_addr >> 16) & 0xFF)
        << "." << (int)((m_sender.bindAddr.sin_addr.s_addr >> 24) & 0xFF)
        << ", local Port: " << ntohs(m_sender.bindAddr.sin_port) << std::endl
        << "          dst IP: " << (int)(m_sender.remoteAddr.sin_addr.s_addr & 0xFF)
        << "." << (int)((m_sender.remoteAddr.sin_addr.s_addr >> 8) & 0xFF)
        << "." << (int)((m_sender.remoteAddr.sin_addr.s_addr >> 16) & 0xFF)
        << "." << (int)((m_sender.remoteAddr.sin_addr.s_addr >> 24) & 0xFF)
        << ", dst Port: " << ntohs(m_sender.remoteAddr.sin_port) << std::endl;
    std::cout << msg.str();
}

///
/// \brief UdpSender::isValid, if udp sender is valid
/// \return
///     true, if udp sender is valid;
///     false, if udp sender is invalid.
///
bool UdpSender::isValid(void)
{
    return m_sender.socket > 0;
}

///
/// \brief UdpSender::close, release udp resource, if fail, will print error code
///
void UdpSender::close()
{
    if (m_sender.socket > 0) {
        int sockErr;
#ifdef WIN32
        sockErr = ::closesocket(m_sender.socket);
#else
        sockErr = ::close(m_sender.socket);
#endif
        if (sockErr < 0) {
            std::cout << "close fail: " << sockErr << std::endl;
            printUdpInfo();
        }
        m_sender.socket = -1;
    } else {
        ; // invalid socket
    }
}

///
/// \brief UdpSender::setLocalAddress, set local address to bind with socket
/// \param localIP, local IP
/// \param localPort, local port
///
void UdpSender::setLocalAddress(const string& localIP, unsigned short localPort)
{
    m_sender.bindAddr.sin_family = AF_INET; // address format, in winsock, only AF_INET can be used
    m_sender.bindAddr.sin_addr.s_addr = inet_addr(localIP.c_str());
    m_sender.bindAddr.sin_port = htons(localPort);
}

///
/// \brief UdpSender::initInWin32, register socket in WIN32 platform
/// \return
///     true, if register success;
///     false, if register false.
///
bool UdpSender::initInWin32()
{
#ifdef WIN32 // ONLY VALID IN WIN32
    static volatile bool isIni(false);
    WORD wVersionRequested;
    WSADATA wsaData;
    int err =0;
    wVersionRequested = MAKEWORD( 2, 2 );

    err = WSAStartup( wVersionRequested, &wsaData );

    if ( err == 0 ) {
        if ( LOBYTE( wsaData.wVersion ) != 2 ||	HIBYTE( wsaData.wVersion ) != 2 ) {
            WSACleanup();
            printf("socket ver error!\n");
            isIni = false;
            return isIni;
        } else {
//            printf("socket in WIN32 start up!\n");
            isIni = true;
            return isIni;
        }
    } else {
        printf("socket load error !\n");
        isIni = false;
        return isIni;
    }
#endif
}

///
/// \brief UdpSender::initUdp, create udp sender socket and bind with local IP
///     and port with assigned buffer length
/// \param bufLen, buffer length of udp sender
/// \param setBroadCast, if broad cast sender, not support in current version
/// \return
///     true, if create socket and bind with IP success;
///     false, if not.
///
bool UdpSender::initUdp(int bufLen, int setBroadCast)
{
    // if need assign socket version, use code below
#ifdef WIN32
    if (!initInWin32()) {
        std::cout << "udp register FAIL under win32!" << std::endl;
        return false;
    }
#endif
    // address format; socket type; protocal according to socket type
    int& udpSock = m_sender.socket;
    udpSock = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udpSock <= 0) {

        std::cout << "socket initial FAIL: ";
        printUdpInfo();
        return false;
    }

    if (setBroadCast) { // set broadcast, in this lib, not use
        bool broadCast(true);
        int sockErr = ::setsockopt(udpSock, SOL_SOCKET, SO_BROADCAST, (const char*)&broadCast, sizeof(bool));
        if (sockErr != 0) {
            std::cout << "set broadcast FAIL: ";
            printUdpInfo();
            close();
            return false;
        }
    }

    // set socket buffer size and bind local IP&port
    ::setsockopt(udpSock, SOL_SOCKET, SO_SNDBUF, (const char*)&bufLen, sizeof(int));
    int sockErr = ::bind(udpSock, (sockaddr*)(&m_sender.bindAddr), sizeof(sockaddr_in));
    if (sockErr != 0) {
        std::cout << "socket initial FAIL: ";
        printUdpInfo();
        close();
        return false;
    }

    std::cout << "socket initial success: ";
    printUdpInfo();
    return true;
}

UdpMapWrap::UdpMapWrap()
{
}

UdpMapWrap::~UdpMapWrap()
{
}

///
/// \brief UdpMapWrap::sendMsg, send out msg with udp sender has udpName, to dstIP
///     and dstPort.
/// \param msgBuf, message to send
/// \param msgLen, message length
/// \param dstIP, destination IP
/// \param dstPort, destination port
/// \param udpName, name of udp sender
/// \return
///     0, if send success;
///     1, if send out length != msgLen
///     2, if udp with udpName not found in m_udpMap
///
int UdpMapWrap::sendMsg(const void* msgBuf, const int msgLen, const string& dstIP,
                        unsigned short dstPort, const string& udpName)
{
    std::lock_guard<std::mutex> guard(m_lockWrap);

    int ret(0);
    auto udpMapIte = m_udpMap.find(udpName);
    if(udpMapIte != m_udpMap.end ()) {
        // assigned udp initialized
        UdpSender* sender = udpMapIte->second;
        int sendLen = sender->sendTo(msgBuf, msgLen, dstIP, dstPort);

        if (sendLen != msgLen) {
            std::cout << "Send failed, msg len is: " << msgLen
                      << "send out len is: " << sendLen
                      << " in udp: " << udpName << std::endl;
            ret = 1;
        }
    } else {
        // assigned udp not found
        std::cout << "Can't find udp: " << udpName << ", udpMap members are below: " << std::endl;
        for(std::map<string, UdpSender*>::iterator it = m_udpMap.begin();
             it != m_udpMap.end(); it++) {
            if(nullptr != it->second) {
                it->second->printUdpInfo();
            }
        }
        ret = 2;
    }

    return ret;
}

///
/// \brief UdpMapWrap::sendMsg, send out msg with udp sender has udpName,
///     to default IP and default Port set with setSenderDstAddr()
/// \param msgBuf, message to send
/// \param msgLen, message length
/// \param udpName, name of udp sender
/// \return
///     0, if send success;
///     1, if send out length != msgLen
///     2, if udp with udpName not found in m_udpMap
///
int UdpMapWrap::sendMsg(const void* msgBuf, const int msgLen, const string& udpName)
{
    std::lock_guard<std::mutex> guard(m_lockWrap);

    int ret(0);
    auto udpMapIte = m_udpMap.find(udpName);
    if(udpMapIte != m_udpMap.end ()) {
        // assigned udp initialized
        UdpSender* sender = udpMapIte->second;
        int sendLen = sender->sendTo(msgBuf, msgLen);

        if (sendLen != msgLen) {
            std::cout << "Send failed, send out msg len: " << sendLen
                      << " with udp: " << udpName << std::endl;
            sender->printUdpInfo();
            ret = 1;
        }
    } else {
        // assigned udp not found
        std::cout << "Can't find udp: " << udpName << ", udpMap member are below: " << std::endl;
        for(std::map<string, UdpSender*>::iterator it = m_udpMap.begin();
             it != m_udpMap.end(); it++) {
            if(nullptr != it->second) {
                it->second->printUdpInfo();
            }
        }
        ret = 2;
    }

    return ret;
}

///
/// \brief UdpMapWrap::insert2Map, insert assigned sender into m_udpMap
/// \param sockName, name of udp sender
/// \param sender, udp sender to add in m_udpMap
/// \return
///     0, if insert success;
///     1, if already exist socket with same sockName, and will automaticly
///         release sender resourse.
///
int UdpMapWrap::insert2Map(const string& sockName, UdpSender* sender)
{
    std::lock_guard<std::mutex> guard(m_lockWrap);

    if (m_udpMap.find(sockName) != m_udpMap.end()) {
        std::cout << "udp " << sockName << " already created." << std::endl;
        sender->close();
        return 1;
    } else {
        m_udpMap.insert(std::make_pair(sockName, sender));
        return 0;
    }
}

///
/// \brief UdpMapWrap::setSenderDstAddr, set destination address for udp sender
///     with sockName
/// \param sockName, name of udp sender to set dst addr
/// \param dstIP, default destination IP
/// \param dstPort, default destination port
/// \return
///     0, if success;
///     1, if sender with sockName not exist
///
int UdpMapWrap::setSenderDstAddr(const string& sockName, const string& dstIP,
                                 unsigned short dstPort)
{
    std::lock_guard<std::mutex> guard(m_lockWrap);

    if (m_udpMap.find(sockName) == m_udpMap.end()) {
        std::cout << "udp " << sockName << " not exist." << std::endl;
        return 1;
    } else {
        m_udpMap[sockName]->setDstAddress(dstIP, dstPort);
        return 0;
    }
}

///
/// \brief UdpMapWrap::destroy, release all udp sources in m_udpMap
///
void UdpMapWrap::destroy()
{
    std::lock_guard<std::mutex> guard(m_lockWrap);

    for(std::map<string, UdpSender*>::iterator it = m_udpMap.begin();
         it != m_udpMap.end(); it++) {
        if(nullptr != it->second) {
            delete it->second;
        } else {
            std::cout << "udp " << it->first << " already destroyed" << std::endl;
        }
    }

    m_udpMap.clear();
}

///
/// \brief UdpSenderMgr::addSender, create new udp sender and add in udpMap
/// \param name, name of udp sender
/// \param localIP, local IP to bind
/// \param localPort, local port to bind
/// \param bufLen, socket buffer length to set, if not assigned, use 1M as default
/// \return
///     0, if success;
///     1, if create udp success but udpMap has udp with same name,
///         newly created udp sources will released automatically
///     2, if create udp fail.
///
int UdpSenderMgr::addSender(const string& name, const string& localIP,
                            unsigned short localPort, const int bufLen)
{
    UdpSender* newUdp = new UdpSender(name, localIP, localPort, bufLen);
    if (newUdp->isValid()) {
        return udpMap.insert2Map(name, newUdp);
    } else {
        return 2; // udp initial fail
    }
}

///
/// \brief UdpSenderMgr::setSenderDstAddress, set destination address for udp sender
///     with sockName
/// \param name, name of udp sender to set dst addr
/// \param dstIP, default destination IP
/// \param dstPort, default destination port
/// \return
///     0, if success;
///     1, if sender with sockName not exist
///
int UdpSenderMgr::setSenderDstAddress(const string& name, const string& dstIP,
                                      unsigned short dstPort)
{
    return udpMap.setSenderDstAddr(name, dstIP, dstPort);
}

///
/// \brief UdpSenderMgr::udpSendMsg, send out msg with udp sender has udpName, to dstIP
///     and dstPort.
/// \param msgBuf, message to send
/// \param msgLen, message length
/// \param dstIP, destination IP
/// \param dstPort, destination port
/// \param udpName, name of udp sender
/// \return
///     0, if send success;
///     1, if send out length != msgLen
///     2, if udp with udpName not found in m_udpMap
///
int UdpSenderMgr::udpSendMsg(const void* msgBuf, const int msgLen, const string& dstIP,
                             unsigned short dstPort, const string& udpName)
{
    return udpMap.sendMsg(msgBuf, msgLen, dstIP, dstPort, udpName);
}

///
/// \brief UdpSenderMgr::udpSendMsg, send out msg with udp sender has udpName,
///     to default IP and default Port set with setSenderDstAddress()
/// \param msgBuf, message to send
/// \param msgLen, message length
/// \param udpName, name of udp sender
/// \return
///     0, if send success;
///     1, if send out length != msgLen
///     2, if udp with udpName not found in m_udpMap
///
int UdpSenderMgr::udpSendMsg(const void* msgBuf, const int msgLen, const string& udpName)
{
    return udpMap.sendMsg(msgBuf, msgLen, udpName);
}

///
/// \brief UdpSenderMgr::destoryUdp
///
void UdpSenderMgr::destoryUdp()
{
    udpMap.destroy();
}

bool isValidIP(const std::string& ip)
{
    const std::regex ipRegex("^(?:(?:\\d|[1-9]\\d|1\\d\\d|2[0-4]\\d|25[0-5])\\.){3}(?:\\d|[1-9]\\d|1\\d\\d|2[0-4]\\d|25[0-5])$");
    std::smatch ip_match;

    std::regex_match(ip, ip_match, ipRegex);

    return ip_match.size() == 1;
}
