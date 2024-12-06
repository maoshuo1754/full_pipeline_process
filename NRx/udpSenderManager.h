#ifndef UDPSENDERMANAGER_H
#define UDPSENDERMANAGER_H

#include "NRxUdpSender.h"


#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <map>
#include <mutex>

using udpSender::gUdp1M;
using std::string;

struct Sender
{
    int socket;
    sockaddr_in bindAddr; // local IP address
    sockaddr_in remoteAddr; // dst IP address
};

///
/// \brief The UdpSender class, udp sender class, support p2p; multicast and broadcast
///
class UdpSender
{
public:
    UdpSender(const string& name, const string& localIP, unsigned short localPort,
              int bufLen = gUdp1M);
    ~UdpSender();
    void setDstAddress(const string& dstIP, unsigned short dstPort);
    int sendTo(const void* msgBuf, int msgLen);
    int sendTo(const void* msgBuf, int msgLen, const string& dstIP, unsigned short dstPort);
    void printUdpInfo(void);
    bool isValid(void);
    void close();

private:
    UdpSender() = delete;
    void setLocalAddress(const string& localIP = "127.0.0.1", unsigned short localPort = 0x3500);
    bool initInWin32(void);
    bool initUdp(int bufLen = 1024 * 1024, int setBroadCast = false);

private:
    string m_sockName;
    Sender m_sender;
};

class UdpMapWrap
{
public:
    UdpMapWrap(void);
    ~UdpMapWrap(void);
    int sendMsg(const void* msgBuf, const int msgLen, const string& dstIP,
                 unsigned short dstPort, const string& udpName);
    int sendMsg(const void* msgBuf, const int msgLen, const string& udpName);
    int insert2Map(const string& sockName, UdpSender* sender);
    int setSenderDstAddr(const string& sockName, const string& dstIP, unsigned short dstPort);
    void destroy(void);

private:
    std::mutex m_lockWrap; // lock m_udpMap
    std::map<string, UdpSender*> m_udpMap;
};

class UdpSenderMgr
{
public:
    static int addSender(const string& name, const string& localIP,
                         unsigned short localPort, const int bufLen = gUdp1M);
    static int setSenderDstAddress(const string& name, const string& dstIP,
                                   unsigned short dstPort);
    static int udpSendMsg(const void* msgBuf, const int msgLen, const string& dstIP,
                          unsigned short dstPort, const string& udpName);
    static int udpSendMsg(const void* msgBuf, const int msgLen, const string& udpName);
    static void destoryUdp(void);

private:
    static UdpMapWrap udpMap;
};

extern bool isValidIP(const string& ip);

#endif // UDPSENDERMANAGER_H
