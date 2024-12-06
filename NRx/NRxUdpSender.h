#ifndef NRXUDPSENDER_H
#define NRXUDPSENDER_H

#include <string>

#ifdef BUILDUDPSNDDLL
#ifdef WIN32
#define UDPSNDDLLAPI __declspec(dllexport)
#else
#define UDPSNDDLLAPI __attribute__((visibility("default")))
#endif
#else
#ifdef WIN32
#define UDPSNDDLLAPI __declspec(dllimport)
#else
#define UDPSNDDLLAPI __attribute__((visibility("default")))
#endif
#endif

namespace udpSender {

using std::string;

static const int gUdp1M = 1024 * 1024;

struct IPInfo {
    string IP;
    int port;
};

///
/// \brief addUdpSender, add udp sender, create new udp sender and add in udpMap
/// \param name, name of udp sender
/// \param localIP, local IP to bind
/// \param localPort, local port to bind
/// \param bufLen, socket buffer length to set, if not assigned, use 1M as default
/// \return
///     0, if success;
///     1, if create udp success but udpMap has udp with same name,
///         newly created udp sources will released automatically
///     2, if create udp fail,
///     3, localIP format wrong, IP invalid
///
extern UDPSNDDLLAPI int addUdpSender(const string& name, const string& localIP, unsigned short localPort,
                                                  const int bufLen = gUdp1M);

///
/// \brief setDstAddress, set destination address for udp sender with sockName
/// \param name, name of udp sender to set dst addr
/// \param dstIP, default destination IP
/// \param dstPort, default destination port
/// \return
///     0, if success;
///     1, if sender with sockName not exist
///     3, dstIP format wrong, IP invalid
///
extern UDPSNDDLLAPI int setDstAddress(const string& name, const string& dstIP, unsigned short dstPort);

///
/// \brief udpSendMsg, send message to assigned destination address, send out msg with udp sender has udpName, to dstIP
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
///     3, dstIP format wrong, IP invalid
///
extern UDPSNDDLLAPI int udpSendMsg(const void* msgBuf, const int msgLen, const string& dstIP,
                                                unsigned short dstPort, const string& udpName);

///
/// \brief udpSendMsg, send out msg with udp sender has udpName,
///     to default IP and default Port set with setSenderDstAddress()
/// \param msgBuf, message to send
/// \param msgLen, message length
/// \param udpName, name of udp sender
/// \return
///     0, if send success;
///     1, if send out length != msgLen
///     2, if udp with udpName not found in m_udpMap
///
extern UDPSNDDLLAPI int udpSendMsg(const void* msgBuf, const int msgLen, const string& udpName);

///
/// \brief udpDestory, release related udp source, call at last
///
extern UDPSNDDLLAPI void udpDestory(const string& name = "");

///
/// \brief getUdpLocalInfo, get local IP info of udp
/// \param name
///
extern UDPSNDDLLAPI IPInfo getUdpLocalInfo(const string& name);

///
/// \brief getUdpLocalInfo, get destination IP info of udp
/// \param name
///
extern UDPSNDDLLAPI IPInfo getUdpDstInfo(const string& name);

}

#endif // NRXUDPSENDER_H
