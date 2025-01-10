#include "NRxInterface.h"
#include <cstring>
#ifdef WIN32
#include <winsock.h>
#else
#include <arpa/inet.h>
#include <sys/time.h>
#endif// WIN32

NRxIf::NRxIfHeader::NRxIfHeader()
{
    head      = NRxIfHead; // 0, 报文头，常数 0xF1A2B4C8.
    protocol  = NRxProtolVerion;  // 4, 协议版本号，当前版本号为0.
    counter   = 0;   // 6, 计数器，各类报文独自计数.

    time      = 0;      // 8, 时间.
    microSecs = 0; // 12, 表示秒以下的微秒数.

    msgBytes  = 0;  // 16, 报文总长度.
    tag       = 0;       // 18, 报文识别符.
    rdID      = 0;      // 20, 雷达ID.应用场景自定义.
    sendID    = 0;    // 22, 发方节点号.系统为能够发送或接收数据的软硬件实体分配节点号.
    rcvID     = 0;     // 23, 收方节点号.

    cpr       = 0;       // 24, 压缩标记.
    rdrChnID = 0; // 25, 数据采集时的数据通道号
    res1  = 0;  // 26, 预留，填0.
    res2  = 0;  // 28, 预留，填0.
    res3  = 0;  // 30, 预留，填0.
}

void NRxIf::NRxIfHeader::NetToHostEndian(const NRxIf::NRxIfHeader &src)
{
    head      = ntohl(src.head); // 0, 报文头，常数 0xF1A2B4C8.
    protocol  = ntohs(src.protocol);  // 4, 协议版本号，当前版本号为0.
    counter   = ntohs(src.counter);   // 6, 计数器，各类报文独自计数.

    time      = ntohl(src.time);      // 8, 时间.
    microSecs = ntohl(src.microSecs); // 12, 表示秒以下的微秒数.

    msgBytes  = ntohs(src.msgBytes);  // 16, 报文总长度.
    tag       = ntohs(src.tag);       // 18, 报文识别符.
    rdID      = ntohs(src.rdID);      // 20, 雷达ID.应用场景自定义.
    sendID    = src.sendID;    // 22, 发方节点号.系统为能够发送或接收数据的软硬件实体分配节点号.
    rcvID     = src.rcvID;     // 23, 收方节点号.

    cpr       = src.cpr;       // 24, 压缩标记.
    rdrChnID  = src.rdrChnID;  // 25, 数据采集时的数据通道号
    res1  = ntohs(src.res1);  // 26, 预留，填0.
    res2  = ntohs(src.res2);  // 28, 预留，填0.
    res3  = ntohs(src.res3);  // 30, 预留，填0.
}

void NRxIf::NRxIfHeader::HostToNetEndian(const NRxIf::NRxIfHeader &src)
{
    head      = htonl(src.head); // 0, 报文头，常数 0xF1A2B4C8.
    protocol  = htons(src.protocol);  // 4, 协议版本号，当前版本号为0.
    counter   = htons(src.counter);   // 6, 计数器，各类报文独自计数.

    time      = htonl(src.time);      // 8, 时间.
    microSecs = htonl(src.microSecs); // 12, 表示秒以下的微秒数.

    msgBytes  = htons(src.msgBytes);  // 16, 报文总长度.
    tag       = htons(src.tag);       // 18, 报文识别符.
    rdID      = htons(src.rdID);      // 20, 雷达ID.应用场景自定义.
    sendID    = src.sendID;    // 22, 发方节点号.系统为能够发送或接收数据的软硬件实体分配节点号.
    rcvID     = src.rcvID;     // 23, 收方节点号.

    cpr       = src.cpr;       // 24, 压缩标记.
    rdrChnID  = src.rdrChnID;  // 25, 数据采集时的数据通道号
    res1  = htons(src.res1);  // 26, 预留，填0.
    res2  = htons(src.res2);  // 28, 预留，填0.
    res3  = htons(src.res3);  // 30, 预留，填0.
}

NRxIf::NRxIfProjectMsgHeader::NRxIfProjectMsgHeader()
    : secondTag(0)
    , projectTag(0)
{
}

void NRxIf::NRxIfProjectMsgHeader::NetToHostEndian(const NRxIf::NRxIfProjectMsgHeader& src)
{
    secondTag = ntohs(src.secondTag); // 0, 报文二级标志
    projectTag = ntohs(src.projectTag); // 2, 项目编码
//    uint32 res1; // 4, 预留1
}

void NRxIf::NRxIfProjectMsgHeader::HostToNetEndian(const NRxIf::NRxIfProjectMsgHeader& src)
{
    secondTag = htons(src.secondTag); // 0, 报文二级标志
    projectTag = htons(src.projectTag); // 2, 项目编码
//    uint32 res1; // 4, 预留1
}

NRxIf::NRxIfEnd::NRxIfEnd()
    : CRC(0)
    , end1(NRxIfEnd1)
    , end2(NRxIfEnd2)
{
}

void NRxIf::NRxIfEnd::NetToHostEndian(const NRxIf::NRxIfEnd &src)
{
    CRC  = ntohl(src.CRC );
    end1 = ntohs(src.end1);
    end2 = ntohs(src.end2);
}

void NRxIf::NRxIfEnd::HostToNetEndian(const NRxIf::NRxIfEnd &src)
{
    CRC  = htonl(src.CRC );
    end1 = htons(src.end1);
    end2 = htons(src.end2);
}

NRxIf::NRxVidInfo::NRxVidInfo()
{
    memset(this, 0, sizeof(*this));
    vidSyncHead = NRxVidSyncHead;
    vidHeadLength = sizeof(NRxVidInfo);
    vidSyncTail = NRxVidSyncTail;
}

void NRxIf::NRxVidInfo::NetToHostEndian(const NRxIf::NRxVidInfo& src)
{
    vidSyncHead = ntohl(src.vidSyncHead); // 0, 同步头
    vidLength = ntohl(src.vidLength); // 4, 雷达视频长度

    vidHeadLength = ntohs(src.vidHeadLength); // 8, 雷达视频头长度
    vidFormat = ntohs(src.vidFormat); // 10, 编码格式
    pulseCombineMode = src.pulseCombineMode; // 12, 脉冲组合方式
    subPulseNum = src.subPulseNum; // 13, 子脉冲个数
    subPulseNo = src.subPulseNo; // 14, 子脉冲序号
    res0 = src.res0; // 15, 预留, 表示脉冲组合方式

    absTime0 = ntohl(src.absTime0); // 16, 绝对时间1
    absTime1 = ntohl(src.absTime1); // 20, 绝对时间2

    relTime0 = ntohl(src.relTime0); // 24, 相对时间
    relTime1 = ntohl(src.relTime1); // 28, 相对时间

    bandWidth = ntohl(src.bandWidth); // 32, 信号带宽
    sampleRate = ntohl(src.sampleRate); // 36, 采样率

    azi = ntohs(src.azi); // 40, 方位
    pulseWidth = ntohs(src.pulseWidth); // 42, 脉宽
    prt = ntohs(src.prt); // 44, PRT
    startCellNo = ntohs(src.startCellNo); // 46, 起始单元序号

    cellNum = ntohl(src.cellNum); // 48, 采样单元个数
    res1 = src.res1; // 52, 预留

    res2 = src.res2; // 56, 预留
    PIMFlag = src.PIMFlag; // 57, PIM 标识
    dataFlag = src.dataFlag; // 58, 数据标识
    mapPreLowerDB = src.mapPreLowerDB; // 59, 线性映射参数
    mapPreUpperDB = src.mapPreUpperDB; // 60, 线性映射参数
    res3 = src.res3; // 61, 预留
    dataSource = ntohs(src.dataSource); // 62, 数据源

    longitude = ntohl(src.longitude); // 64 经度
    latitude = ntohl(src.latitude); // 68 纬度

    high = ntohs(src.high); // 72, 高度
    absCourse = ntohs(src.absCourse); // 74, 绝对航向
    absVel = ntohs(src.absVel); // 76, 绝对航速
    relCourse = ntohs(src.relCourse); // 78, 相对航向

    relVel = ntohs(src.relVel); // 80, 相对航速 LSB: 0.1m/s，默认填0
    headSway = ntohs(src.headSway); // 82, 首摇 LSB: 360度/32768 默认值0
    rollSway = ntohs(src.rollSway); // 84, 横摇 LSB: 360度/32768 默认值0
    pitchSway = ntohs(src.pitchSway); // 86, 纵摇 LSB: 360度/32768 默认值0

    scanType = src.scanType; // 88, 扫描方式
    res4 = src.res4; // 89, 预留
    servoScanSpeed = ntohs(src.servoScanSpeed); // 90, 天线扫描速度 LSB:  0.1 deg/s, 0 = 无效
    servoStartAzi = ntohs(src.servoStartAzi); // 92, 天线扇扫前沿 LSB: 360.f / 65536.f, 0 = 无效
    servoEndAzi = ntohs(src.servoEndAzi); // 94, 天线扇扫后沿 LSB: 360.f / 65536.f, 0 = 无效

    channelSpeed = ntohl(src.channelSpeed); // 96, 通道速度 速度用补码表示, LSB = 0.1m/s 0xFFFF时无效
    channelNo = ntohs(src.channelNo); // 100, 通道序号 0xFF时无效
    res5 = src.res5; // 102, 预留

    memcpy(res6, src.res6, sizeof(char) * 16); // 104, 预留

    res7 = src.res7; // 120, 报文尾 预留
    vidSyncTail = ntohl(src.vidSyncTail); // 124, 报文尾
}

void NRxIf::NRxVidInfo::HostToNetEndian(const NRxIf::NRxVidInfo& src)
{
    vidSyncHead = htonl(src.vidSyncHead); // 0, 同步头
    vidLength = htonl(src.vidLength); // 4, 雷达视频长度

    vidHeadLength = htons(src.vidHeadLength); // 8, 雷达视频头长度
    vidFormat = htons(src.vidFormat); // 10, 编码格式
    pulseCombineMode = src.pulseCombineMode; // 12, 脉冲组合方式
    subPulseNum = src.subPulseNum; // 13, 子脉冲个数
    subPulseNo = src.subPulseNo; // 14, 子脉冲序号
    res0 = src.res0; // 15, 预留, 表示脉冲组合方式

    absTime0 = htonl(src.absTime0); // 16, 绝对时间1
    absTime1 = htonl(src.absTime1); // 20, 绝对时间2

    relTime0 = htonl(src.relTime0); // 24, 相对时间
    relTime1 = htonl(src.relTime1); // 28, 相对时间

    bandWidth = htonl(src.bandWidth); // 32, 信号带宽
    sampleRate = htonl(src.sampleRate); // 36, 采样率

    azi = htons(src.azi); // 40, 方位
    pulseWidth = htons(src.pulseWidth); // 42, 脉宽
    prt = htons(src.prt); // 44, PRT
    startCellNo = htons(src.startCellNo); // 46, 起始单元序号

    cellNum = htonl(src.cellNum); // 48, 采样单元个数
    res1 = src.res1; // 52, 预留

    res2 = src.res2; // 56, 预留
    PIMFlag = src.PIMFlag; // 57, PIM 标识
    dataFlag = src.dataFlag; // 58, 数据标识
    mapPreLowerDB = src.mapPreLowerDB; // 59, 线性映射参数
    mapPreUpperDB = src.mapPreUpperDB; // 60, 线性映射参数
    res3 = src.res3; // 61, 预留
    dataSource = htons(src.dataSource); // 62, 数据源

    longitude = htonl(src.longitude); // 64 经度
    latitude = htonl(src.latitude); // 68 纬度

    high = htons(src.high); // 72, 高度
    absCourse = htons(src.absCourse); // 74, 绝对航向
    absVel = htons(src.absVel); // 76, 绝对航速
    relCourse = htons(src.relCourse); // 78, 相对航向

    relVel = htons(src.relVel); // 80, 相对航速 LSB: 0.1m/s，默认填0
    headSway = htons(src.headSway); // 82, 首摇 LSB: 360度/32768 默认值0
    rollSway = htons(src.rollSway); // 84, 横摇 LSB: 360度/32768 默认值0
    pitchSway = htons(src.pitchSway); // 86, 纵摇 LSB: 360度/32768 默认值0

    scanType = src.scanType; // 88, 扫描方式
    res4 = src.res4; // 89, 预留
    servoScanSpeed = htons(src.servoScanSpeed); // 90, 天线扫描速度 LSB:  0.1 deg/s, 0 = 无效
    servoStartAzi = htons(src.servoStartAzi); // 92, 天线扇扫前沿 LSB: 360.f / 65536.f, 0 = 无效
    servoEndAzi = htons(src.servoEndAzi); // 94, 天线扇扫后沿 LSB: 360.f / 65536.f, 0 = 无效

    channelSpeed = htonl(src.channelSpeed); // 96, 通道速度 速度用补码表示, LSB = 0.1m/s 0xFFFF时无效
    channelNo = htons(src.channelNo); // 100, 通道序号 0xFF时无效
    res5 = src.res5; // 102, 预留

    memcpy(res6, src.res6, sizeof(char) * 16); // 104, 预留

    res7 = src.res7; // 120, 报文尾 预留
    vidSyncTail = htonl(src.vidSyncTail); // 124, 报文尾
}

NRxIf::NRxPlot::NRxPlot()
{
    memset(this, 0, sizeof(*this));
}

void NRxIf::NRxPlot::NetToHostEndian(const NRxIf::NRxPlot& src)
{
    absTime0 = ntohl(src.absTime0); // 0, 绝对时间1
    absTime1 = ntohl(src.absTime1); // 4, 绝对时间2

    relTime0 = ntohl(src.relTime0); // 8, 相对时间
    relTime1 = ntohl(src.relTime1); // 12, 相对时间

    dis = ntohl(src.dis); // 16, 距离
    azi = ntohs(src.azi); // 20, 方位
    ele = ntohs(src.ele); // 22, 仰角

    validAttr = ntohl(src.validAttr); // 24 有效标记
    amp = ntohs(src.amp); // 28 幅度
    ep = ntohs(src.ep); // 30 EP 数

    disStart = ntohl(src.disStart); // 32 距离起始
    disEnd = ntohl(src.disEnd); // 36 距离终止

    aziStart = ntohs(src.aziStart); // 40 方位起始
    aziEnd = ntohs(src.aziEnd); // 42 方位终止
    saturate = src.saturate; // 44 饱和度
    SNR = src.SNR; // 45 信杂噪比
    doppVel = ntohs(src.doppVel); // 46 多普勒速度

    maxDoppDiff = ntohs(src.maxDoppDiff); // 48 多普勒速度极差
    areaType = src.areaType; // 50 区域类型
    plotRetain = src.plotRetain; // 51 点迹判别结果
    plotType = src.plotType; // 52 点迹类型
    plotConfLv = src.plotConfLv; // 53 点迹置信度
    memcpy(res0, src.res0, sizeof(char) * 2); // 54 预留

    memcpy(plotTypeConfLv, src.plotTypeConfLv, sizeof(char) * 16); // 56 点迹类型置信度

    BaGAmp = ntohs(src.BaGAmp); // 72, 背景幅度
    ThrAmp = ntohs(src.ThrAmp); // 74, 门限幅度
    plot_id = ntohs(src.plot_id);
    // memcpy(res1, src.res1, sizeof(char) * 4); // 76, 预留
    memcpy(res2, src.res2, sizeof(char) * 16); // 80, 预留
}

void NRxIf::NRxPlot::HostToNetEndian(const NRxIf::NRxPlot& src)
{
    absTime0 = htonl(src.absTime0); // 0, 绝对时间1
    absTime1 = htonl(src.absTime1); // 4, 绝对时间2

    relTime0 = htonl(src.relTime0); // 8, 相对时间
    relTime1 = htonl(src.relTime1); // 12, 相对时间

    dis = htonl(src.dis); // 16, 距离
    azi = htons(src.azi); // 20, 方位
    ele = htons(src.ele); // 22, 仰角

    validAttr = htonl(src.validAttr); // 24 有效标记
    amp = htons(src.amp); // 28 幅度
    ep = htons(src.ep); // 30 EP 数

    disStart = htonl(src.disStart); // 32 距离起始
    disEnd = htonl(src.disEnd); // 36 距离终止

    aziStart = htons(src.aziStart); // 40 方位起始
    aziEnd = htons(src.aziEnd); // 42 方位终止
    saturate = src.saturate; // 44 饱和度
    SNR = src.SNR; // 45 信杂噪比
    doppVel = htons(src.doppVel); // 46 多普勒速度

    maxDoppDiff = htons(src.maxDoppDiff); // 48 多普勒速度极差
    areaType = src.areaType; // 50 区域类型
    plotRetain = src.plotRetain; // 51 点迹判别结果
    plotType = src.plotType; // 52 点迹类型
    plotConfLv = src.plotConfLv; // 53 点迹置信度
    memcpy(res0, src.res0, sizeof(char) * 2); // 54 预留

    memcpy(plotTypeConfLv, src.plotTypeConfLv, sizeof(char) * 16); // 56 点迹类型置信度

    BaGAmp = htons(src.BaGAmp); // 72, 背景幅度
    ThrAmp = htons(src.ThrAmp); // 74, 门限幅度
    plot_id = htons(src.plot_id);
    // memcpy(res1, src.res1, sizeof(char) * 4); // 76, 预留
    memcpy(res2, src.res2, sizeof(char) * 16); // 80, 预留
}

NRxIf::NRxSectorInfo::NRxSectorInfo()
{
    memset(this, 0, sizeof(*this));
}

void NRxIf::NRxSectorInfo::NetToHostEndian(const NRxIf::NRxSectorInfo& src)
{
    secNo = src.secNo; // 0 扇区号
    secNum = src.secNum; // 1 扇区总数
    plotsNum = ntohs(src.plotsNum); // 2 扇区点迹数 最大支持 NRxMaxPlotsNum(680) 个
    memcpy(res0, src.res0, sizeof(char) * 4); // 4 预留

    startAbsTime0 = ntohl(src.startAbsTime0); // 8 前沿绝对时间1
    startAbsTime1 = ntohl(src.startAbsTime1); // 12 前沿绝对时间2

    endAbsTime0 = ntohl(src.endAbsTime0); // 16 后沿绝对时间1
    endAbsTime1 = ntohl(src.endAbsTime1); // 20 后沿绝对时间2

    startRelTime0 = ntohl(src.startRelTime0); // 24 前沿相对时间
    startRelTime1 = ntohl(src.startRelTime1); // 28 前沿相对时间

    endRelTime0 = ntohl(src.endRelTime0); // 32 后沿相对时间
    endRelTime1 = ntohl(src.endRelTime1); // 36 后沿相对时间

    longitude = ntohl(src.longitude); // 40 经度
    latitude = ntohl(src.latitude); // 44 纬度

    height = ntohs(src.height); // 48 高度
    absCourse = ntohs(src.absCourse); // 50 绝对航向
    absVel  = ntohs(src.absVel); // 52 绝对航速
    relCourse = ntohs(src.relCourse); // 54 相对航向

    relVel = ntohs(src.relVel); // 56 相对航速
    headSway = ntohs(src.headSway); // 58 首摇
    rollSway = ntohs(src.rollSway); // 60 横摇
    pitchSway = ntohs(src.pitchSway); // 62 纵摇

    scanType = src.scanType; // 64 扫描方式
    res1 = src.res1; // 65 预留
    servoScanSpeed = ntohs(src.servoScanSpeed); // 66 天线扫描速度
    servoStartAzi = ntohs(src.servoStartAzi); // 68 天线扇扫前沿
    servoEndAzi = ntohs(src.servoEndAzi); // 70 天线扇扫后沿

    memcpy(res2, src.res2, sizeof(char) * 16); // 72 预留
}

void NRxIf::NRxSectorInfo::HostToNetEndian(const NRxIf::NRxSectorInfo& src)
{
    secNo = src.secNo; // 0 扇区号
    secNum = src.secNum; // 1 扇区总数
    plotsNum = htons(src.plotsNum); // 2 扇区点迹数 最大支持 NRxMaxPlotsNum(680) 个
    memcpy(res0, src.res0, sizeof(char) * 4); // 4 预留

    startAbsTime0 = htonl(src.startAbsTime0); // 8 前沿绝对时间1
    startAbsTime1 = htonl(src.startAbsTime1); // 12 前沿绝对时间2

    endAbsTime0 = htonl(src.endAbsTime0); // 16 后沿绝对时间1
    endAbsTime1 = htonl(src.endAbsTime1); // 20 后沿绝对时间2

    startRelTime0 = htonl(src.startRelTime0); // 24 前沿相对时间
    startRelTime1 = htonl(src.startRelTime1); // 28 前沿相对时间

    endRelTime0 = htonl(src.endRelTime0); // 32 后沿相对时间
    endRelTime1 = htonl(src.endRelTime1); // 36 后沿相对时间

    longitude = htonl(src.longitude); // 40 经度
    latitude = htonl(src.latitude); // 44 纬度

    height = htons(src.height); // 48 高度
    absCourse = htons(src.absCourse); // 50 绝对航向
    absVel  = htons(src.absVel); // 52 绝对航速
    relCourse = htons(src.relCourse); // 54 相对航向

    relVel = htons(src.relVel); // 56 相对航速
    headSway = htons(src.headSway); // 58 首摇
    rollSway = htons(src.rollSway); // 60 横摇
    pitchSway = htons(src.pitchSway); // 62 纵摇

    scanType = src.scanType; // 64 扫描方式
    res1 = src.res1; // 65 预留
    servoScanSpeed = htons(src.servoScanSpeed); // 66 天线扫描速度
    servoStartAzi = htons(src.servoStartAzi); // 68 天线扇扫前沿
    servoEndAzi = htons(src.servoEndAzi); // 70 天线扇扫后沿

    memcpy(res2, src.res2, sizeof(char) * 16); // 72 预留
}

NRxIf::NRxTrk_BasicInfo::NRxTrk_BasicInfo()
{
    memset(this, 0, sizeof(*this));
    msgTag = NRxIfTag_Trk_BasicInfo;
    bytes = sizeof(NRxIf::NRxTrk_BasicInfo);
}

void NRxIf::NRxTrk_BasicInfo::NetToHostEndian(const NRxIf::NRxTrk_BasicInfo& src)
{
    msgTag = ntohs(src.msgTag); // 0, 航迹报文二级标识
    bytes = ntohs(src.bytes);// 2, 基础信息从0地址开始的总字节数.
    memcpy(res0, src.res0, sizeof(char) * 4); // 2, 预留

    batchNo = ntohs(src.batchNo); // 8, 批号
    seqNo = ntohs(src.seqNo); // 10, 序号
    trkStat = ntohs(src.trkStat); // 12, 状态
    ctrlWord0 = ntohs(src.ctrlWord0); // 14, 控制字

    absTime0 = ntohl(src.absTime0); // 16, 绝对时间1
    absTime1 = ntohl(src.absTime1); // 20, 绝对时间2

    relTime0 = ntohl(src.relTime0); // 24, 相对时间
    relTime1 = ntohl(src.relTime1); // 28, 相对时间

    dis = ntohl(src.dis); // 32, 距离
    azi = ntohs(src.azi); // 36, 方位
    ele = ntohs(src.ele); // 38, 仰角

    absCourse = ntohs(src.absCourse); // 40, 绝对航向
    absVel = ntohs(src.absVel); // 42, 绝对航速
    relCourse = ntohs(src.relCourse); // 44, 相对航向
    relVel = ntohs(src.relVel); // 46, 相对航速

    longitude = ntohl(src.longitude); // 48, 经度
    latitude = ntohl(src.latitude); // 52, 纬度

    high = ntohs(src.high); // 56, 高度
    contiLostPlots = src.contiLostPlots; // 58, 连续丢点数
    trkQuality = src.trkQuality; // 59, 航迹质量
    updateTime = ntohs(src.updateTime); // 60, 更新次数
    ampFiltVal = ntohs(src.ampFiltVal); // 62, 幅度滤波值

    epFiltVal = ntohs(src.epFiltVal); // 64, EP 数滤波值
    disSpanFiltVal = ntohs(src.disSpanFiltVal); // 66, 距离展宽滤波值
    aziSpanFiltVal = ntohs(src.aziSpanFiltVal); // 68, 方位展宽滤波值
    eleSpanFiltVal = ntohs(src.eleSpanFiltVal); // 70, 仰角展宽滤波值

    doppVelFiltVal = ntohl(src.doppVelFiltVal); // 72, 径向速度滤波值
    duration = ntohl(src.duration); // 76, 航迹持续时间

    disCoast = ntohl(src.disCoast); // 80, 距离外推值 LSB: 1m
    aziCoast = ntohs(src.aziCoast); // 84, 方位外推值 LSB: 360.f / 65536.f
    eleCoast = ntohs(src.eleCoast); // 86, 仰角外推值 LSB: 180.f / 32768.f

    mmsi = ntohl(src.mmsi); // 88, MMSI 号, 0xFFFFFFFF 表示无效
    memcpy(res1, src.res1, sizeof(char) * 4); // 92, 预留

    memcpy(res2, src.res2, sizeof(char) * 8); // 96, 预留
}

void NRxIf::NRxTrk_BasicInfo::HostToNetEndian(const NRxIf::NRxTrk_BasicInfo& src)
{
    msgTag = htons(src.msgTag); // 0, 航迹报文二级标识
    bytes = htons(src.bytes); // 2, 基础信息从0地址开始的总字节数.
    memcpy(res0, src.res0, sizeof(char) * 4); // 4, 预留

    batchNo = htons(src.batchNo); // 8, 批号
    seqNo = htons(src.seqNo); // 10, 序号
    trkStat = htons(src.trkStat); // 12, 状态
    ctrlWord0 = htons(src.ctrlWord0); // 14, 控制字

    absTime0 = htonl(src.absTime0); // 16, 绝对时间1
    absTime1 = htonl(src.absTime1); // 20, 绝对时间2

    relTime0 = htonl(src.relTime0); // 24, 相对时间
    relTime1 = htonl(src.relTime1); // 28, 相对时间

    dis = htonl(src.dis); // 32, 距离
    azi = htons(src.azi); // 36, 方位
    ele = htons(src.ele); // 38, 仰角

    absCourse = htons(src.absCourse); // 40, 绝对航向
    absVel = htons(src.absVel); // 42, 绝对航速
    relCourse = htons(src.relCourse); // 44, 相对航向
    relVel = htons(src.relVel); // 46, 相对航速

    longitude = htonl(src.longitude); // 48, 经度
    latitude = htonl(src.latitude); // 52, 纬度

    high = htons(src.high); // 56, 高度
    contiLostPlots = src.contiLostPlots; // 58, 连续丢点数
    trkQuality = src.trkQuality; // 59, 航迹质量
    updateTime = htons(src.updateTime); // 60, 更新次数
    ampFiltVal = htons(src.ampFiltVal); // 62, 幅度滤波值

    epFiltVal = htons(src.epFiltVal); // 64, EP 数滤波值
    disSpanFiltVal = htons(src.disSpanFiltVal); // 66, 距离展宽滤波值
    aziSpanFiltVal = htons(src.aziSpanFiltVal); // 68, 方位展宽滤波值
    eleSpanFiltVal = htons(src.eleSpanFiltVal); // 70, 仰角展宽滤波值

    doppVelFiltVal = htonl(src.doppVelFiltVal); // 72, 径向速度滤波值
    duration = htonl(src.duration); // 76, 航迹持续时间

    disCoast = htonl(src.disCoast); // 80, 距离外推值 LSB: 1m
    aziCoast = htons(src.aziCoast); // 84, 方位外推值 LSB: 360.f / 65536.f
    eleCoast = htons(src.eleCoast); // 86, 仰角外推值 LSB: 180.f / 32768.f

    mmsi = htonl(src.mmsi); // 88, MMSI 号, 0xFFFFFFFF 表示无效
    memcpy(res1, src.res1, sizeof(char) * 4); // 92, 预留

    memcpy(res2, src.res2, sizeof(char) * 8); // 96, 预留
}

NRxIf::NRxTrk_CommonInfo::NRxTrk_CommonInfo()
{
    memset(this, 0, sizeof(*this));
    msgTag = NRxIfTag_Trk_CommonInfo;
    bytes = sizeof(NRxIf::NRxTrk_CommonInfo);
}

void NRxIf::NRxTrk_CommonInfo::NetToHostEndian(const NRxIf::NRxTrk_CommonInfo& src)
{
    msgTag = ntohs(src.msgTag); // 0, 航迹报文二级标识
    bytes = ntohs(src.bytes);  // 2, 基础信息从0地址开始的总字节数.
    memcpy(res0, src.res0, sizeof(char) * 4); // 2, 预留

    ctrlWord0 = ntohl(src.ctrlWord0); // 8, 控制字
    assoPlotDis = ntohl(src.assoPlotDis); // 12, 关联点迹距离

    assoPlotAzi = ntohs(src.assoPlotAzi); // 16, 关联点迹方位
    assoPlotEle = ntohs(src.assoPlotEle); // 18, 关联点迹仰角
    assoPlotAmp = ntohs(src.assoPlotAmp); // 20, 关联点迹幅度
    assoPlotEP = ntohs(src.assoPlotEP); // 22, 关联点迹 EP 数

    assoPlotDisSpan = ntohs(src.assoPlotDisSpan); // 24, 关联点迹距离展宽
    assoPlotAziSpan = ntohs(src.assoPlotAziSpan); // 26, 关联点迹方位展宽
    memcpy(res1, src.res1, sizeof(char) * 28); // 28, 关联点迹预留 28(4 + 24) 字节

    memcpy(res2, src.res2, sizeof(char) * 5); // 56, 预留
    gatePlotsCount = src.gatePlotsCount; // 61, 波门内点迹数
    gateType = ntohs(src.gateType); // 62, 波门类型

    gateCenterDis = ntohl(src.gateCenterDis); // 64, 波门中心距离
    gateCenterAzi = ntohs(src.gateCenterAzi); // 68, 波门中心方位
    gateCenterEle = ntohs(src.gateCenterEle); // 70, 波门中心仰角

    gateParam0 = ntohl(src.gateParam0); // 72, 波门参数
    gateParam1 = ntohl(src.gateParam1); // 76, 波门参数

    gateParam2 = ntohl(src.gateParam2); // 80, 波门参数
    gateParam3 = ntohl(src.gateParam3); // 84, 波门参数

    gateParam4 = ntohl(src.gateParam4); // 88, 波门参数
    gateParam5 = ntohl(src.gateParam5); // 92, 波门参数

    disStd = ntohs(src.disStd); // 96, 距离标准差
    aziStd = ntohs(src.aziStd); // 98, 方位标准差
    eleStd = ntohs(src.eleStd); // 100, 仰角标准差
    ampStd = ntohs(src.ampStd); // 102, 幅度标准差

    epStd = ntohs(src.epStd); // 104, EP数标准差
    disSpanStd = ntohs(src.disSpanStd); // 106, 距离展宽标准差
    aziSpanStd = ntohs(src.aziSpanStd); // 108, 方位展宽标准差
    eleSpanStd = ntohs(src.eleSpanStd); // 110, 仰角展宽标准差

    doppVelStd = ntohs(src.doppVelStd); // 112, 径向速度标准差
    linearAcceleration = ntohs(src.linearAcceleration); // 114, 直线加速度
    angularVel = src.angularVel; // 116, 角速度 无效时填0
    motionState = src.motionState; // 117, 运动状态 0 不明; 1 静止; 2 匀速直线运动; 3 机动
    res3 = src.res3; // 118, 预留
    res4 = src.res4; // 119, 预留

    elecWarn1 = ntohs(elecWarn1); // 120, 电子围栏告警信息.
    elecWarn2 = ntohs(elecWarn2); // 122, 电子围栏告警信息.
    memcpy(res5, src.res5, sizeof(char) * 12); // 124, 预留
}

void NRxIf::NRxTrk_CommonInfo::HostToNetEndian(const NRxIf::NRxTrk_CommonInfo& src)
{
    msgTag = htons(src.msgTag); // 0, 航迹报文二级标识
    bytes = htons(src.bytes);  // 2, 基础信息从0地址开始的总字节数.
    memcpy(res0, src.res0, sizeof(char) * 4); // 4, 预留

    ctrlWord0 = htonl(src.ctrlWord0); // 8, 控制字
    assoPlotDis = htonl(src.assoPlotDis); // 12, 关联点迹距离

    assoPlotAzi = htons(src.assoPlotAzi); // 16, 关联点迹方位
    assoPlotEle = htons(src.assoPlotEle); // 18, 关联点迹仰角
    assoPlotAmp = htons(src.assoPlotAmp); // 20, 关联点迹幅度
    assoPlotEP = htons(src.assoPlotEP); // 22, 关联点迹 EP 数

    assoPlotDisSpan = htons(src.assoPlotDisSpan); // 24, 关联点迹距离展宽
    assoPlotAziSpan = htons(src.assoPlotAziSpan); // 26, 关联点迹方位展宽
    memcpy(res1, src.res1, sizeof(char) * 28); // 28, 关联点迹预留 28(4 + 24) 字节

    memcpy(res2, src.res2, sizeof(char) * 5); // 56, 预留
    gatePlotsCount = src.gatePlotsCount; // 61, 波门内点迹数
    gateType = htons(src.gateType); // 62, 波门类型

    gateCenterDis = htonl(src.gateCenterDis); // 64, 波门中心距离
    gateCenterAzi = htons(src.gateCenterAzi); // 68, 波门中心方位
    gateCenterEle = htons(src.gateCenterEle); // 70, 波门中心仰角

    gateParam0 = htonl(src.gateParam0); // 72, 波门参数
    gateParam1 = htonl(src.gateParam1); // 76, 波门参数

    gateParam2 = htonl(src.gateParam2); // 80, 波门参数
    gateParam3 = htonl(src.gateParam3); // 84, 波门参数

    gateParam4 = htonl(src.gateParam4); // 88, 波门参数
    gateParam5 = htonl(src.gateParam5); // 92, 波门参数

    disStd = htons(src.disStd); // 96, 距离标准差
    aziStd = htons(src.aziStd); // 98, 方位标准差
    eleStd = htons(src.eleStd); // 100, 仰角标准差
    ampStd = htons(src.ampStd); // 102, 幅度标准差

    epStd = htons(src.epStd); // 104, EP数标准差
    disSpanStd = htons(src.disSpanStd); // 106, 距离展宽标准差
    aziSpanStd = htons(src.aziSpanStd); // 108, 方位展宽标准差
    eleSpanStd = htons(src.eleSpanStd); // 110, 仰角展宽标准差

    doppVelStd = htons(src.doppVelStd); // 112, 径向速度标准差
    linearAcceleration = htons(src.linearAcceleration); // 114, 直线加速度
    angularVel = src.angularVel; // 116, 角速度 无效时填0
    motionState = src.motionState; // 117, 运动状态 0 不明; 1 静止; 2 匀速直线运动; 3 机动
    res3 = src.res3; // 118, 预留
    res4 = src.res4; // 119, 预留

    elecWarn1 = htons(elecWarn1); // 120, 电子围栏告警信息.
    elecWarn2 = htons(elecWarn2); // 122, 电子围栏告警信息.
    memcpy(res5, src.res5, sizeof(char) * 12); // 124, 预留
}

NRxIf::NRxTrk_MarkInfo::NRxTrk_MarkInfo()
{
    memset(this, 0, sizeof(NRxTrk_MarkInfo));
    msgTag = 0x0C;
}

void NRxIf::NRxTrk_MarkInfo::NetToHostEndian(const NRxIf::NRxTrk_MarkInfo& src)
{
    msgTag = ntohs(src.msgTag); // 0 航迹报文二级标识
    bytes = ntohs(src.bytes); // 2, 标识信息从0地址开始的总字节数

    tag0 = ntohs(src.tag0); // 8, 标志 0
    tag1 = ntohs(src.tag1); // 10, 标志 1
    tag2 = ntohs(src.tag2); // 12, 标志 2
    tag3 = ntohs(src.tag3); // 14, 标志 3
    tag4 = ntohs(src.tag4); // 16, 标志 4
    tag5 = ntohs(src.tag5); // 18, 标志 5
    tag6 = ntohs(src.tag6); // 20, 标志 6
    tag7 = ntohs(src.tag7); // 22, 标志 7
    tag8 = ntohs(src.tag8); // 24, 标志 8
    tag9 = ntohs(src.tag9); // 26, 标志 9
    tag10 = ntohs(src.tag10); // 28, 标志 10
    tag11 = ntohs(src.tag11); // 30, 标志 11
    tag12 = ntohs(src.tag12); // 32, 标志 12
    tag13 = ntohs(src.tag13); // 34, 标志 13
    tag14 = ntohs(src.tag14); // 36, 标志 14
    tag15 = ntohs(src.tag15); // 38, 标志 15
}

void NRxIf::NRxTrk_MarkInfo::HostToNetEndian(const NRxIf::NRxTrk_MarkInfo& src)
{
    msgTag = htons(src.msgTag); // 0 航迹报文二级标识
    bytes = htons(src.bytes); // 2, 标识信息从0地址开始的总字节数

    tag0 = htons(src.tag0); // 8, 标志 0
    tag1 = htons(src.tag1); // 10, 标志 1
    tag2 = htons(src.tag2); // 12, 标志 2
    tag3 = htons(src.tag3); // 14, 标志 3
    tag4 = htons(src.tag4); // 16, 标志 4
    tag5 = htons(src.tag5); // 18, 标志 5
    tag6 = htons(src.tag6); // 20, 标志 6
    tag7 = htons(src.tag7); // 22, 标志 7
    tag8 = htons(src.tag8); // 24, 标志 8
    tag9 = htons(src.tag9); // 26, 标志 9
    tag10 = htons(src.tag10); // 28, 标志 10
    tag11 = htons(src.tag11); // 30, 标志 11
    tag12 = htons(src.tag12); // 32, 标志 12
    tag13 = htons(src.tag13); // 34, 标志 13
    tag14 = htons(src.tag14); // 36, 标志 14
    tag15 = htons(src.tag15); // 38, 标志 15
}

NRxIf::NRxAISBody::NRxAISBody()
{
    memset(this, 0, sizeof(*this));
    std::string callLeters = "@@@@@@@";
    memcpy(this->callLeters, callLeters.c_str(), callLeters.size());
    std::string boatName = "@@@@@@@@@@@@@@@@@@@@@";
    memcpy(this->boatName, boatName.c_str(), boatName.size());
}

void NRxIf::NRxAISBody::NetToHostEndian(const NRxIf::NRxAISBody& src)
{
    status = ntohs(src.status); // 0, AIS 状态字
    aisType = ntohs(src.aisType); // 2, 导航定位类型
    mmsi = ntohl(src.mmsi); // 4, 用户识别码

    longitude = ntohl(src.longitude); // 8, 经度
    latitude = ntohl(src.latitude); // 12, 纬度

    res0 = ntohl(src.res0);// 16, 预留
    groundSpeed = ntohs(src.groundSpeed); // 20, 真航速
    groundCourse = ntohs(src.groundCourse); // 22, 对地航向

    trueCourse = ntohs(src.trueCourse); // 24, 真航向
    power = ntohs(src.power); // 26, 功率
    memcpy(callLeters, src.callLeters, sizeof(char) * 7); // 28, 呼号
    memcpy(boatName, src.boatName, sizeof(char) * 21); // 35, 船名

    expectRcvTime = ntohl(src.expectRcvTime); // 56, 预定到达时间
    length = ntohs(src.length); // 60, 船长，0.1m，0=无效
    width = ntohs(src.width); // 62, 船宽，0.1m，0=无效

    naviStatus = src.naviStatus; // 64, 航行状态
    res1 = src.res1; // 65, 预留
    shipType = ntohs(src.shipType); // 66, 船舶及载货类型
    maxDrawWaterLine = ntohs(src.maxDrawWaterLine); // 68, 最大吃水深度
    turnRate = ntohs(src.turnRate); // 70, 转弯率

    posPrecision = ntohs(src.posPrecision); // 72, 位置精度
    memcpy(origin, src.origin, sizeof(char) * 21); // 74, 始发地, 没有填 @, 参照 ASCII 码表
    res2 = src.res2; // 95, 预留

    revTime0 = ntohl(src.revTime0); // 96, 消息接收绝对时间1
    revTime1 = ntohl(src.revTime1); // 100, 消息接收绝对时间2

    memcpy(destination, src.destination, sizeof(char) * 21); // 104, 目的地, 没有填 @, 参照 ASCII 码表
    memcpy(res3, src.res3, sizeof(char) * 11); // 125, 备用
}

void NRxIf::NRxAISBody::HostToNetEndian(const NRxIf::NRxAISBody &src)
{
    status = htons(src.status); // 0, AIS 状态字
    aisType = htons(src.aisType); // 2, 导航定位类型
    mmsi = htonl(src.mmsi); // 4, 用户识别码

    longitude = htonl(src.longitude); // 8, 经度
    latitude = htonl(src.latitude); // 12, 纬度

    res0 = htonl(src.res0); // 16, 预留
    groundSpeed = htons(src.groundSpeed); // 20, 真航速
    groundCourse = htons(src.groundCourse); // 22, 对地航向

    trueCourse = htons(src.trueCourse); // 24, 真航向
    power = htons(src.power); // 26, 功率
    memcpy(callLeters, src.callLeters, sizeof(char) * 7); // 28, 呼号
    memcpy(boatName, src.boatName, sizeof(char) * 21); // 35, 船名

    expectRcvTime = htonl(src.expectRcvTime); // 56, 预定到达时间
    length = htons(src.length); // 60, 船长，0.1m，0=无效
    width = htons(src.width); // 62, 船宽，0.1m，0=无效

    naviStatus = src.naviStatus; // 64, 航行状态
    res1 = src.res1; // 65, 预留
    shipType = htons(src.shipType); // 66, 船舶及载货类型
    maxDrawWaterLine = htons(src.maxDrawWaterLine); // 68, 最大吃水深度
    turnRate = htons(src.turnRate); // 70, 转弯率

    posPrecision = htons(src.posPrecision); // 72, 位置精度
    memcpy(origin, src.origin, sizeof(char) * 21); // 74, 始发地, 没有填 @, 参照 ASCII 码表
    res2 = src.res2; // 95, 预留

    revTime0 = htonl(src.revTime0); // 96, 消息接收绝对时间1
    revTime1 = htonl(src.revTime1); // 100, 消息接收绝对时间2

    memcpy(destination, src.destination, sizeof(char) * 21); // 104, 目的地, 没有填 @, 参照 ASCII 码表
    memcpy(res3, src.res3, sizeof(char) * 11); // 125, 备用
}

NRxIf::NRxAIS::NRxAIS()
{
    head.tag = NRxIfTag_AISDEC;
    head.msgBytes = sizeof(NRxIf::NRxAIS);
}

void NRxIf::NRxAIS::NetToHostEndian(const NRxIf::NRxAIS& src)
{
    head.NetToHostEndian(src.head);
    body.NetToHostEndian(src.body);
    end.NetToHostEndian(src.end);
}

void NRxIf::NRxAIS::HostToNetEndian(const NRxIf::NRxAIS& src)
{
    head.HostToNetEndian(src.head);
    body.HostToNetEndian(src.body);
    end.HostToNetEndian(src.end);
}

NRxIf::NRxObjMgrBody::NRxObjMgrBody()
{
    memset(this, 0, sizeof(*this));
}

void NRxIf::NRxObjMgrBody::NetToHostEndian(const NRxIf::NRxObjMgrBody& src)
{
    ctrlWord0 = ntohs(src.ctrlWord0); // 0, 键盘命令字 0
    ctrlWord1 = ntohs(src.ctrlWord1); // 2, 键盘命令字 1
    ctrlWord2 = ntohs(src.ctrlWord2); // 4, 键盘命令字 2
    ctrlWord3 = ntohs(src.ctrlWord3); // 6, 键盘命令字 3

    ctrlWord4 = ntohs(src.ctrlWord4); // 8, 键盘命令字 4
    ctrlWord5 = ntohs(src.ctrlWord5); // 10, 键盘命令字 5
    ctrlWord6 = ntohs(src.ctrlWord6); // 12, 键盘命令字 6
    ctrlWord7 = ntohs(src.ctrlWord7); // 14, 键盘命令字 7

    ctrlWord8 = ntohs(src.ctrlWord8); // 16, 键盘命令字 8
    ctrlWord9 = ntohs(src.ctrlWord9); // 18, 键盘命令字 9
    ctrlWord10 = ntohs(src.ctrlWord10); // 20, 键盘命令字 10
    ctrlWord11 = ntohs(src.ctrlWord11); // 22, 键盘命令字 11

    ctrlWord12 = ntohs(src.ctrlWord12); // 24, 键盘命令字 12
    ctrlWord13 = ntohs(src.ctrlWord13); // 26, 键盘命令字 13
    ctrlWord14 = ntohs(src.ctrlWord14); // 28, 键盘命令字 14
    ctrlWord15 = ntohs(src.ctrlWord15); // 30, 键盘命令字 15
}

void NRxIf::NRxObjMgrBody::HostToNetEndian(const NRxIf::NRxObjMgrBody& src)
{
    ctrlWord0 = htons(src.ctrlWord0); // 0, 键盘命令字 0
    ctrlWord1 = htons(src.ctrlWord1); // 2, 键盘命令字 1
    ctrlWord2 = htons(src.ctrlWord2); // 4, 键盘命令字 2
    ctrlWord3 = htons(src.ctrlWord3); // 6, 键盘命令字 3

    ctrlWord4 = htons(src.ctrlWord4); // 8, 键盘命令字 4
    ctrlWord5 = htons(src.ctrlWord5); // 10, 键盘命令字 5
    ctrlWord6 = htons(src.ctrlWord6); // 12, 键盘命令字 6
    ctrlWord7 = htons(src.ctrlWord7); // 14, 键盘命令字 7

    ctrlWord8 = htons(src.ctrlWord8); // 16, 键盘命令字 8
    ctrlWord9 = htons(src.ctrlWord9); // 18, 键盘命令字 9
    ctrlWord10 = htons(src.ctrlWord10); // 20, 键盘命令字 10
    ctrlWord11 = htons(src.ctrlWord11); // 22, 键盘命令字 11

    ctrlWord12 = htons(src.ctrlWord12); // 24, 键盘命令字 12
    ctrlWord13 = htons(src.ctrlWord13); // 26, 键盘命令字 13
    ctrlWord14 = htons(src.ctrlWord14); // 28, 键盘命令字 14
    ctrlWord15 = htons(src.ctrlWord15); // 30, 键盘命令字 15
}

NRxIf::NRxObjMgr::NRxObjMgr()
{
    head.tag = NRxIfTag_TrkMgr;
    head.msgBytes = sizeof(NRxIf::NRxObjMgr);
}

void NRxIf::NRxObjMgr::NetToHostEndian(const NRxIf::NRxObjMgr& src)
{
    head.NetToHostEndian(src.head);
    body.NetToHostEndian(src.body);
    end.NetToHostEndian(src.end);
}

void NRxIf::NRxObjMgr::HostToNetEndian(const NRxIf::NRxObjMgr& src)
{
    head.HostToNetEndian(src.head);
    body.HostToNetEndian(src.body);
    end.HostToNetEndian(src.end);
}

NRxIf::NRxAreaPlot::NRxAreaPlot()
{
    memset(this, 0, sizeof(*this));
}

void NRxIf::NRxAreaPlot::NetToHostEndian(const NRxIf::NRxAreaPlot &src)
{
    dis = ntohl(src.dis); // 0, 距离
    azi = ntohs(src.azi); // 4, 方位
    res = src.res; // 6, 预留
}

void NRxIf::NRxAreaPlot::HostToNetEndian(const NRxIf::NRxAreaPlot &src)
{
    dis = htonl(src.dis); // 0, 距离
    azi = htons(src.azi); // 4, 方位
    res = src.res; // 6, 预留
}

NRxIf::NRxAreaMgr::NRxAreaMgr()
{
    memset(this, 0, sizeof(*this));
}

void NRxIf::NRxAreaMgr::NetToHostEndian(const NRxIf::NRxAreaMgr& src)
{
    ctrlWord = src.ctrlWord; // 0, 控制字
    areaType = src.areaType; // 1, 区域类型
    plotNum = src.plotNum; // 2, 顶点数
    attrs = src.attrs;// 3,区域附件属性.
    role = src.role;// 4, 区域用途
    memcpy(res0, src.res0, sizeof(char) * 3); // 5, 预留

    memcpy(name, src.name, sizeof(char) * 64); // 8, 区域名
}

void NRxIf::NRxAreaMgr::HostToNetEndian(const NRxIf::NRxAreaMgr &src)
{
    ctrlWord = src.ctrlWord; // 0, 控制字
    areaType = src.areaType; // 1, 区域类型
    plotNum = src.plotNum; // 2, 顶点数
    attrs = src.attrs;// 3,区域附件属性.
    role = src.role;// 4, 区域用途
    memcpy(res0, src.res0, sizeof(char) * 3); // 5, 预留

    memcpy(name, src.name, sizeof(char) * 64); // 8, 区域名
}

NRxIf::NRxParamMgrHead::NRxParamMgrHead()
{
    memset(this, 0, sizeof(*this));
}

void NRxIf::NRxParamMgrHead::NetToHostEndian(const NRxIf::NRxParamMgrHead& src)
{
    ctrlWord0 = src.ctrlWord0; // 0, 用途
    sendID = src.sendID; // 1, 软件 ID
    memcpy(res, src.res, sizeof(char) * 6); // 2, 预留
}

void NRxIf::NRxParamMgrHead::HostToNetEndian(const NRxIf::NRxParamMgrHead& src)
{
    ctrlWord0 = src.ctrlWord0; // 0, 用途
    sendID = src.sendID; // 1, 软件 ID
    memcpy(res, src.res, sizeof(char) * 6); // 2, 预留
}

NRxIf::NRxParamMgrResetComParBody::NRxParamMgrResetComParBody()
{
    memset(this, 0, sizeof(*this));
}

void NRxIf::NRxParamMgrResetComParBody::NetToHostEndian(const NRxIf::NRxParamMgrResetComParBody& src)
{
    memcpy(tagName, src.tagName, sizeof(char) * 32); // 0, 标签名
}

void NRxIf::NRxParamMgrResetComParBody::HostToNetEndian(const NRxIf::NRxParamMgrResetComParBody& src)
{
    memcpy(tagName, src.tagName, sizeof(char) * 32); // 0, 标签名
}

NRxIf::NRxParamMgrResetComPar::NRxParamMgrResetComPar()
{
    head.tag = NRxIfTag_ParmMgr;
    head.msgBytes = sizeof(NRxIf::NRxParamMgrResetComPar);
    paramMgrHead.ctrlWord0 = 0;
}

void NRxIf::NRxParamMgrResetComPar::NetToHostEndian(const NRxIf::NRxParamMgrResetComPar& src)
{
    head.NetToHostEndian(src.head);
    paramMgrHead.NetToHostEndian(src.paramMgrHead);
    body.NetToHostEndian(src.body);
    end.NetToHostEndian(src.end);
}

void NRxIf::NRxParamMgrResetComPar::HostToNetEndian(const NRxIf::NRxParamMgrResetComPar& src)
{
    head.HostToNetEndian(src.head);
    paramMgrHead.HostToNetEndian(src.paramMgrHead);
    body.HostToNetEndian(src.body);
    end.HostToNetEndian(src.end);
}

NRxIf::NRxParamMgrResetAreaParBody::NRxParamMgrResetAreaParBody()
{
    memset(this, 0, sizeof(*this));
}

void NRxIf::NRxParamMgrResetAreaParBody::NetToHostEndian(const NRxIf::NRxParamMgrResetAreaParBody& src)
{
    memcpy(areaName, src.areaName, sizeof(char) * 64); // 0, 区域名

    memcpy(tagName, src.tagName, sizeof(char) * 32); // 64, 标签名
}

void NRxIf::NRxParamMgrResetAreaParBody::HostToNetEndian(const NRxIf::NRxParamMgrResetAreaParBody& src)
{
    memcpy(areaName, src.areaName, sizeof(char) * 64); // 0, 区域名

    memcpy(tagName, src.tagName, sizeof(char) * 32); // 64, 标签名
}

NRxIf::NRxParamMgrResetAreaPar::NRxParamMgrResetAreaPar()
{
    head.tag = NRxIfTag_ParmMgr;
    head.msgBytes = sizeof(NRxIf::NRxParamMgrResetAreaPar);
    paramMgrHead.ctrlWord0 = 1;
}

void NRxIf::NRxParamMgrResetAreaPar::NetToHostEndian(const NRxIf::NRxParamMgrResetAreaPar& src)
{
    head.NetToHostEndian(src.head);
    paramMgrHead.NetToHostEndian(src.paramMgrHead);
    body.NetToHostEndian(src.body);
    end.NetToHostEndian(src.end);
}

void NRxIf::NRxParamMgrResetAreaPar::HostToNetEndian(const NRxIf::NRxParamMgrResetAreaPar& src)
{
    head.HostToNetEndian(src.head);
    paramMgrHead.HostToNetEndian(src.paramMgrHead);
    body.HostToNetEndian(src.body);
    end.HostToNetEndian(src.end);
}

NRxIf::NRxParamMgrRead::NRxParamMgrRead()
{
    head.tag = NRxIfTag_ParmMgr;
    head.msgBytes = sizeof(NRxIf::NRxParamMgrRead);
    paramMgrHead.ctrlWord0 = 2;
}

void NRxIf::NRxParamMgrRead::NetToHostEndian(const NRxIf::NRxParamMgrRead& src)
{
    head.NetToHostEndian(src.head);
    paramMgrHead.NetToHostEndian(src.paramMgrHead);
    end.NetToHostEndian(src.end);
}

void NRxIf::NRxParamMgrRead::HostToNetEndian(const NRxIf::NRxParamMgrRead& src)
{
    head.HostToNetEndian(src.head);
    paramMgrHead.HostToNetEndian(src.paramMgrHead);
    end.HostToNetEndian(src.end);
}

NRxIf::NRxParamMgrSave::NRxParamMgrSave()
{
    head.tag = NRxIfTag_ParmMgr;
    head.msgBytes = sizeof(NRxIf::NRxParamMgrSave);
    paramMgrHead.ctrlWord0 = 3;
}

void NRxIf::NRxParamMgrSave::NetToHostEndian(const NRxIf::NRxParamMgrSave& src)
{
    head.NetToHostEndian(src.head);
    paramMgrHead.NetToHostEndian(src.paramMgrHead);
    end.NetToHostEndian(src.end);
}

void NRxIf::NRxParamMgrSave::HostToNetEndian(const NRxIf::NRxParamMgrSave& src)
{
    head.HostToNetEndian(src.head);
    paramMgrHead.HostToNetEndian(src.paramMgrHead);
    end.HostToNetEndian(src.end);
}

NRxIf::NRxParamMgrSaveAsBody::NRxParamMgrSaveAsBody()
{
    memset(this, 0, sizeof(*this));
}

void NRxIf::NRxParamMgrSaveAsBody::NetToHostEndian(const NRxIf::NRxParamMgrSaveAsBody& src)
{
    memcpy(pathName, src.pathName, sizeof(char) * 256); // 0, 路径名
}

void NRxIf::NRxParamMgrSaveAsBody::HostToNetEndian(const NRxIf::NRxParamMgrSaveAsBody& src)
{
    memcpy(pathName, src.pathName, sizeof(char) * 256); // 0, 路径名
}

NRxIf::NRxParamMgrSaveAs::NRxParamMgrSaveAs()
{
    head.tag = NRxIfTag_ParmMgr;
    head.msgBytes = sizeof(NRxIf::NRxParamMgrSaveAs);
    paramMgrHead.ctrlWord0 = 4;
}

void NRxIf::NRxParamMgrSaveAs::NetToHostEndian(const NRxIf::NRxParamMgrSaveAs& src)
{
    head.NetToHostEndian(src.head);
    paramMgrHead.NetToHostEndian(src.paramMgrHead);
    body.NetToHostEndian(src.body);
    end.NetToHostEndian(src.end);
}

void NRxIf::NRxParamMgrSaveAs::HostToNetEndian(const NRxIf::NRxParamMgrSaveAs& src)
{
    head.HostToNetEndian(src.head);
    paramMgrHead.HostToNetEndian(src.paramMgrHead);
    body.HostToNetEndian(src.body);
    end.HostToNetEndian(src.end);
}

NRxIf::NRxParamMgrSetOneParaBody::NRxParamMgrSetOneParaBody()
{
    memset(this, 0, sizeof(*this));
}

void NRxIf::NRxParamMgrSetOneParaBody::NetToHostEndian(const NRxIf::NRxParamMgrSetOneParaBody& src)
{
    memcpy(areaName, src.areaName, sizeof(char) * 64); // 0, 区域名

    memcpy(tagName, src.tagName, sizeof(char) * 32); // 64, 标签名

    memcpy(paramName, src.paramName, sizeof(char) * 32); // 96, 参数名

    memcpy(paramVal, src.paramVal, sizeof(char) * 32); // 128, 参数值
}

void NRxIf::NRxParamMgrSetOneParaBody::HostToNetEndian(const NRxIf::NRxParamMgrSetOneParaBody& src)
{
    memcpy(areaName, src.areaName, sizeof(char) * 64); // 0, 区域名

    memcpy(tagName, src.tagName, sizeof(char) * 32); // 64, 标签名

    memcpy(paramName, src.paramName, sizeof(char) * 32); // 96, 参数名

    memcpy(paramVal, src.paramVal, sizeof(char) * 32); // 128, 参数值
}

NRxIf::NRxParamMgrSetParasBody::NRxParamMgrSetParasBody()
{
    memset(this, 0, sizeof(*this));
}

void NRxIf::NRxParamMgrSetParasBody::NetToHostEndian(const NRxIf::NRxParamMgrSetParasBody& src)
{
    paraNum = src.paraNum; // 0, 有效参数个数, <=8 (MTU = 1500)
    memcpy(res0, src.res0, sizeof(char) * 7); // 1, 预留

    for (int i = 0; i < 8; ++i) { // 8, 修改参数信息
        params[i].NetToHostEndian(src.params[i]);
    }
}

void NRxIf::NRxParamMgrSetParasBody::HostToNetEndian(const NRxIf::NRxParamMgrSetParasBody& src)
{
    paraNum = src.paraNum; // 0, 有效参数个数, <=8 (MTU = 1500)
    memcpy(res0, src.res0, sizeof(char) * 7); // 1, 预留

    for (int i = 0; i < 8; ++i) { // 8, 修改参数信息
        params[i].HostToNetEndian(src.params[i]);
    }
}

NRxIf::NRxParamMgrSetParas::NRxParamMgrSetParas()
{
    head.tag = NRxIfTag_ParmMgr;
    head.msgBytes = sizeof(NRxIf::NRxParamMgrSetParas);
    paramMgrHead.ctrlWord0 = 5;
}

void NRxIf::NRxParamMgrSetParas::NetToHostEndian(const NRxIf::NRxParamMgrSetParas& src)
{
    head.NetToHostEndian(src.head);
    paramMgrHead.NetToHostEndian(src.paramMgrHead);
    body.NetToHostEndian(src.body);
    end.NetToHostEndian(src.end);
}

void NRxIf::NRxParamMgrSetParas::HostToNetEndian(const NRxIf::NRxParamMgrSetParas& src)
{
    head.HostToNetEndian(src.head);
    paramMgrHead.HostToNetEndian(src.paramMgrHead);
    body.HostToNetEndian(src.body);
    end.HostToNetEndian(src.end);
}

NRxIf::NRxParamMgrFileGened::NRxParamMgrFileGened()
{
    head.tag = NRxIfTag_ParmMgr;
    head.msgBytes = sizeof(NRxIf::NRxParamMgrFileGened);
    paramMgrHead.ctrlWord0 = 6;
}

void NRxIf::NRxParamMgrFileGened::NetToHostEndian(const NRxIf::NRxParamMgrFileGened& src)
{
    head.NetToHostEndian(src.head);
    paramMgrHead.NetToHostEndian(src.paramMgrHead);
    end.NetToHostEndian(src.end);
}

void NRxIf::NRxParamMgrFileGened::HostToNetEndian(const NRxIf::NRxParamMgrFileGened& src)
{
    head.HostToNetEndian(src.head);
    paramMgrHead.HostToNetEndian(src.paramMgrHead);
    end.HostToNetEndian(src.end);
}

NRxIf::NRxSWHeartInfoBody::NRxSWHeartInfoBody()
{
    memset(this, 0, sizeof(*this));
}

void NRxIf::NRxSWHeartInfoBody::NetToHostEndian(const NRxIf::NRxSWHeartInfoBody& src)
{
    softwareID = ntohs(src.softwareID); // 0, 软件 ID
    memcpy(res0, src.res0, sizeof(char) * 6); // 2, 预留
}

void NRxIf::NRxSWHeartInfoBody::HostToNetEndian(const NRxIf::NRxSWHeartInfoBody& src)
{
    softwareID = htons(src.softwareID); // 0, 软件 ID
    memcpy(res0, src.res0, sizeof(char) * 6); // 2, 预留
}

NRxIf::NRxSWHeartInfo::NRxSWHeartInfo()
{
    init();
}

void NRxIf::NRxSWHeartInfo::NetToHostEndian(const NRxIf::NRxSWHeartInfo& src)
{
    head.NetToHostEndian(src.head);
    body.NetToHostEndian(src.body);
    end.NetToHostEndian(src.end);
}

void NRxIf::NRxSWHeartInfo::HostToNetEndian(const NRxIf::NRxSWHeartInfo& src)
{
    head.HostToNetEndian(src.head);
    body.HostToNetEndian(src.body);
    end.HostToNetEndian(src.end);
}

void NRxIf::NRxSWHeartInfo::init()
{
    head.head = NRxIfHead;
    head.tag = NRxIfTag_SWHeart;
    head.msgBytes = sizeof(NRxIf::NRxSWHeartInfo);
}

NRxIf::NRxSWStateInfoBody::NRxSWStateInfoBody()
{
    memset(this, 0, sizeof(*this));
}

void NRxIf::NRxSWStateInfoBody::NetToHostEndian(const NRxIf::NRxSWStateInfoBody& src)
{
    softwareID = ntohs(src.softwareID);
//    uint8 logGrade; // 1, 状态等级 0, debug; 1, info; 2, warning; 3, critical; 4, fatal
//    uint8 res[5]; // 2, 预留

//    uint8 state[1024]; // 8, 状态报文
}

void NRxIf::NRxSWStateInfoBody::HostToNetEndian(const NRxIf::NRxSWStateInfoBody &src)
{
    softwareID = htons(src.softwareID);
//    uint8 logGrade; // 1, 状态等级 0, debug; 1, info; 2, warning; 3, critical; 4, fatal
//    uint8 res[5]; // 2, 预留

//    uint8 state[1024]; // 8, 状态报文
}

NRxIf::NRxSWStateInfo::NRxSWStateInfo()
{
    head.tag = NRxIfTag_SWStat;
}

void NRxIf::NRxSWStateInfo::NetToHostEndian(const NRxIf::NRxSWStateInfo& src)
{
    head.NetToHostEndian(src.head);
    body.NetToHostEndian(src.body);
    end.NetToHostEndian(src.end);
}

void NRxIf::NRxSWStateInfo::HostToNetEndian(const NRxIf::NRxSWStateInfo& src)
{
    head.HostToNetEndian(src.head);
    body.HostToNetEndian(src.body);
    end.HostToNetEndian(src.end);
}

double NRxIf::jointHL32bit(unsigned int high, unsigned int low)
{
    typedef unsigned long long ull;
    return double( ((ull)(high) << 32) + low );
}

std::pair<unsigned int, unsigned int> NRxIf::getHL32bit(double data)
{
    typedef unsigned long long ull;
    const ull maxT32 = 0xFFFFFFFF;
    ull data64Bit = (ull)(data);
    unsigned int low = (data64Bit & maxT32); // get low 32 bit
    unsigned int high = ((data64Bit >> 32) & maxT32); // get high 32 bit

    return std::make_pair(low, high);
}

uint32 NRxIf::getCurMicroSeconds()
{
#ifdef WIN32
    SYSTEMTIME sys;
    GetLocalTime(&sys);
    return sys.wMilliseconds*1000;
#else
    timeval now;
    gettimeofday(&now, NULL);
    return now.tv_usec;
#endif
}

uint32 NRxIf::getCurLocalTime2US()
{
    unsigned int curTime(0);
#ifdef _WIN32
    SYSTEMTIME sys; // linux 不同
    GetLocalTime(&sys); // 获取Windows平台时间
    curTime = (unsigned int)(sys.wHour)*3600000 + (unsigned int)(sys.wMinute)*60000 +
            sys.wSecond*1000 + sys.wMilliseconds;
#else
    timeval now;
    gettimeofday(&now, NULL);
    unsigned int resSec = (unsigned long long)now.tv_sec % 86400 + 8 * 3600; // plus 8h
    curTime = resSec * 1000 + now.tv_usec / 1000.f; // s->ms; us->ms
//    std::cout << ("time is: %d", curTime) << std::endl;
#endif
    return curTime;
}

NRxIf::NRxSWRDRNoteBody::NRxSWRDRNoteBody()
{
    memset(this, 0, 1024);
}

void NRxIf::NRxSWRDRNoteBody::NetToHostEndian(const NRxIf::NRxSWRDRNoteBody& src)
{
    NRX_UNUSED(src);
}

void NRxIf::NRxSWRDRNoteBody::HostToNetEndian(const NRxIf::NRxSWRDRNoteBody& src)
{
    NRX_UNUSED(src);
}

NRxIf::NRxSWRDRNote::NRxSWRDRNote()
{
    head.msgBytes = sizeof(NRxIf::NRxSWRDRNote);
    head.tag = NRxIfTag_RDRNote;
}

void NRxIf::NRxSWRDRNote::NetToHostEndian(const NRxIf::NRxSWRDRNote& src)
{
    head.NetToHostEndian(src.head);
    body.NetToHostEndian(src.body);
    end.NetToHostEndian(src.end);
}

void NRxIf::NRxSWRDRNote::HostToNetEndian(const NRxIf::NRxSWRDRNote& src)
{
    head.HostToNetEndian(src.head);
    body.HostToNetEndian(src.body);
    end.HostToNetEndian(src.end);
}
