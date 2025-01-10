#ifndef NRXINTERFACE_H
#define NRXINTERFACE_H

#include <iostream>
#include "NRxType.h"

#pragma pack(1) // 内存中 1 字节对齐
// 代码格式 8 字节对齐, 同时 8 字节内部确保 4 字节对齐

namespace NRxIf
{
// 报文识别符
// 视频标签
#define NRxIfTag_FPGA_VidIQ (0x0100) // 雷达脉压前中频
#define NRxIfTag_FPGA_VidPC (0x0101) // 雷达脉压后中频
#define NRxIfTag_FPGA_Vid (0x0102) // 雷达脉压求模视频
#define NRxIfTag_VidOri (0x0103) // 原始视频
#define NRxIfTag_VidDet (0x0104) // 检测视频
// 点迹、航迹、AIS、ADS 标签
#define NRxIfTag_SectorPlots (0x010A) // 扇区点迹
#define NRxIfTag_Trk (0x010B) // 航迹
#define NRxIfTag_AISSP (0x0110) // AIS 原始报文, 串口(Serial Port)接收的标准报文
#define NRxIfTag_AISDEC (0x0111) // AIS 解码后报文, NRxAIS
#define NRxIfTag_ADSB (0x0120) // ADS-B
// 平台、雷达位置信息
#define NRxIfTag_RGCI (0x0130) // 雷达地理坐标 (Radar Geographic Coordinates Information)
#define NRxIfTag_RPNI (0x0131) // 平台导航 (Radar Platform Navigation Information)
// 人工备注信息
#define NRxIfTag_MNI (0x0140) // 人工备注信息 (Manual Notes Information)
// 操控命令
#define NRxIfTag_TrkMgr (0x0141) // 目标管理命令
#define NRxIfTag_AreaMgr (0x0142) // 区域管理
#define NRxIfTag_ParmMgr (0x0143) // 参数化
// 特殊命令与报文
#define NRxIfTag_ProjectMgr (0x0144) // 项目特定报文, 如导航中的试操船控制、试操船航迹等
// 软件状态通信
#define NRxIfTag_SWHeart (0x0200) // 软件心跳
#define NRxIfTag_SWStat (0x0201) // 软件实体状态
// 工作方式与天线控制
#define NRxIfTag_WorkMode (0x0300) // 工作方式
#define NRxIfTag_ServoCtrl (0x0301) // 伺服控制
// IFF
#define NRxIfTag_IFFReg (0x0400) // IFF 申请
#define NRxIfTag_IFFInfo (0x0401) // IFF 申请
// 雷达备注信息
#define NRxIfTag_RDRNote (0x0500) // 雷达备注信息

// 软件实体 ID
#define NRxSW_Disp (0x00) // 值勤显控软件 0x00-0x07
#define NRxSW_RewindDisp (0x08) // 回溯显控软件 0x08-0x0F
#define NRxSW_SigPro (0xA0) // 相参处理软件(信号处理) 0xA0-0xA3, 0xA4-0xA7预留
#define NRxSW_ObjDet (0xA8) // 非相参处理软件(目标检测) 0xA8-0xAB, 0xAC-0xAF预留
#define NRxSW_SglTracker (0xB0) // 单雷达跟踪软件 0xB0-0xB3, 0xB4-0xB7预留
#define NRxSW_EleWarn (0xB1) // 电子围栏软件 单独的进程由跟踪软件启动，视为跟踪软件的一部分.
#define NRxSW_AISFusion (0xB8) // 航迹与AIS/ADS融合软件 0xB8-0xBB, 0xBC-0xBF预留
#define NRxSW_MultTracker (0xC0) // 多雷达跟踪软件 0xC0-0xC3, 0xC4-0xC7预留
#define NRxSW_MultTrkFusion (0xC8) // 多雷达航迹融合软件 0xC8-0xCB

#define NRxIfHead (0xF1A2B4C8u) // 公共数据头 报文头
#define NRxProtolVerion (0u) // 当前协议版本号
#define NRxIfEnd1 (0xABCDu) // 公共数据尾 校验码1
#define NRxIfEnd2 (0xEF89u) // 公共数据尾 校验码2
#define NRxVidSyncHead (0xA5A61234u) // 雷达回波视频信息 同步头
#define NRxVidSyncTail (0xB5B65678u) // 雷达回波视频信息 同步尾
#define NRxDefaultSecNum (32) // 默认扇区数量
#define NRxMaxPlotsNum (680) // 点迹 最大点迹数量

// 航迹报文二级标识
#define NRxIfTag_Trk_BasicInfo (0x0A) // 航迹基础信息
#define NRxIfTag_Trk_CommonInfo (0x0B) // 航迹常用信息
#define NRxIfTag_Trk_TagInfo (0x0C) // 航迹标识信息
#define NRxIfTag_Trk_AdditionalInfo (0x0D) // 航迹附加信息
#define NRxIfTag_Trk_ProjectTrkInfo (0x0E) // 项目指定航迹信息，各项目相关
#define NRxIfTag_Trk_AisInfo (0xA0) // AIS信息
#define NRxIfTag_Trk_ADSBInfo (0xA2) // ADS-B
#define NRxIfTag_Trk_BatchTable (0xA4) // 雷达批号对照表(多雷达融合使用)

// 公共数据头 (32 bytes)
typedef struct NRxIfHeader
{
    uint32 head;      // 0, 报文头, 常数 0xF1A2B4C8.
    uint16 protocol;  // 4, 协议版本号, 当前版本号为0.
    uint16 counter;   // 6, 计数器, 各类报文独自计数.

    uint32 time;      // 8, 发送时刻1.    UTC. 由发方读取本地时间填写.
    uint32 microSecs; // 12, 发送时刻2.   表示秒以下的微秒数.

    uint16 msgBytes;  // 16, 报文总长度.
                        //  包含公共数据头、数据、数据尾的总字节数, 最大为64K字节.
                        //  FPGA发送的数据此项填0.
    uint16 tag;       // 18, 报文识别符.外部报文使用0-255, 内部报文使用256-65535.
    uint16 rdID;      // 20, 雷达ID.应用场景自定义.
    uint8  sendID;    // 22, 发方节点号.系统为能够发送或接收数据的软硬件实体分配节点号.
    uint8  rcvID;     // 23, 收方节点号.

    uint8  cpr;       // 24, 压缩标记.
                        //  b7-4数据压缩标记. 0x0,数据未压缩; 0x1,数据使用 qt sdk 压缩; 其它待定义.
                        //  b3, 数据中绝对时间格式.
                            // 0, 32位UTC.
                            // 1, 自定义: 绝对时间1表示0-86399999ms, 绝对时间2无效填0.
                        // b2-0, 预留, 填0.
    uint8  rdrChnID;  // 25, 数据采集时的数据通道号
    uint16 res1;  // 26, 预留, 填0.
    uint16 res2;  // 28, 预留, 填0.
    uint16 res3;  // 30, 预留, 填0.

    NRxIfHeader(void);// constructor
    void NetToHostEndian(const NRxIfHeader& src);
    void HostToNetEndian(const NRxIfHeader& src);
}NRxIfHeader;

// 项目相关特殊命令报文头（二级头） NRxIfHeader + NRxIfProjectMsgHeader + `ProjectMsg` + NRxIfEnd
typedef struct NRxIfProjectMsgHeader
{ // secondTag = 0x01 & res0 = 1 & res1 = 1 为调试显控报文
    uint16 secondTag; // 0, 报文二级标志
    uint16 projectTag; // 2, 项目编码 0x00 - 调试显控; 0x01 - 导航; 0x02 - 系留; 0x03 - RPS
    uint32 res1; // 4, 预留1

    NRxIfProjectMsgHeader(void);
    void NetToHostEndian(const NRxIfProjectMsgHeader& src);
    void HostToNetEndian(const NRxIfProjectMsgHeader& src);
}NRxIfProjectMsgHeader;

// 数据尾 (8 bytes)
typedef struct NRxIfEnd
{
    uint32 CRC ;  // 0, 校验码, 填0.
    uint16 end1;  // 4, 信息尾1, 填0xABCD.
    uint16 end2;  // 6, 信息尾2, 填0xEF89.

    NRxIfEnd(void);// constructor
    void NetToHostEndian(const NRxIfEnd& src);
    void HostToNetEndian(const NRxIfEnd& src);
}NRxIfEnd;

// 雷达回波视频头 (128 bytes)
// FPGA发送格式为: NRxIfHeader + NRxVidInfo + data (uint8[n]).
// 数据块发送格式为: NRxIfHeader + NRxVidInfo + data (uint8[n]) + NRxIfEnd.
typedef struct NRxVidInfo
{
    uint32 vidSyncHead; // 0, 同步头 0xA5A61234
    uint32 vidLength; // 4, 雷达视频长度 雷达视频编码后不压缩的字节数,
                      // 不包含数据头、雷达视频头和数据尾的长度. FPGA 填 0xFFFFFFFF.

    uint16 vidHeadLength; // 8, 雷达视频头长度 不包含数据, 填128
    uint16 vidFormat; // 10, 编码格式 0, 8位视频; 1, 16位视频; 2, 8位视频+8位背景
                            // 3, 16位视频+16位背景; 4, 8位视频+8位背景+8位速度+8位预留. 速度用补码表示.
                            // 64，16位I+16位Q
                            // 65，32位I+32位Q
                            // 66，16位I+16位Q+16位幅度+16位预留
                            // 67，32位I+32位Q+32位幅度+32位预留
                            // 100~127: 专用于内部计算使用; 100: 32位浮点数幅度(dB)+通道速度chnnalSpeed
                            // 101: 32位浮点数幅度(dB)+32位浮点数背景(dB)+通道速度chnnalSpeed
                            // 102: 32位浮点数幅度(dB)+32位浮点数背景(dB)+32位浮点数速度(m/s)
    uint8 pulseCombineMode; // 12, 脉冲组合方式 0: 单脉冲; 1: 多脉冲补盲; 2: MTD多通道; 3: 1长+多短
    uint8 subPulseNum; // 13, 子脉冲个数 脉组中子脉冲个数。
                            // 例如: 单脉冲无补盲脉冲, 该值填1; 单脉冲 1 补盲, 该值填 2;
                            // 1 个长脉冲 16 个短脉冲, 该值填17；
    uint8 subPulseNo; // 14, 子脉冲序号 当前脉冲在脉组中的子脉冲序号, [0, n)
    uint8 res0; // 15, 预留, 表示脉冲组合方式

    uint32 absTime0; // 16, 绝对时间1 32位UTC表示秒 或  0-86399999ms（由公共数据头中的数据标记的b3位决定）
    uint32 absTime1; // 20, 绝对时间2 表示秒以下的微秒数或无效（由公共数据头中的数据标记的b3位决定）

    uint32 relTime0; // 24, 相对时间 高32位, LSB: 1ms
    uint32 relTime1; // 28, 相对时间 低32位, LSB: 1ms

    uint32 bandWidth; // 32, 信号带宽 LSB: 1Hz, 0xFFFFFFFF
    uint32 sampleRate; // 36, 采样率 LSB: 1Hz

    uint16 azi; // 40, 方位 16位编码, 360度/65536
    uint16 pulseWidth; // 42, 脉宽 LSB: 0.1us, 0xFFFF
    uint16 prt; // 44, PRT LSB: 0.1us, 0xFFFF
    int16 startCellNo; // 46, 起始单元序号 第 0 个距离单元相对零距的单元数

    uint32 cellNum; // 48, 采样单元个数
    uint32 res1; // 52, 预留

    uint8 res2; // 56, 预留
    uint8 PIMFlag; // 57, PIM 标识 0: 原始, 1: 积累, 2: 填充.(外部填 0xFF 表示无效, 进入 PIM 后填)
    uint8 dataFlag; // 58, 数据标识, b7-5 视频幅度量纲. 0, dB; 1, 线性映射; 2, 约定非线性映射. b4-0预留
    uint8 mapPreLowerDB; // 59, 线性映射参数 线性映射时有效. 映射前幅度下限
    uint8 mapPreUpperDB; // 60, 线性映射参数 线性映射时有效. 映射前幅度上限
                                        // val = 0 (if DB<= mapPreLowerDB)
                                        // val = 2^n - 1 (if DB>= mapPreUpperDB)
                                        // val = (DB-mapPreLowerDB) / (mapPreUpperDB-mapPreLowerDB) * (2^n - 1)
    uint8 res3; // 61, 预留
    uint16 dataSource; // 62, 数据源 配置文件中通过按 位或 的方式使用. 最多支持16中数据源.
                            // 1:Simple Test; 2:Scenario Generator; 4:Replay From Recording;
                            // 8:Receive By UDP; 2^(4:15)预留

    int32 longitude; // 64 经度 LSB: 1/10000分, 181度表示无效
    int32 latitude; // 68 纬度 LSB: 1/10000分, 91度表示无效

    int16 high; // 72, 高度 LSB: 1米, 默认填0
    uint16 absCourse; // 74, 绝对航向 LSB: 360/65536, 默认填0
    uint16 absVel; // 76, 绝对航速 LSB: 0.1m/s, 默认填0
    uint16 relCourse; // 78, 相对航向 LSB: 360/65536, 默认填0

    uint16 relVel; // 80, 相对航速 LSB: 0.1m/s, 默认填0
    int16 headSway; // 82, 首摇 LSB: 360度/32768 默认值0
    int16 rollSway; // 84, 横摇 LSB: 360度/32768 默认值0
    int16 pitchSway; // 86, 纵摇 LSB: 360度/32768 默认值0

    uint8 scanType; // 88, 扫描方式
                        // b7: 表示以下天线扫描方式是否有效. 1, 有效 0, 无效.
                        // b6: 0, 固定平台; 1, 移动平台.
                        // b5 预留.
                        // b4-0: 0 顺时针环扫, 1 逆时针环扫, 2 顺时针机械扇扫, 3 逆时针机械扇扫,
                        // 4 顺时针单向电扫, 5 逆时针单向电扫, 6 随机扫描, 7 定位, 8 停车首, 9 手轮. 31 表示无效
    uint8 res4; // 89, 预留
    uint16 servoScanSpeed; // 90, 天线扫描速度 LSB:  0.1 deg/s, 0 = 无效
    uint16 servoStartAzi; // 92, 天线扇扫前沿 LSB: 360.f / 65536.f, 0 = 无效
    uint16 servoEndAzi; // 94, 天线扇扫后沿 LSB: 360.f / 65536.f, 0 = 无效

    int32 channelSpeed; // 96, 通道速度 速度用补码表示, LSB = 0.1m/s 0xFFFF时无效
    uint16 channelNo; // 100, 通道序号 0xFF时无效
    uint16 res5; // 102, 预留

    uint8 res6[16]; // 104, 预留

    uint32 res7; // 120, 报文尾 预留
    uint32 vidSyncTail; // 124, 报文尾 0xB5B65678

    NRxVidInfo(void);
    void NetToHostEndian(const NRxVidInfo& src);
    void HostToNetEndian(const NRxVidInfo& src);
}NRxVidInfo;

// 扇区点迹 (96 bytes)
// NRxMaxPlotsNum (680) 最大点迹数量 接收超过该数量的点迹丢弃
typedef struct NRxPlot
{
    uint32 absTime0; // 0, 绝对时间1, 32位UTC表示秒 或  0-86399999ms（由公共数据头中的数据标记的b3位决定）
    uint32 absTime1; // 4, 绝对时间2, 表示秒以下的微秒数或无效（由公共数据头中的数据标记的b3位决定）

    uint32 relTime0; // 8, 相对时间 高32位, LSB: 1ms
    uint32 relTime1; // 12, 相对时间 低32位, LSB: 1ms

    uint32 dis; // 16, 距离 LSB: 1m
    uint16 azi; // 20, 方位 LSB: 360.f / 65536.f
    int16 ele; // 22, 仰角 LSB: 180.f / 32768.f

    uint32 validAttr; // 24, 有效标记 0, 无效; 1, 有效
                      // b0 幅度; b1 EP数; b2 距离前后沿; b3 方位前后沿; b4 饱和度; b5 信杂噪比;
                      // b6 多普勒速度; b7 多普勒速度极差; b8 区域类型; b9 点迹判别结果;
                      // b10 点迹类型; b11 目标置信度; b12-31 预留.
    uint16 amp; // 28, 幅度
    uint16 ep; // 30, EP 数

    uint32 disStart; // 32, 距离起始 LSB: 0.01m
    uint32 disEnd; // 36, 距离终止 LSB: 0.01m

    uint16 aziStart; // 40, 方位起始 LSB: 360.f / 65536.f, 顺时针方向
    uint16 aziEnd; // 42, 方位终止 LSB: 360.f / 65536.f
    uint8 saturate; // 44, 饱和度 百分比量化为 0-100
    uint8 SNR; // 45, 信杂噪比
    int16 doppVel; // 46, 多普勒速度 LSB: 0.1m/s

    uint16 maxDoppDiff; // 48, 多普勒速度极差 LSB: 0.1m/s
    uint8 areaType; // 50, 区域类型 0 不明 1 噪声 2 杂波
    uint8 plotRetain; // 51, 点迹判别结果 0 保留; 1 智能剔除; 2 副瓣剔除; 3 滤波剔除 显控和跟踪只认该标志
    uint8 plotType; // 52, 点迹类型 0 目标; 1 距离副瓣; 2 方位副瓣; 3 云雨; 4 海浪; 5 地物 智能化学习结果 显控和跟踪不使用该标记
    uint8 plotConfLv; // 53, 点迹置信度 百分比量化为 0-100, 点迹不是杂波的置信度
    uint8 res0[2]; // 54, 预留

    uint8 plotTypeConfLv[16]; // 56, 点迹类型置信度 百分比量化为 0-100 所有点迹类型置信度的和为 100 顺序按点迹类型填写, 多余的为预留

    uint16 BaGAmp; // 72, 背景幅度
    uint16 ThrAmp; // 74, 门限幅度
    int32 plot_id;
    // uint8 res1[4]; // 76, 预留

    uint8 res2[16]; // 80, 预留


    NRxPlot(void);// constructor
    void NetToHostEndian(const NRxPlot& src);
    void HostToNetEndian(const NRxPlot& src);
}NRxPlot;

// 扇区点迹头 (88 bytes)
// 发送格式为: NRxIfHeader + NRxSectorInfo + NRxPlot*n (n <= NRxMaxPlotsNum) + NRxIfEnd
// NRxMaxPlotsNum (680) 最大点迹数量 接收超过该数量的点迹丢弃
typedef struct NRxSectorInfo
{
    uint8 secNo; // 0, 扇区号
    uint8 secNum; // 1, 扇区总数
    uint16 plotsNum; // 2, 扇区点迹数 最大支持 NRxMaxPlotsNum(680) 个
    uint8 res0[4]; // 4, 预留

    uint32 startAbsTime0; // 8, 前沿绝对时间1 32位UTC表示秒 或  0-86399999ms（由公共数据头中的数据标记的b3位决定）
    uint32 startAbsTime1; // 12, 前沿绝对时间2 表示秒以下的微秒数或无效, 由公共数据头中的数据标记的b3位决定

    uint32 endAbsTime0; // 16, 后沿绝对时间1 32位UTC表示秒 或  0-86399999ms（由公共数据头中的数据标记的b3位决定）
    uint32 endAbsTime1; // 20, 后沿绝对时间2 表示秒以下的微秒数或无效, 由公共数据头中的数据标记的b3位决定

    uint32 startRelTime0; // 24, 前沿相对时间 高32位, LSB: 1ms
    uint32 startRelTime1; // 28, 前沿相对时间 低32位, LSB: 1ms

    uint32 endRelTime0; // 32, 后沿相对时间 高32位, LSB: 1ms
    uint32 endRelTime1; // 36, 后沿相对时间 低32位, LSB: 1ms

    int32 longitude; // 40, 平台经度 LSB: 1/10000分, 181度表示无效
    int32 latitude; // 44, 平台纬度 LSB: 1/10000分, 91度表示无效

    int16 height; // 48, 平台高度 LSB: 1米, 默认填0
    uint16 absCourse; // 50, 平台绝对航向 LSB: 360/65536, 默认填0
    uint16 absVel; // 52, 平台绝对航速 LSB: 0.1m/s, 默认填0
    uint16 relCourse; // 54, 平台相对航向 LSB: 360/65536, 默认填0

    uint16 relVel; // 56, 平台相对航速 LSB: 0.1m/s, 默认填0
    int16 headSway; // 58, 平台首摇 LSB: 360度/32768 默认值0
    int16 rollSway; // 60, 平台横摇 LSB: 360度/32768 默认值0
    int16 pitchSway; // 62, 平台纵摇 LSB: 360度/32768 默认值0

    uint8 scanType; // 64, 扫描方式
                        // b7: 表示以下天线扫描方式是否有效. 1, 有效 0, 无效.
                        // b6: 0, 固定平台; 1, 移动平台.
                        // b5 预留.
                        // b4-0: 0 顺时针环扫, 1 逆时针环扫, 2 顺时针机械扇扫, 3 逆时针机械扇扫,
                        // 4 顺时针单向电扫, 5 逆时针单向电扫, 6 随机扫描, 7 定位, 8 停车首, 9 手轮. 31 表示无效
    uint8 res1; // 65, 预留
    uint16 servoScanSpeed; // 66, 天线扫描速度 LSB:  0.1 deg/s, 0 = 无效
    uint16 servoStartAzi; // 68, 天线扇扫前沿 LSB: 360.f / 65536.f, 0 = 无效
    uint16 servoEndAzi; // 70, 天线扇扫后沿 LSB: 360.f / 65536.f, 0 = 无效

    uint8 res2[16]; // 72, 预留

    NRxSectorInfo(void);
    void NetToHostEndian(const NRxSectorInfo& src);
    void HostToNetEndian(const NRxSectorInfo& src);
}NRxSectorInfo;

// 航迹 多个短报文拼接而成
// 航迹 - 基础信息 (104 bytes)
typedef struct NRxTrk_BasicInfo
{
    uint16 msgTag; // 0, 航迹报文二级标识 0x0A 基础信息; 0x0B 常用信息; 0x0C 标识信息;
                        // 0x0D 附加信息; 0xA0 AIS信息; ADS-B 0xA2; 0xA4 雷达批号对照表(多雷达融合使用)
    uint16 bytes;  // 2, 基础信息从0地址开始的总字节数.
    uint8 res0[4]; // 4, 预留

    uint16 batchNo; // 8, 批号 四位BCD码
    uint16 seqNo; // 10, 序号 0-65536, 跟踪内部序号
    uint16 trkStat; // 12, 状态 b15-14: 0 新航迹; 1 航迹更新; 2 航迹撤消; 3航迹丢失.
                        // b13: 0 临时航迹; 1 正式航迹.
                        // b12: 0 海目标; 1 空目标.
                        // b11: 0 人工录取; 1 自动录取.
                        // b10-9: 0 自动跟踪; 1 辅助跟踪; 2 人工跟踪.
                        // b8: 0 自动删批; 1 人工删批.
                        // b7-0: 备用.
    uint16 ctrlWord0; // 14, 控制字 表示相应字段是否有效 0, 无效; 1, 有效
                            // b0 幅度滤波值; b1 EP数滤波值; b2 距离展宽滤波值; b3 方位展宽滤波值;
                            // b4 仰角展宽滤波值; b5 径向速度滤波值; b6 -15 预留.

    uint32 absTime0; // 16, 绝对时间1, 32位UTC表示秒 或  0-86399999ms（由公共数据头中的数据标记的b3位决定）
    uint32 absTime1; // 20, 绝对时间2, 表示秒以下的微秒数或无效（由公共数据头中的数据标记的b3位决定）

    uint32 relTime0; // 24, 相对时间 高32位, LSB: 1ms
    uint32 relTime1; // 28, 相对时间 低32位, LSB: 1ms

    uint32 dis; // 32, 距离 LSB: 1m
    uint16 azi; // 36, 方位 LSB: 360.f / 65536.f
    int16 ele; // 38, 仰角 LSB: 180.f / 32768.f

    uint16 absCourse; // 40, 绝对航向 LSB: 360/65536, 默认填0
    uint16 absVel; // 42, 绝对航速 LSB: 0.1m/s, 默认填0
    uint16 relCourse; // 44, 相对航向 LSB: 360/65536, 默认填0
    uint16 relVel; // 46, 相对航速 LSB: 0.1m/s, 默认填0

    int32 longitude; // 48, 经度 LSB: 1/10000分, 181度表示无效
    int32 latitude; // 52, 纬度 LSB: 1/10000分, 91度表示无效

    int16 high; // 56, 高度 LSB: 1米, 默认填0
    uint8 contiLostPlots; // 58, 连续丢点数 允许范围: 0-99
    uint8 trkQuality; // 59, 航迹质量 允许范围: 0-99
    uint16 updateTime; // 60, 更新次数 关联+外推次数
    uint16 ampFiltVal; // 62, 幅度滤波值

    uint16 epFiltVal; // 64, EP 数滤波值
    uint16 disSpanFiltVal; // 66, 距离展宽滤波值 LSB: 0.01m
    uint16 aziSpanFiltVal; // 68, 方位展宽滤波值 LSB: 360/65536
    uint16 eleSpanFiltVal; // 70, 仰角展宽滤波值 LSB: 180/32768

    uint32 doppVelFiltVal; // 72, 径向速度滤波值 LSB: 0.1m/s
    uint32 duration; // 76, 航迹持续时间 LSB: 1s

    uint32 disCoast; // 80, 距离外推值 LSB: 1m
    uint16 aziCoast; // 84, 方位外推值 LSB: 360.f / 65536.f
    int16 eleCoast; // 86, 仰角外推值 LSB: 180.f / 32768.f

    uint32 mmsi; // 88, MMSI 号, 0xFFFFFFFF 表示无效
    uint8 res1[4]; // 92, 预留

    uint8 res2[8]; // 96, 预留

    NRxTrk_BasicInfo(void);
    void NetToHostEndian(const NRxTrk_BasicInfo& src);
    void HostToNetEndian(const NRxTrk_BasicInfo& src);
}NRxTrk_BasicInfo;

// 航迹 - 常用信息 (136 bytes)
typedef struct NRxTrk_CommonInfo
{
    uint16 msgTag; // 0, 航迹报文二级标识 0x0A 基础信息; 0x0B 常用信息; 0x0C 标识信息;
        // 0x0D 附加信息; 0xA0 AIS信息; ADS-B 0xA2; 0xA4 雷达批号对照表(多雷达融合使用)
    uint16 bytes;  // 2, 常用信息从0地址开始的总字节数.
    uint8 res0[4]; // 4, 预留

    uint32 ctrlWord0; // 8, 控制字 表示相应字段是否有效, 0, 无效; 1, 有效
        // b0 距离标准差; b1 方位标准差; b2 仰角标准差; b3 幅度标准差;
        // b4 EP数标准差; b5 距离展宽标准差; b6 方位展宽标准差; b7 仰角展宽标准差;
        // b8 径向速度标准差; b9 直线加速度; b10 角速度; b11 - 31 预留.
    uint32 assoPlotDis; // 12, 关联点迹距离 LSB: 1m

    uint16 assoPlotAzi; // 16, 关联点迹方位 LSB: 360.f / 65536.f
    int16 assoPlotEle; // 18, 关联点迹仰角 LSB: 180.f / 32768.f
    uint16 assoPlotAmp; // 20, 关联点迹幅度
    uint16 assoPlotEP; // 22, 关联点迹 EP 数

    uint16 assoPlotDisSpan; // 24, 关联点迹距离展宽 LSB: 0.01m
    uint16 assoPlotAziSpan; // 26, 关联点迹方位展宽 LSB: 360.f / 65536.f
    uint8 res1[28]; // 28, 关联点迹预留 28(4 + 24) 字节

    uint8 res2[5]; // 56, 预留
    uint8  gatePlotsCount; // 61, 波门内点迹数	0-255.
    uint16 gateType; // 62, 波门类型 0波门无效. 后续波门字段按注释顺序排列.
        // 1 二维椭圆波门, 椭圆波门切向半轴长度, 单位: 1m; 椭圆波门径向半轴长度, 单位: 1m.
        // 2 二维扇形波门,  距离前沿, 距离后沿, 方位前沿, 方位后沿.

    uint32 gateCenterDis; // 64, 波门中心距离 LSB: 1m
    uint16 gateCenterAzi; // 68, 波门中心方位 LSB: 360.f / 65536.f
    int16 gateCenterEle; // 70, 波门中心仰角 LSB: 180.f / 32768.f

    uint32 gateParam0; // 72, 波门参数 1 无效时填0
    uint32 gateParam1; // 76, 波门参数 2 无效时填0

    uint32 gateParam2; // 80, 波门参数 3 无效时填0
    uint32 gateParam3; // 84, 波门参数 4 无效时填0

    uint32 gateParam4; // 88, 波门参数 5 无效时填0
    uint32 gateParam5; // 92, 波门参数 6 无效时填0

    uint16 disStd; // 96, 距离标准差 无效时填0
    uint16 aziStd; // 98, 方位标准差 无效时填0
    uint16 eleStd; // 100, 仰角标准差 无效时填0
    uint16 ampStd; // 102, 幅度标准差 无效时填0

    uint16 epStd; // 104, EP数标准差 无效时填0
    uint16 disSpanStd; // 106, 距离展宽标准差 LSB：0.01米,无效时填0
    uint16 aziSpanStd; // 108, 方位展宽标准差 LSB：360/65536,无效时填0
    uint16 eleSpanStd; // 110, 仰角展宽标准差 LSB：180/32768,无效时填0

    uint16 doppVelStd; // 112, 径向速度标准差 单位: 0.1m/s, 无效时填 0
    uint16 linearAcceleration; // 114, 直线加速度 无效时填 0
    uint8 angularVel; // 116, 角速度 无效时填0
    uint8 motionState; // 117, 运动状态 0 不明; 1 静止; 2 匀速直线运动; 3 机动
    uint8 res3; // 118, 预留
    uint8 res4; // 119, 预留

    uint16 elecWarn1; // 120, 电子围栏告警信息. b15表示是否告警(1告警0不告警); b14-12预留; b11-0表示航迹至最近围栏的秒数(最大4095秒).
    uint16 elecWarn2; // 122, 电子围栏告警信息. 表示航迹至最近围栏的距离，单位: 1米(最大65535米).
    uint8 res5[12]; // 124, 预留

    NRxTrk_CommonInfo(void);
    void NetToHostEndian(const NRxTrk_CommonInfo& src);
    void HostToNetEndian(const NRxTrk_CommonInfo& src);
}NRxTrk_CommonInfo;

// 航迹 - 标识信息 (40 bytes)
typedef struct NRxTrk_MarkInfo
{
    uint16 msgTag; // 0, 航迹报文二级标识 0x0A 基础信息; 0x0B 常用信息; 0x0C 标识信息;
        // 0x0D 附加信息; 0xA0 AIS信息; ADS-B 0xA2; 0xA4 雷达批号对照表(多雷达融合使用)
        // NRxIfTag_Trk_TagInfo
    uint16 bytes; // 2, 标识信息从0地址开始的总字节数.
    uint8 res0[4]; // 4, 预留

    uint16 tag0; // 8, 标志 0, b15 目标是否为目指(0: 非目指, 1: 目指); b14 目标是否为重点(0非重点, 1重点)
            // b13 上报(0 不上报, 1 上报); b12-b5 预留; b4-0 敌我(0 不明, 1 敌, 2 我, 3 友)
    uint16 tag1; // 10, 标志 1 国籍
    uint16 tag2; // 12, 标志 2 数量
    uint16 tag3; // 14, 标志 3 型号

    uint16 tag4; // 16, 标志 4 队形
    uint16 tag5; // 18, 标志 5
    uint16 tag6; // 20, 标志 6
    uint16 tag7; // 22, 标志 7

    uint16 tag8; // 24, 标志 8
    uint16 tag9; // 26, 标志 9
    uint16 tag10; // 28, 标志 10
    uint16 tag11; // 30, 标志 11

    uint16 tag12; // 32, 标志 12
    uint16 tag13; // 34, 标志 13
    uint16 tag14; // 36, 标志 14
    uint16 tag15; // 38, 标志 15

    NRxTrk_MarkInfo(void);
    void NetToHostEndian(const NRxTrk_MarkInfo& src);
    void HostToNetEndian(const NRxTrk_MarkInfo& src);
}NRxTrk_MarkInfo;

//// 航迹 - 附加信息
//typedef struct NRxTrk_AdditionalInfo
//{
//    uint16 msgTag; // 0, 航迹报文二级标识 0x0A 基础信息; 0x0B 常用信息; 0x0C 标识信息;
//        // 0x0D 附加信息; 0xA0 AIS信息; ADS-B 0xA2; 0xA4 雷达批号对照表(多雷达融合使用)
//        // NRxIfTag_Trk_AdditionalInfo
//    uint16 bytes;  // 2, 标识信息从0地址开始的总字节数.
//    uint8 res0[4]; // 4, 预留

//}NRxTrk_AdditionalInfo;

//// 航迹 - AIS 信息, 一般不发, 若显控有要求, 与后续AIS报文内容一致
//typedef struct NRxTrk_AisInfo
//{
//    uint16 msgTag; // 0, 航迹报文二级标识 0x0A 基础信息; 0x0B 常用信息; 0x0C 标识信息;
//        // 0x0D 附加信息; 0xA0 AIS信息; ADS-B 0xA2; 0xA4 雷达批号对照表(多雷达融合使用)
//        // #define NRxIfTag_Trk_AisInfo (0xA0) // AIS信息
//    uint16 bytes;  // 2, 标识信息从0地址开始的总字节数.
//    uint8 res0[4]; // 4, 预留

//}NRxTrk_AisInfo;

//// 航迹 - ADS-B 信息, 报文暂缺
//typedef struct NRxTrk_ADSBInfo
//{
//    uint16 msgTag; // 0, 航迹报文二级标识 0x0A 基础信息; 0x0B 常用信息; 0x0C 标识信息;
//        // 0x0D 附加信息; 0xA0 AIS信息; ADS-B 0xA2; 0xA4 雷达批号对照表(多雷达融合使用)
//    uint16 bytes;  // 2, 标识信息从0地址开始的总字节数.
//    uint8 res0[4]; // 4, 预留

//}NRxTrk_ADSBInfo;

// AIS 数据 (136 bytes)
// 系统设计时，应预先将静态信息和动态信息合并.
typedef struct NRxAISBody
{
    uint16 status; // 0, AIS 状态字. 0 新 AIS; 1 AIS 更新; 2 AIS 删除.
    uint16 aisType; // 2, 导航定位类型, BCD 码表示. 0x00: 未定义; 0x01: GPS; 0x02: GLDNASS;
                        // 0x03: GPS/GLDNASS; 0x04: 罗兰; 0x05: chayka; 0x06: 综合导航系统;
                        // 0x07: 观测; 0x08及以上不可用
    uint32 mmsi; // 4, 用户识别码, MMSI号码（AIS 国籍为该号码前 3 位, 其中 412, 413 为中国）

    int32 longitude; // 8, 经度 单位：1/10000分;单位取值：-180°~+180°, 东经为正, 西经为负; 181°=不可用=默认
    int32 latitude; // 12, 纬度 单位：1/10000分;单位取值：-90°~+90°, 北纬为正, 南纬为负; 91°=不可用=默认

    uint32 res0; // 16, 预留
    uint16 groundSpeed; // 20, 真航速 0.1m/s
    uint16 groundCourse; // 22, 对地航向 360/32768, 65535 为无效

    uint16 trueCourse; // 24, 真航向, 航向角/艏向角, 360/32768, 65535 为无效
    int16 power; // 26, 功率, 1dBm, 一般为负值
    uint8 callLeters[7]; // 28, 呼号, 8 位 ASCII 字符, “@@@@@@@”=不可用=默认
    uint8 boatName[21]; // 35, 船名, 21 位 ASCII 字符, “@@@@@@@@@@@@@@@@@@@@@”=不可用=默认

    uint32 expectRcvTime; // 56, 预定到达时间, 单位:b31-24月, b23-16日, b15-8时, b7-0分, 月: 0无效; 日: 0无效; 时: 24无效; 分: 60无效
    uint16 length; // 60, 船长, 0.1m, 0=无效
    uint16 width; // 62, 船宽, 0.1m, 0=无效

    uint8 naviStatus; // 64, 航行状态 用BCD码表示:0x00：在航（主机推动）;0x01：锚泊;0x02：失控;0x03：操纵受限;0x04：吃水受限;
                        // 0x05：靠泊;0x06：搁浅;0x07：捕捞作业;0x08：靠船帆提供动力;0x09：为将来HSC航行状态修正所保留;0x10：为将来WIG航行状态修正所保留;0x15：未定义, 缺省
    uint8 res1; // 65, 预留
    uint16 shipType; // 66, 船舶及载货类型, 0=不可用或没有船舶=预设;
                        // 1~99:50=引航船 51=搜救船  52=拖轮 53=港口供应船 54=载有放污染装置和设备的船舶
                        // 55=执法艇 56=备用-用于当地船舶的任务分配 57=备用-用于当地船舶的任务分配 58=医疗船 59=符合18号决议的船舶等;
                        // 100~199=为地区性使用保留; 200~255=为今后使用保留
    uint16 maxDrawWaterLine; // 68, 最大吃水深度, 0.1m, 0 为无效
    int16 turnRate; // 70, 转弯率, -128: 无法获得; +127: 每分钟右转 720°; -127: 每分钟左转 720°;
                        // -127~127: 以 4.733°每分钟偏转, 负数表示向右转, 正数表示向左转

    uint16 posPrecision; // 72, 位置精度, BCD 码表示, 默认为 0. 0x01: 高(<10m, DGNSS 接收机差分模式); 0x00: 低(>10m, GNSS 接收机或其他电子定位装置的自主模式)
    uint8 origin[21]; // 74, 始发地, 没有填 @, 参照 ASCII 码表
    uint8 res2; // 95, 预留

    uint32 revTime0; // 96, 消息接收绝对时间1, 32位UTC表示秒 或  0-86399999ms（由公共数据头中的数据标记的b3位决定）
    uint32 revTime1; // 100, 消息接收绝对时间2, 表示秒以下的微秒数或无效（由公共数据头中的数据标记的b3位决定）

    uint8 destination[21]; // 104, 目的地, 没有填 @, 参照 ASCII 码表
    uint8 res3[11]; // 125, 备用

    NRxAISBody(void);
    void NetToHostEndian(const NRxAISBody& src);
    void HostToNetEndian(const NRxAISBody& src);
}NRxAISBody;

// AIS 报文
typedef struct NRxAIS
{
    NRxIfHeader head;
    NRxAISBody body;
    NRxIfEnd end;

    NRxAIS(void);
    void NetToHostEndian(const NRxAIS& src);
    void HostToNetEndian(const NRxAIS& src);
}NRxAIS;

// 目标管理命令 (32 bytes)
// 根据 ctrlWord0 中标识, 表示不同操控报文
typedef struct NRxObjMgrBody
{
    uint16 ctrlWord0; // 0, 键盘命令字 0
    uint16 ctrlWord1; // 2, 键盘命令字 1
    uint16 ctrlWord2; // 4, 键盘命令字 2
    uint16 ctrlWord3; // 6, 键盘命令字 3

    uint16 ctrlWord4; // 8, 键盘命令字 4
    uint16 ctrlWord5; // 10, 键盘命令字 5
    uint16 ctrlWord6; // 12, 键盘命令字 6
    uint16 ctrlWord7; // 14, 键盘命令字 7

    uint16 ctrlWord8; // 16, 键盘命令字 8
    uint16 ctrlWord9; // 18, 键盘命令字 9
    uint16 ctrlWord10; // 20, 键盘命令字 10
    uint16 ctrlWord11; // 22, 键盘命令字 11

    uint16 ctrlWord12; // 24, 键盘命令字 12
    uint16 ctrlWord13; // 26, 键盘命令字 13
    uint16 ctrlWord14; // 28, 键盘命令字 14
    uint16 ctrlWord15; // 30, 键盘命令字 15
    // ============================== 操控报文 ============================== //
    // 下述报文中所有未涉及到的控制字均填 0

    // ---------- 0. 录取 ---------- //
    // cmd0, 0x30, 表示录取(自动编批)、编批录取、多点录取.
    // cmd1, 目标类型 0海目标 1空目标
    // cmd2, 方位, LSB:360/65536.
    // cmd3, 距离(低字)LSB：1m
    // cmd4, 距离(高字)LSB：1m
    // cmd5, 录取方式 (0 自动编批; 1 手动; 2 多点录取)
    // cmd6, 批号BCD码, cmd5 == 1 (或cmd5==2且非首点录取)时有效
    // cmd7, 录取时所在窗口单像素点表示的距离(1m) // 根据量程与距离换算
    // cmd8, 多点录取次数 cmd5==2 时有效，两点录取填 2.
    // cmd9, 多点录取序号(从0开始) 多点录取首点可以自动编批或人工编批，后续必须指定批号.
    // cmd10, 导航: 0 - 正常录取; 1 - 录取为参考目标

    // ---------- 1. 删批 ---------- //
    // cmd0, 0x31, 删批.
    // cmd1, 删批方式. 1, 删1批; 2, 区域删除(cmd3-8表示扇形区域); 0xFFFF, 全删.
    // cmd2, 批号BCD码（cmd1 == 1 时有效，cmd1其它值无效）
    // cmd3, 距离前沿（低字）LSB：1m, cmd1 == 2 时有效
    // cmd4, 距离前沿（高字）LSB：1m, cmd1 == 2 时有效
    // cmd5, 距离后沿（低字）LSB：1m, cmd1 == 2 时有效
    // cmd6, 距离后沿（高字）LSB：1m, cmd1 == 2 时有效
    // cmd7, 方位前沿, LSB:360/65536, cmd1 == 2 时有效
    // cmd8, 方位后沿, LSB:360/65536, cmd1 == 2 时有效

    // ---------- 2. 辅跟、人工跟踪 ---------- //
    // cmd0, 0x32, 辅跟、人工跟踪
    // cmd1, 执行码. 0 取消; 1 设置.
    // cmd2, 跟踪类型. 0 辅跟; 1 人工跟踪
    // cmd3, 批号 BCD 码
    // cmd4, 方位, LSB: 360/65536.
    // cmd5, 距离（低字）LSB：1m
    // cmd6, 距离（高字）LSB：1m
    // cmd7, 录取时所在窗口单像素点表示的距离(1m) // 根据量程与距离换算

    // ---------- 3. 改批 ---------- //
    // cmd0, 0x33, 改批
    // cmd1, 原 BCD 码
    // cmd2, 新 BCD 码

    // ---------- 4. 换批 ---------- //
    // cmd0, 0x34, 换批
    // cmd1, BCD 码 1
    // cmd2, BCD 码 2

    // ---------- 5. 属性设置 ---------- //
    // cmd0, 0x35, 属性设置
    // cmd1, BCD 码
    // cmd2-cmd15 可用于依次表示多个属性，第一个属性序号为 0xFFFF 时，表示后续属性无效
    // cmd2, 属性的序号. 0-255 表示跟踪相关属性, 256-0xFFFE 仅作为标记使用.
    // 预设置标记: 0 目指（0非目指, 1目指）; 1 重点（0非重点, 1重点）; 2 删除标记（0自动删除, 1人工删除）;
    // 3 上报（0 不上报, 1 上报）; 4 交接（预留）; 5 主动引导被动（预留）; 6 敌我（0 不明, 1 敌, 2 我, 3 友）;
    // 7 目标海空类型（0 海, 1 空）; 8 国籍; 9 数量; 10 型号; 11 队形
    // cmd3, 属性值

    // ---------- 20. 电子围栏告警（告警程序 --> 跟踪） ---------- //
    // cmd0, 0x90, 电子围栏告警
    // cmd1, 批号(BCD 码)
    // cmd2, b15 表示是否告警(1 告警 0 不告警); b14-12 预留; b11 - 0 表示航迹至最近围栏的秒数(最大4095秒).
    // cmd3, 表示航迹至最近围栏的距离，单位: 1米(最大65535米).
    // cmd2 与 cmd3 的定义与 "航迹 - 常用信息 NRxTrk_CommonInfo" 中的 elecWarn1 和 elecWarn2 一致.

    // ===================================================================== //

    NRxObjMgrBody(void);
    void NetToHostEndian(const NRxObjMgrBody& src);
    void HostToNetEndian(const NRxObjMgrBody& src);
}NRxObjMgrBody;

// 目标管理命令
typedef struct NRxObjMgr
{
    NRxIfHeader head;
    NRxObjMgrBody body;
    NRxIfEnd end;

    NRxObjMgr(void);
    void NetToHostEndian(const NRxObjMgr& src);
    void HostToNetEndian(const NRxObjMgr& src);
}NRxObjMgr;

// 区域管理命令 - 区域顶点 (4 bytes)
typedef struct NRxAreaPlot
{
    uint32 dis; // 0, 距离 LSB: 1m
    uint16 azi; // 4, 方位 LSB: 360.f / 65536.f
    uint16 res; // 6, 预留

    NRxAreaPlot(void);
    void NetToHostEndian(const NRxAreaPlot& src);
    void HostToNetEndian(const NRxAreaPlot& src);
}NRxAreaPlot;

// 区域管理命令 (72 + 4 * n bytes) 变长报文, 根据实际顶点数量发送
typedef struct NRxAreaMgr
{
    uint8 ctrlWord; // 0, 控制字 0 增加一个区域; 1 删除一个区域; 2 删除所有用户区域;
                        // 3 修改区域（增减多边形区域的顶点, 或修改某个顶点的位置; 内部实现时先删除原区域, 再增加新的区域, 区域间的先后顺序发生变化）
                // 4，修改区域属性（只有区域附加属性和区域名有效）
    uint8 areaType; // 1, 区域类型 0 扇形; 1 多边形. 扇形时, 方位0表示方位前沿, 方位1表示方位后沿, 距离在内部判断, 使用较小值作为距离起始.
    uint8 plotNum; // 2, 顶点数 最大 256. 扇形时填2. 多边形的顶点是不封闭的，即首末两点不同.
    uint8 attrs;    // 3,区域附加属性. b7对海自动录取开关（0关1开）；b6对空自动录取开关（0关1开）；b5-0预留
    uint8 role; // 4, 区域用途. 0 自动录取; 1 信号处理; 2 电子围栏; 其他预留.
    uint8 res0[3]; // 5, 预留

    uint8 name[64]; // 8, 区域名, 最大64字符. 有效字符数不足64时 必须以'\0'结尾；有效字符数等于64时不需要'\0'结尾.

    // 72 bytes
    // NRxAreaPlot plot[N]; // 变长顶点

    NRxAreaMgr(void);
    void NetToHostEndian(const NRxAreaMgr& src);
    void HostToNetEndian(const NRxAreaMgr& src);
}NRxAreaMgr;

// 参数化管理命令
// 发送格式为: NRxIfHeader + NRxParamMgrHead + NRxParamMgrInfo(对应控制字下的具体报文内容)
// 参数化管理命令 - 二级数据头
typedef struct NRxParamMgrHead
{
    uint8 ctrlWord0; // 0, 用途 0 表示恢复通用参数中指定或所有Tag下参数的默认值.
                        // 1 表示恢复指定区域或所有区域中指定或所有Tag下参数的默认值.
                        // 2 表示读配置文件. 无后续报文
                        // 3 表示写配置文件. 无后续报文
                        // 4 表示另存为.
                        // 5 表示修改参数值.
                        // 6 表示配置文件已生成 无后续报文（在处理软件初始化完成后通知显控，只有一份配置文件存在处理软件路径中，
                                // 通过挂载或网络硬盘实现. 可以配置一小型数据库实现参数在不同机器中的同步功能）
    uint8 sendID; // 1, 软件 ID, 与公共数据头中发方节点号相同
    uint8 res[6]; // 2, 预留

    NRxParamMgrHead(void);
    void NetToHostEndian(const NRxParamMgrHead& src);
    void HostToNetEndian(const NRxParamMgrHead& src);
}NRxParamMgrHead;

// 参数化管理命令 - 恢复通用默认参数
typedef struct NRxParamMgrResetComParBody
{
    uint8 tagName[32]; // 0, 标签名, 最大32字符. 有效字符数不足32时 必须以'\0'结尾
                            // 有效字符数等于32时不需要'\0'结尾. tagName[0] = '\0'时, 表示所有 tag

    NRxParamMgrResetComParBody(void);
    void NetToHostEndian(const NRxParamMgrResetComParBody& src);
    void HostToNetEndian(const NRxParamMgrResetComParBody& src);
}NRxParamMgrResetComParBody;

typedef struct NRxParamMgrResetComPar
{
    NRxIfHeader head; // 报文头
    NRxParamMgrHead paramMgrHead; // 参数管理二级头
    NRxParamMgrResetComParBody body;
    NRxIfEnd end;

    NRxParamMgrResetComPar(void);
    void NetToHostEndian(const NRxParamMgrResetComPar& src);
    void HostToNetEndian(const NRxParamMgrResetComPar& src);
}NRxParamMgrResetComPar;

// 参数化管理命令 - 恢复区域参数默认值
typedef struct NRxParamMgrResetAreaParBody
{
    uint8 areaName[64]; // 0, 区域名, 最大64字符. 有效字符数不足64时 必须以'\0'结尾;
                            // 有效字符数等于64时不需要'\0'结尾. areaName[0] = '\0'时, 表示所有 area

    uint8 tagName[32]; // 64, 标签名, 最大32字符. 有效字符数不足32时 必须以'\0'结尾
                            // 有效字符数等于32时不需要'\0'结尾. tagName[0] = '\0'时, 表示所有 tag

    NRxParamMgrResetAreaParBody(void);
    void NetToHostEndian(const NRxParamMgrResetAreaParBody& src);
    void HostToNetEndian(const NRxParamMgrResetAreaParBody& src);
}NRxParamMgrResetAreaParBody;

typedef struct NRxParamMgrResetAreaPar
{
    NRxIfHeader head; // 报文头
    NRxParamMgrHead paramMgrHead; // 参数管理二级头
    NRxParamMgrResetAreaParBody body;
    NRxIfEnd end;

    NRxParamMgrResetAreaPar(void);
    void NetToHostEndian(const NRxParamMgrResetAreaPar& src);
    void HostToNetEndian(const NRxParamMgrResetAreaPar& src);
}NRxParamMgrResetAreaPar;

// 参数化管理命令 - 读配置报文
typedef struct NRxParamMgrRead
{
    NRxIfHeader head; // 报文头
    NRxParamMgrHead paramMgrHead; // 参数管理二级头
    NRxIfEnd end;

    NRxParamMgrRead(void);
    void NetToHostEndian(const NRxParamMgrRead& src);
    void HostToNetEndian(const NRxParamMgrRead& src);
}NRxParamMgrRead;

// 参数化管理命令 - 写配置报文
typedef struct NRxParamMgrSave
{
    NRxIfHeader head; // 报文头
    NRxParamMgrHead paramMgrHead; // 参数管理二级头
    NRxIfEnd end;

    NRxParamMgrSave(void);
    void NetToHostEndian(const NRxParamMgrSave& src);
    void HostToNetEndian(const NRxParamMgrSave& src);
}NRxParamMgrSave;

// 参数化管理命令 - 另存为 (256 bytes)
typedef struct NRxParamMgrSaveAsBody
{
    uint8 pathName[256]; // 0, 路径名, 修改当前默认参数文件路径，并将软件内存中参数写入该路径
                            // 最大256字符. 有效字符数不足256时 必须以'\0'结尾;
                            // 有效字符数等于256时不需要'\0'结尾

    NRxParamMgrSaveAsBody(void);
    void NetToHostEndian(const NRxParamMgrSaveAsBody& src);
    void HostToNetEndian(const NRxParamMgrSaveAsBody& src);
}NRxParamMgrSaveAsBody;

typedef struct NRxParamMgrSaveAs
{
    NRxIfHeader head; // 报文头
    NRxParamMgrHead paramMgrHead; // 参数管理二级头
    NRxParamMgrSaveAsBody body;
    NRxIfEnd end;

    NRxParamMgrSaveAs(void);
    void NetToHostEndian(const NRxParamMgrSaveAs& src);
    void HostToNetEndian(const NRxParamMgrSaveAs& src);
}NRxParamMgrSaveAs;

// 参数化管理命令 - 修改参数值(paraBody 160 bytes)
typedef struct NRxParamMgrSetOneParaBody
{
    uint8 areaName[64]; // 0, 区域名, 最大64字符. 有效字符数不足64时 必须以'\0'结尾;
                            // 有效字符数等于64时不需要'\0'结尾. areaName[0] = "Global" 时, 表示通用参数

    uint8 tagName[32]; // 64, 标签名, 最大32字符. 有效字符数不足32时 必须以'\0'结尾
                            // 有效字符数等于32时不需要'\0'结尾.

    uint8 paramName[32]; // 96, 参数名, 最大32字符. 有效字符数不足32时 必须以'\0'结尾
                            // 有效字符数等于32时不需要'\0'结尾.

    uint8 paramVal[32]; // 128, 参数值, 最大32字符. 有效字符数不足32时 必须以'\0'结尾
                            // 有效字符数等于32时不需要'\0'结尾.

    NRxParamMgrSetOneParaBody(void);
    void NetToHostEndian(const NRxParamMgrSetOneParaBody& src);
    void HostToNetEndian(const NRxParamMgrSetOneParaBody& src);
}NRxParamMgrSetOneParaBody;

// 参数化管理命令 - 修改参数值
typedef struct NRxParamMgrSetParasBody
{
    uint8 paraNum; // 0, 有效参数个数, <=8 (MTU = 1500)
    uint8 res0[7]; // 1, 预留

    NRxParamMgrSetOneParaBody params[8]; // 8, 修改参数信息

    NRxParamMgrSetParasBody(void);
    void NetToHostEndian(const NRxParamMgrSetParasBody& src);
    void HostToNetEndian(const NRxParamMgrSetParasBody& src);
}NRxParamMgrSetParasBody;

typedef struct NRxParamMgrSetParas
{
    NRxIfHeader head; // 报文头
    NRxParamMgrHead paramMgrHead; // 参数管理二级头
    NRxParamMgrSetParasBody body;
    NRxIfEnd end;

    NRxParamMgrSetParas(void);
    void NetToHostEndian(const NRxParamMgrSetParas& src);
    void HostToNetEndian(const NRxParamMgrSetParas& src);
}NRxParamMgrSetParas;

// 参数化管理命令 - 配置文件已生成 0 bytes, 无后续命令
typedef struct NRxParamMgrFileGened
{
    NRxIfHeader head; // 报文头
    NRxParamMgrHead paramMgrHead; // 参数管理二级头
    NRxIfEnd end;

    NRxParamMgrFileGened(void);
    void NetToHostEndian(const NRxParamMgrFileGened& src);
    void HostToNetEndian(const NRxParamMgrFileGened& src);
}NRxParamMgrFileGened;

// 软件心跳
typedef struct NRxSWHeartInfoBody
{
    uint16 softwareID; // 0, 软件 ID, 参见宏定义 - 软件实体 ID 部分
    uint8 res0[6]; // 2, 预留

    NRxSWHeartInfoBody(void);
    void NetToHostEndian(const NRxSWHeartInfoBody& src);
    void HostToNetEndian(const NRxSWHeartInfoBody& src);
}NRxSWHeartInfoBody;

typedef struct NRxSWHeartInfo
{
    NRxIfHeader head;
    NRxSWHeartInfoBody body;
    NRxIfEnd end;

    NRxSWHeartInfo(void);
    void NetToHostEndian(const NRxSWHeartInfo& src);
    void HostToNetEndian(const NRxSWHeartInfo& src);
    void init(void);
}NRxSWHeartInfo;

// 软件实体状态
typedef struct NRxSWStateInfoBody
{
    uint16 softwareID; // 0, 软件 ID, 参见宏定义 - 软件实体 ID 部分
    uint8 logGrade; // 1, 状态等级 0, debug; 1, info; 2, warning; 3, critical; 4, fatal
    uint8 res[5]; // 2, 预留

    uint8 state[1024]; // 8, 状态报文

    NRxSWStateInfoBody();
    void NetToHostEndian(const NRxSWStateInfoBody& src);
    void HostToNetEndian(const NRxSWStateInfoBody& src);
}NRxSWStateInfoBody;

typedef struct NRxSWStateInfo
{
    NRxIfHeader head;
    NRxSWStateInfoBody body;
    NRxIfEnd end;

    NRxSWStateInfo();
    void NetToHostEndian(const NRxSWStateInfo& src);
    void HostToNetEndian(const NRxSWStateInfo& src);
}NRxSWStateInfo;

typedef struct NRxSWRDRNoteBody
{
    uint8 note[1024];

    NRxSWRDRNoteBody();
    void NetToHostEndian(const NRxSWRDRNoteBody& src);
    void HostToNetEndian(const NRxSWRDRNoteBody& src);
}NRxSWRDRNoteBody;

typedef struct NRxSWRDRNote
{
    NRxIfHeader head;
    NRxSWRDRNoteBody body;
    NRxIfEnd end;

    NRxSWRDRNote();
    void NetToHostEndian(const NRxSWRDRNote& src);
    void HostToNetEndian(const NRxSWRDRNote& src);
}NRxSWRDRNote;

///
/// \brief jointHL32bit, transform high and low 32 bits to double
/// \param high, high 32 bit of data
/// \param low, low 32 bit of data
/// \return
///     double data
///
extern double jointHL32bit(unsigned int high, unsigned int low);

///
/// \brief getHL32bit, transform double to high and low 32 bits
/// \param time, time to conversion, LSB: 1s
/// \return
///     pair made by <low, high>
///     usage:
///         auto hlData = getHL32bit(data);
///         unsigned int dataLow = hlData.first;
///         unsigned int dataHigh = hlData.second
///
extern std::pair<unsigned int, unsigned int> getHL32bit(double data);

///
/// \brief getCurMicroSeconds, get microseconds of current time
/// \return
///     microseconds of current time
///
extern uint32 getCurMicroSeconds(void);

///
/// \brief getCurLocalTime2US, get current local time(ms)
/// \return
///     local time(us, microseconds)
///
extern uint32 getCurLocalTime2US(void);

} // namespace NRxIf
#pragma pack()// 恢复之前的对齐
#endif // NRXINTERFACE_H
