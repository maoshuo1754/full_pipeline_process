#ifndef OTHERINTERFACE_H
#define OTHERINTERFACE_H

#include <iostream>
#include <cstring>
#ifdef WIN32 // WIN32
#include "winsock.h"
#else // Linux
#include "arpa/inet.h"
#endif

static const unsigned int XX9_MaxSampleCellNum = 25000;
static const unsigned int SPx_MaxSampleCellNum = 8192;

#pragma pack(1) // 内存中 1 字节对齐
// 代码格式 8 字节对齐, 同时 8 字节内部确保 4 字节对齐
extern long long htonll(long long src);

namespace msgXS {
//报文头结构体
#define XSJCTAG_SECPLOTS (0xD3) // 本地扇区点迹
struct  tagMsgHead
{
    unsigned short               usHead;                 //信息头 0x7E7E
    unsigned int                 ulTime;
    unsigned char                ucSendNode;             //发方节点号 1 DD;2 ZJ;目前永兴调度按此转发20230609
    unsigned char                ucRecvNode;             //收方节点号
    unsigned short               usDragLength;           //信息总长度 包括报文头报文尾
    unsigned char                ucDragCAT;              //信息识别符
    unsigned char                ucInSureMark;           //信息确认标识
    unsigned char                ucRes1;                 //信息单元备用
    unsigned char                ucRes2;                 //信息单元备用
    unsigned short               usRes3;                 //信息单元备用

    tagMsgHead()
    {
        memset(this, 0, sizeof(*this));
        usHead = 0x7E7E;
    }
};

struct tagNetQBHead
{
    unsigned short  usDragLength;         //报文长度
    unsigned short  usResus;           //备用
    unsigned char  ucSendIP[4];           //报文源地址
    unsigned char  ucReceiveIP[4];        //报文目的地址

    unsigned long ucSendDeviceDrag;//高
    //unsigned char  ucSendDeviceDrag;      //发方设备标识
    //unsigned char  ucSendSystemDrag;      //发方系统标识
    //unsigned short  ucSendEntityDrag;      //发方实体标识

    unsigned long ucRecDeviceDrag;

    //unsigned char  ucRecDeviceDrag;   //收方设备标识
    //unsigned char  ucRecSystemDrag;   //收方系统标识
    //unsigned short  ucRecEntityDrag;   //收方实体标识
    unsigned char  ucSerialNumber;        //序列号
    unsigned char  ucInforNum;            //信息单元个数
    unsigned short  usRes;                //备用

    tagNetQBHead()
    {
        memset(this,0,sizeof(*this));
        //ucSendIP[0] = IP_1_INITIDISPLAYCTRL;
        //ucSendIP[1] = IP_2_INITIDISPLAYCTRL;
        //ucSendIP[2] = IP_3_INITIDISPLAYCTRL;
        //ucSendIP[3] = IP_4_INITIDISPLAYCTRL;
        //ucReceiveIP[0] = IP_1_QBBZSystem;
        //ucReceiveIP[1] = IP_2_QBBZSystem;
        //ucReceiveIP[2] = IP_3_QBBZSystem;
        //ucReceiveIP[3] = IP_4_QBBZSystem;
        ucInforNum = 1;
        usResus = 3;
        ucSendDeviceDrag = (0x0150<<16)+0x3001;
        ucRecDeviceDrag  = (0x0301<<16)+0x4ae0;
    }
};

//28所信息单元头
struct tagInforQBHead
{
    unsigned char  ucInforNumber;       //信息单元序号
    unsigned char  ucResuc;             //备用
    unsigned short  usResus;            //备用
    unsigned short  usInforDragCATFlag;        //信息单元标识
    unsigned short  usInforLength;      //信息单元长度
    unsigned long   ulInforTimer;       //信息时戳
};

//视频增强切换
struct SendSetKeyTargetMsg_videoswitch
{
    tagNetQBHead  st_tagNetQBHead;
    tagInforQBHead  st_tagInforQBHead; // usInforDragCATFlag = 1520
    int  IQVideo;//0 检测 1原始 2增强
    int  iEnhaced;//0,不增强，1增强 暂时不用
};

// 报文结构体 视频
struct _EntireMessageInfo
{
    unsigned int   TrigleFlag;			// [0000-0001]同步头：0xD8D80606
    unsigned int   ServoFlag;			// [0002-0003]同步头：0xF4F4F4F4
    unsigned char RecvPoint; 			//雷达站点号,6a:0;19:1~255
    unsigned char PackKind;				//包数据种类,原始IQ-0;脉压后IQ-1；脉压后IQ＋求模-2;原始求模视频-3；检测后视频-4；点迹-5；增强视频-6
    unsigned char ResWord1[6];			//备用
    unsigned short AziVal;				// [0004]方位，单位：360/65536
    unsigned short DisCellNum;			//距离采样单元数
    unsigned long long Time1;			//时统时间
    unsigned long long Time2;			//自定义时间
    unsigned char FlagBit;				//标志位
    unsigned char ucSendSwitch;			//发射控制开关
    unsigned char ucDisPatch;			//主动调度工作模式
    unsigned short AreaWord;			//区域标志
    unsigned char STCDistance;			//STC距离
    unsigned char STCDepth;				//STC深度
    unsigned char STCLine;				//STC曲线
    unsigned char RecChannGainSet;		//接收通道增益控制
    unsigned char AntState;				//天线状态
    unsigned short FanScanCenter;		//扇扫中心
    unsigned short FanScanRange;		//扇扫范围
    unsigned char NoiseParaTest;		//噪声系数测试
    unsigned char TarAnalogSwitch;		//目标模拟开关
    unsigned char TarAnalogPara1[40];	//6A用模拟视频
    unsigned short NarwTrSlntStartAz1;	//发射方位静默起始1
    unsigned short NarwTrSlntEndAz1;	//发射方位静默结束1
    unsigned short NarwTrSlntStartAz2;	//发射方位静默起始2
    unsigned short NarwTrSlntEndAz2;	//发射方位静默结束2
    unsigned short NarwTrSlntStartAz3;	//发射方位静默起始3
    unsigned short NarwTrSlntEndAz3;	//发射方位静默结束3
    unsigned short NarwTrSlntStartAz4;	//发射方位静默起始4
    unsigned short NarwTrSlntEndAz4;	//发射方位静默结束4
    unsigned short ProcessSwitch;		//信号处理功能开关
    unsigned short fThrCFAR;			//目标检测门限
    unsigned short antiIntSwitch;		//抗干扰功能开关
    unsigned short AreaEnhance1;		//区域１增强开关
    unsigned short AreaEnhance2;		//区域２增强开关
    unsigned char WideBandPara[10];		//宽带参数
    unsigned char FrequencyPoint;		//固定工作频点
    unsigned int Fault;					//故障
    unsigned char GroupPerNum;			//周期有效总数
    unsigned short GroupPer;			//当前周期
    unsigned short PulseNumTotal;		//脉冲总数
    unsigned short PulseNumCurr;		//当前脉冲数
    unsigned short PulseWidth;			//当前脉宽参数
    unsigned short LurePara;			//当前诱骗参数
    unsigned char TarAnalogPara;		//目标模拟参数
    unsigned char TarVel[9];			//目标１－９速度
    unsigned char SendFrontAdj;			//发射前沿调整
    unsigned char SendBackAdj;			//发射后沿调整
    unsigned char ColFunSwitch;			//采集功能开关
    unsigned char ResWord3[3];			//保留字
    unsigned char usCultterPara[9];		//杂波感知参数
    unsigned char usMTIPara[11];		//MTI参数
    unsigned char usMTDPara[11];		//MTD参数
    unsigned char usNCIPara[5];			//脉冲积累参数
    unsigned char usCFARPara[10];		//CAFR参数
    unsigned char usCFARFeedPara[5];	//虚警反馈参数
    unsigned char usPlotExtractPara[12];//点迹提取参数
    unsigned char usPlotMergePara;		//点迹合并参数
    unsigned char usVCPara[11];			//帧间积累参数
    unsigned char usDatatrans[2];		//数据转换参数
    unsigned char usMaxCh[2];			//通道取大参数
    unsigned char usPulsegroupPara[11]; //脉组处理参数
    unsigned char ContAreaDet[2];	    //联通域检测
    unsigned short ShipAzi1;			//航向1
    unsigned short ShipVel1;			//航速1
    unsigned short ShipAzi2;			//航向２
    unsigned short ShipVel2;			//航速2
    int longitude;						//经度;单位:1/10000分
    int latitude;						//纬度;单位:1/10000分
    short height;						//高度;单位:1米
    unsigned char ResWord2[6];			//保留字
    unsigned int EchoFlag1;			    // 零距开始：0xA5A51234
    unsigned int EchoFlag2;			    //零距开始：0xA5A51234
    unsigned int EchoFlag3;			    //

    _EntireMessageInfo()
    {
        memset(this, 0, sizeof(*this));
        TrigleFlag = 0xD8D80606;
        ServoFlag = 0xF4F4F4F4;
        EchoFlag1 = 0xA5A51234;
        EchoFlag2 = 0xA5A51234;
    }
};

//报文尾结构体
struct tagMsgEnd
{
    unsigned short              usCRC;                  //CRC校验码
    unsigned short              usEnd;                  //信息尾 0xAAAA

    tagMsgEnd()
    {
        memset(this, 0, sizeof(*this));
        usEnd = 0xAAAA;
    }
};

//视频数据格式
struct VideoMessageInfo
{
    tagMsgHead  st_tagMsgHead;
    _EntireMessageInfo  m_MessageInfo;//主动定义报文头
    unsigned char data[XX9_MaxSampleCellNum];       //回波数据
    tagMsgEnd st_tagMsgEnd;
    VideoMessageInfo()
    {
        memset(data, 0, sizeof(unsigned char)*XX9_MaxSampleCellNum);
    }
};

// XSJC head and tail
typedef struct XSJCMsgHead
{
    unsigned short usHead; // 0 报文头   0x7E7E
    unsigned int unTime; // 2 报文发送时间，单位:1毫秒
    unsigned char ucSendNode; // 6 发方节点号
    unsigned char ucRecvNode; // 7 收方节点号
    unsigned short usLen; // 8 报文长度 ，除头尾最长不超过1024字节
    unsigned char ucCat; // 10 报文识别符
    unsigned char ucMask; // 11 确认标志
    unsigned short usCounter; // 12 流水号，各类报文独自编号
    unsigned short usReserve2; // 16 备用2

public:
    XSJCMsgHead()
    {
        memset(this, 0, sizeof(*this));
        usHead = 0x7E7E;
    }
    void Reset(void)
    {
        memset(this, 0, sizeof(*this));
        usHead = 0x7E7E;
        ucSendNode = 0;
    }
    // 主机 -> 网络
    void SwapBytesToNet(void)
    {
        usHead = htons(usHead);// 0 报文头   0x7E7E
        unTime = htonl(unTime);// 2 报文发送时间，单位:1毫秒
        //        unsigned char	ucSendNode;	// 6 发方节点号
        //        unsigned char	ucRecvNode;	// 7 收方节点号
        usLen = htons(usLen);// 8 报文长度 ，除头尾最长不超过1024字节
        //        unsigned char	ucCat;		// 10 报文识别符
        //        unsigned char	ucMask;		// 11 确认标志
        usCounter = htons(usCounter);	// 12 流水号，各类报文独自编号
        //        unsigned short	usReserve2;	// 16 备用2
        //                            // 20 end
    }
    // 网络 -> 主机
    void SwapBytesToHost(void)
    {
        usHead = ntohs(usHead);// 0 报文头   0x7E7E
        unTime = ntohl(unTime);// 2 报文发送时间，单位:1毫秒
        //        unsigned char	ucSendNode;	// 6 发方节点号
        //        unsigned char	ucRecvNode;	// 7 收方节点号
        usLen = ntohs(usLen);// 8 报文长度 ，除头尾最长不超过1024字节
        //        unsigned char	ucCat;		// 10 报文识别符
        //        unsigned char	ucMask;		// 11 确认标志
        usCounter = ntohs(usCounter);	// 12 流水号，各类报文独自编号
        //        unsigned short	usReserve2;	// 16 备用2
        //                            // 20 end
    }
}XSJCMsgHead;

// msg end
typedef struct XSJCMsgEnd
{
    unsigned short  usCRC;  //校验码
    unsigned short  usEnd;  //信息尾    0xAAAA

    XSJCMsgEnd(void)
        : usCRC(0)
        , usEnd(0xAAAA)
    {
    }
    bool Check(void) const
    {
        return (usCRC == 0 && usEnd == 0xAAAA);
    }
    // 主机 -> 网络
    void SwapBytesToNet(void)
    {
        usCRC = htons(usCRC);//校验码
        usEnd = htons(usEnd);//信息尾    0xAAAA
    }
    // 网络 -> 主机
    void SwapBytesToHost(void)
    {
        usCRC = ntohs(usCRC);//校验码
        usEnd = ntohs(usEnd);//信息尾    0xAAAA
    }
}XSJCMsgEnd;

// 64位整数表示ms
typedef struct tagTms64
{
    typedef unsigned long long uint64;
    unsigned int tHigh; // 64位整数的高32位
    unsigned int tLow; // 64位整数的低32位

    tagTms64(void)
        : tHigh(0)
        , tLow(0)
    {
    }
    tagTms64(unsigned int high, unsigned int low)
        : tHigh(high)
        , tLow(low)
    {
    }

    unsigned long long toUint64(void)const
    {
        return ((uint64(tHigh) << 32) + tLow);
    }
    double toDouble(void)const
    {
        return double((uint64(tHigh) << 32) + tLow);
    }
    tagTms64 &fromDouble(double t)
    {
        const uint64 maxT32 = 0xFFFFFFFF;
        uint64 t64 = uint64(t);
        tLow = (t64 & maxT32);
        tHigh = ((t64 >> 32) & maxT32);
        return (*this);
    }

    void SwapBytesToNet(void) // 主机 -> 网络
    {
        tHigh = htonl(tHigh);   // 64位整数的高32位
        tLow = htonl(tLow);     // 64位整数的低32位
    }
    void SwapBytesToHost(void) // 网络 -> 主机
    {
        tHigh = ntohl(tHigh);   // 64位整数的高32位
        tLow = ntohl(tLow);     // 64位整数的低32位
    }

    tagTms64 &AddMs(int ms)
    {
        const unsigned long long maxT32 = 0xFFFFFFFF;
        unsigned long long t = (uint64(tHigh) << 32) + tLow;
        t += ms;
        tLow = (t & maxT32);
        tHigh = ((t >> 32) & maxT32);
        return (*this);
    }
}tagTms64;

// 点迹(调度 --> 显控)
typedef struct XSJCDetectPlot
{
    tagTms64 tRefMs;    // 64位相对时间，单位:1毫秒
    unsigned int unTime; // 绝对时间, 1ms

    unsigned int  unCommand;  // 非必填字段是否有效. b0仰角, b1幅度, b2多普勒通道号, b3主通道比例, b4多普勒通道数, b3点迹属性, b4用途属性, b5信杂噪比,
        // b6饱和度, b7点迹属性, b8杂波等级.
    unsigned short usPlotNo;

    unsigned short usAzi; // 360/65536

    unsigned int unDis; // 1m

    unsigned short usEle; // 360/65536
    unsigned short usAmp; // 1

    unsigned short usDisSpan; // 1m
    unsigned short usAziSpan; // 360/65536

    unsigned short usEpNum; // 1
//    unsigned short usReverse;

    unsigned char usDoppChannel;
    unsigned char usDoppRat; // 0-255
    unsigned char usDoppNum;
    unsigned char ucBackFlag; // 未知0,噪声区1，海杂波区2，气象杂波区3,干扰区4，地杂波5

    unsigned char ucTaskFlag; // 未知0,军港1，民港2,锚区3，航道4
    unsigned char usScnr; // 0-255
    unsigned char ucSat; // 0-255
    unsigned char ucFeature; // 未知0,距离副瓣1，方位副瓣2
    unsigned char ucQuality;      // 点迹质量 0代表不合格，1代表合格
    unsigned char ucClutterLevel; // 杂波等级，0-低，1-高
    unsigned short usRev0;         // 备用0
    unsigned short usRev1;         // 备用1

    // 总共是 48 字节
    XSJCDetectPlot(void)
    {
        memset(this, 0, sizeof(*this));
    }

    void SwapBytesToNet(void) // 主机 -> 网络
    {
        unTime = htonl(unTime); // 绝对时间, 1ms

        unCommand = htonl(unCommand); // 非必填字段是否有效. b0仰角, b1幅度, b2多普勒通道号, b3主通道比例, b4多普勒通道数, b3点迹属性, b4用途属性, b5信杂噪比,
        // b6饱和度, b7点迹属性, b8杂波等级.
        usPlotNo = htons(usPlotNo);

        usAzi = htons(usAzi); // 360/65536

        unDis = htonl(unDis); // 1m

        usEle = htons(usEle); // 360/65536
        usAmp = htons(usAmp); // 1

        usDisSpan = htons(usDisSpan); // 1m
        usAziSpan = htons(usAziSpan); // 360/65536

        usEpNum = htons(usEpNum); // 1

//        unsigned char usDoppChannel;
//        unsigned char usDoppRat; // 0-255
//        unsigned char usDoppNum;
//        unsigned char ucBackFlag; // 未知0,噪声区1，海杂波区2，气象杂波区3,干扰区4，地杂波5

//        unsigned char ucTaskFlag; // 未知0,军港1，民港2,锚区3，航道4
//        unsigned char usScnr; // 0-255
//        unsigned char ucSat; // 0-255
//        unsigned char ucFeature; // 未知0,距离副瓣1，方位副瓣2
        usRev0 = htons(usRev0);         // 备用0
        usRev1 = htons(usRev1);         // 备用1
    }

    void SwapBytesToHost(void) // 网络 -> 主机
    {
        unTime = ntohl(unTime); // 绝对时间, 1ms

        unCommand = ntohl(unCommand); // 非必填字段是否有效. b0仰角, b1幅度, b2多普勒通道号, b3主通道比例, b4多普勒通道数, b3点迹属性, b4用途属性, b5信杂噪比,
        // b6饱和度, b7点迹属性, b8杂波等级.
        usPlotNo = ntohs(usPlotNo);

        usAzi = ntohs(usAzi); // 360/65536

        unDis = ntohl(unDis); // 1m

        usEle = ntohs(usEle); // 360/65536
        usAmp = ntohs(usAmp); // 1

        usDisSpan = ntohs(usDisSpan); // 1m
        usAziSpan = ntohs(usAziSpan); // 360/65536

        usEpNum = ntohs(usEpNum); // 1

//        unsigned char usDoppChannel;
//        unsigned char usDoppRat; // 0-255
//        unsigned char usDoppNum;
//        unsigned char ucBackFlag; // 未知0,噪声区1，海杂波区2，气象杂波区3,干扰区4，地杂波5

//        unsigned char ucTaskFlag; // 未知0,军港1，民港2,锚区3，航道4
//        unsigned char usScnr; // 0-255
//        unsigned char ucSat; // 0-255
//        unsigned char ucFeature; // 未知0,距离副瓣1，方位副瓣2
        usRev0 = ntohs(usRev0);         // 备用0
        usRev1 = ntohs(usRev1);         // 备用1
    }
}XSJCDetectPlot;

// 点迹凝聚产生的扇区点迹 14436字节
#define XSJC_MAXPTSONESEC (800u)
typedef struct XSJCDetectSectorBody
{
    unsigned char dotSrc; // 0: 6a; 1-255: 19
    unsigned char   ucSectorNo; // 扇区号
    unsigned short  usPlotNum;  // 当前扇区内点迹数

    unsigned int  unWorkMode; // (0,1,2,3)

    unsigned int usShipCourse; // 360/65536

    unsigned int usShipSpeed; // 0.1m/s

    unsigned int nLat; // 1e-6 deg

    unsigned int nLon; // 1e-6 deg

    unsigned int usHeight; // 0.1m

    unsigned int usResv1;

    unsigned int unStartAbsTime; // 扇区前沿绝对时间，单位：1ms
    unsigned int unStopAbsTime; // 扇区后沿绝对时间，单位：1ms

    unsigned char usAttenScanVel; // 0.1deg
    unsigned char ucAttenScanFlag; // 0 环扫; 1 扇扫
    unsigned char ucAttenScanDir; // 0 顺扫; 1 逆扫
    unsigned char usAttenScanCenter; // 360/65536

    unsigned char usAttenScanWide; // 360/65536

    XSJCDetectPlot stPlot[XSJC_MAXPTSONESEC]; // 扇区内nDotNum个点的结构

    XSJCDetectSectorBody(void)
    {
        memset(this, 0, sizeof(*this));
    }

    void SwapBytesToNet(void) // 主机 -> 网络
    {
//        unsigned char dotSrc; // 0: 6a; 1-255: 19
//        unsigned char   ucSectorNo; // 扇区号
        unWorkMode = htonl(unWorkMode); // (0,1,2,3)

        usShipCourse = htonl(usShipCourse); // 360/65536

        usShipSpeed = htonl(usShipSpeed); // 0.1m/s

        nLat = htonl(nLat); // 1e-6 deg

        nLon = htonl(nLon); // 1e-6 deg

        usHeight = htonl(usHeight); // 0.1m

        usResv1 = htonl(usResv1);

        unStartAbsTime = htonl(unStartAbsTime); // 扇区前沿绝对时间，单位：1ms
        unStopAbsTime =  htonl(unStopAbsTime); // 扇区后沿绝对时间，单位：1ms

//        unsigned char usAttenScanVel; // 0.1deg
//        unsigned char ucAttenScanFlag; // 0 环扫; 1 扇扫
//        unsigned char ucAttenScanDir; // 0 顺扫; 1 逆扫
//        unsigned char usAttenScanCenter; // 360/65536

//        unsigned char usAttenScanWide; // 360/65536

        if(usPlotNum > XSJC_MAXPTSONESEC)
        {
#ifdef DEBUG
            std::cout << "usPlotNum = " << usPlotNum << std::endl;
#endif // DEBUG
            usPlotNum = 0;
        }

//        tagDetectSector stPlot[MAX_PLOT_NUM_IN_SECTOR]; // 扇区内nDotNum个点的结构
        for(unsigned short i = 0; i < usPlotNum; i++)
        {
            stPlot[i].SwapBytesToNet();
        }
        usPlotNum = htons(usPlotNum);  // 当前扇区内点迹数
    }

    void SwapBytesToHost(void) // 网络 -> 主机
    {
//        unsigned char dotSrc; // 0: 6a; 1-255: 19
//        unsigned char   ucSectorNo; // 扇区号
        usPlotNum = ntohs(usPlotNum);  // 当前扇区内点迹数

        unWorkMode = ntohl(unWorkMode); // (0,1,2,3)

        usShipCourse = ntohl(usShipCourse); // 360/65536

        usShipSpeed = ntohl(usShipSpeed); // 0.1m/s

        nLat = ntohl(nLat); // 1e-6 deg

        nLon = ntohl(nLon); // 1e-6 deg

        usHeight = ntohl(usHeight); // 0.1m

        usResv1 = ntohl(usResv1);

        unStartAbsTime = ntohl(unStartAbsTime); // 扇区前沿绝对时间，单位：1ms
        unStopAbsTime =  ntohl(unStopAbsTime); // 扇区后沿绝对时间，单位：1ms

//        unsigned char usAttenScanVel; // 0.1deg
//        unsigned char ucAttenScanFlag; // 0 环扫; 1 扇扫
//        unsigned char ucAttenScanDir; // 0 顺扫; 1 逆扫
//        unsigned char usAttenScanCenter; // 360/65536

//        unsigned char usAttenScanWide; // 360/65536

        // 涉及到数组，只检查点迹数，其余关键字段在外部检查 *****
        if(usPlotNum <= XSJC_MAXPTSONESEC)
        {
            for(unsigned short i = 0; i < usPlotNum; i++)
            {
                stPlot[i].SwapBytesToHost();
            }
        }
        else
        {
            std::cout << "Rev plots num overflow" << std::endl;
        }
    }

    unsigned int Bytes(void) // 注意此函数只有在转到本地字节序后才可调用
    {
        if(usPlotNum <= XSJC_MAXPTSONESEC)
        {
            return (sizeof(XSJCDetectSectorBody)-(XSJC_MAXPTSONESEC-usPlotNum)*sizeof(XSJCDetectPlot));
        }
        else
        {
            return 0xFFFFF;
        }
    }
}XSJCDetectSectorBody;

// 本站扇区点迹
typedef struct XSJCDetectSector
{
    XSJCMsgHead head;
    XSJCDetectSectorBody body;
    XSJCMsgEnd end;

    XSJCDetectSector(void)
    {
        head.usLen = sizeof(*this);
        head.ucCat = XSJCTAG_SECPLOTS;
    }

    void SwapBytesToNet(void) // 主机 -> 网络
    {
        head.SwapBytesToNet();
        body.SwapBytesToNet();
    }

    void SwapBytesToHost(void) // 网络 -> 主机
    {
        head.SwapBytesToHost();
        body.SwapBytesToHost();
    }

    unsigned int Bytes(void) // 注意此函数只有在转到本地字节序后才可调用
    {
        return (sizeof(XSJCMsgHead) + body.Bytes() + sizeof(XSJCMsgEnd));
    }
}XSJCDetectSector;

typedef struct XSJCPlotXX9
{
    double dAzi;
    double dDis;
    double dAmp;
    double dDisSpan;
    double dAziSpan;
    double dTime;
    double dAziLft;
    double dAziRgt;
    double dDizTop;
    double dDizBtm;

public:
    XSJCPlotXX9() {
        memset(this, 0, sizeof(XSJCPlotXX9));
    }
}XSJCPlotXX9;

typedef struct XSJCSecPlotXX9
{
    unsigned short wUDP2SerialHead; // 网络转串口报头
    unsigned char bySendNode; // 发方节点号
    unsigned char byRecvNode; // 收方节点号
    unsigned short wDragLength; // 报文总长度
    unsigned char byDragCATFlag; // 报文识别符
    unsigned char byInSureMark; // 应答标志
    int nPreSectorNum; // 上一个扇区号
    int nSectorNum; // 当前扇区号
    int nDotNum; // 当前扇区点迹数
    int nPreDotNum; // 上一个扇区点迹数
    int nFanDirection; // 天线扫描方向
    XSJCPlotXX9 stDot[XSJC_MAXPTSONESEC]; // 点迹结构体

public:
    XSJCSecPlotXX9() {
        memset(this, 0, sizeof(XSJCSecPlotXX9));
    }
}XSJCSecPlotXX9;
}

namespace msgSPx {
// SPxSimulator视频包网络传输头24字节
typedef struct SPxVideoNetTransHeader
{
    /* Bytes 0 to 3 */
    unsigned int Magic;		// Constant magic number for integrity checks, 0x5350584E
    /* Bytes 4 to 7 */
    unsigned char ProtocolVersion;	// Protocol version number
    unsigned char HeaderSize;		// Size of header
    unsigned char NumChunks;        // The number of packets.
    unsigned char ChunkId;          // The ID of this chunk (0 ..NumChunks ? 1).
    /* Bytes 8 to 11 */
    unsigned short SequenceNumber;  // Packet sequence numbers (wraps around after 65536).
    unsigned short PayloadSize;     // The number of data bytes in this chunk.
    /* Bytes 12 to 15 */
    unsigned int TotalPayloadSize;  // Total size of all payloads.
    /* Bytes 16 to 19 */
    unsigned int PayloadOffset;     // Position of this payload in message.
    /* Bytes 20 to 23 */
    unsigned int SourceIdentifier;  // Server-assigned source identifier.
    SPxVideoNetTransHeader()
    {
        memset(this, 0, sizeof(*this));
        Magic = 0x5350584E;
        HeaderSize = 0x18;
    }
    void bytesHost2Net(void) // 主机 -> 网络
    {
        Magic = htonl(Magic);
        SequenceNumber = htons(SequenceNumber);
        PayloadSize = htons(PayloadSize);
        TotalPayloadSize = htonl(TotalPayloadSize);
        PayloadOffset = htonl(PayloadOffset);
        SourceIdentifier = htonl(SourceIdentifier);
    }
    void bytesNet2Host(void) // 网络 -> 主机
    {
        Magic = ntohl(Magic);
        SequenceNumber = ntohs(SequenceNumber);
        PayloadSize = ntohs(PayloadSize);
        TotalPayloadSize = ntohl(TotalPayloadSize);
        PayloadOffset = ntohl(PayloadOffset);
        SourceIdentifier = ntohl(SourceIdentifier);
    }
} SPxVideoNetTransHeader;

// SPx视频报文格式48字节
typedef struct SPxReturnHeader
{
    /* Bytes 0 to 3 */
    unsigned int magic1;		// Constant magic number for integrity checks, 0xC0DE5837
    /* Bytes 4 to 7 */
    unsigned short protocolVersion;	// Protocol version number
    unsigned char reserved06;
    unsigned char headerSize;		// Size of header.
    /* Bytes 8 to 11 */
    unsigned short radarVideoSize;	// Size of encoded radar video data.
    unsigned char numTriggers;		// Number of triggers represented by this data
    unsigned char sourceType;		// Source where this return came from
    /* Bytes 12 to 15 */
    unsigned char reserved12;
    unsigned char sourceCode;		// Source-specific code.
    unsigned short count;		// Incrementing count from source
    /* Bytes 16 to 19 */
    unsigned short nominalLength;	// Nominal length of the return.
    unsigned short thisLength;		// Length of this return (<= nominal length)
    /* Bytes 20 to 23 */
    unsigned short azimuth;		// The azimuth (0..65536)
    unsigned char packing;		// Packing type
    unsigned char scanMode;		// One of the SPX_SCAN_MODE_... modes.
    /* Bytes 24 to 27 */
    unsigned int totalSize;		// headerSize + radarVideoSize + extraBytes
    /* Bytes 28 to 31 */
    unsigned short heading;		// Platform heading (0..65536 = 0..360degs)
    unsigned short reserved30;
    /* Bytes 32 to 35 */
    unsigned short timeInterval;	// Interval between returns (microseconds)
    unsigned char pimFlags;		// Bitwise combination of SPX_PIM_FLAGS_...
    unsigned char dataFlags;            // Bitwise combination of SPX_RIB_DATA_FLAGS_...
    /* Bytes 36 to 39 */
    unsigned int startRange;		// Start range in world units for first sample 需手动转为float
    /* Bytes 40 to 43 */
    unsigned int endRange;		// End range in world units for nominalLength. 需手动转为float
    /* Bytes 44 to 47 */
    unsigned int magic2;		// Magic number for integrity checking
    SPxReturnHeader()
    {
        memset(this, 0, sizeof(*this));
        magic1 = 0xC0DE5837;
        magic2 = 0xC0DE6948;
        headerSize = 0x30;
    }
    void bytesHost2Net(void) // 主机 -> 网络
    {
        magic1 = htonl(magic1);
        protocolVersion = htons(protocolVersion);
        radarVideoSize = htons(radarVideoSize);
        count = htons(count);
        nominalLength = htons(nominalLength);
        thisLength = htons(thisLength);
        azimuth = htons(azimuth);
        totalSize = htonl(totalSize);
        heading = htons(heading);
        reserved30 = htons(reserved30);
        timeInterval = htons(timeInterval);
        startRange = htonl(startRange);
        endRange = htonl(endRange);
        magic2 = htonl(magic2);
    }
    void bytesNet2Host(void) // 网络 -> 主机
    {
        magic1 = ntohl(magic1);
        protocolVersion = ntohs(protocolVersion);
        radarVideoSize = ntohs(radarVideoSize);
        count = ntohs(count);
        nominalLength = ntohs(nominalLength);
        thisLength = ntohs(thisLength);
        azimuth = ntohs(azimuth);
        totalSize = ntohl(totalSize);
        heading = ntohs(heading);
        reserved30 = ntohs(reserved30);
        timeInterval = ntohs(timeInterval);
        startRange = ntohl(startRange);
        endRange = ntohl(endRange);
        magic2 = ntohl(magic2);
    }
} SPxReturnHeader;

// 第一包网络视频
typedef struct SPxFirstReturn
{
    SPxVideoNetTransHeader NetHeader;
    SPxReturnHeader ReturnHeader;
    unsigned char ReturnData[SPx_MaxSampleCellNum];       //回波数据
    SPxFirstReturn()
    {
         memset(ReturnData, 0, sizeof(unsigned char)*SPx_MaxSampleCellNum);
    }
    void bytesHost2Net(void) // 主机 -> 网络
    {
        NetHeader.bytesHost2Net();
        ReturnHeader.bytesHost2Net();
    }
    void bytesNet2Host(void) // 网络 -> 主机
    {
        NetHeader.bytesNet2Host();
        ReturnHeader.bytesNet2Host();
    }
    int bytes(void) // 注意此函数只有在转到本地字节序后才可调用
    {
        return (sizeof(NetHeader) + sizeof(unsigned char)*NetHeader.PayloadSize);
    }
}SPxFirstReturn;

// 第2...包网络视频
typedef struct SPxNextReturn
{
    SPxVideoNetTransHeader NetHeader;
    unsigned char ReturnData[SPx_MaxSampleCellNum];       //回波数据
    SPxNextReturn()
    {
         memset(ReturnData, 0, sizeof(unsigned char)*SPx_MaxSampleCellNum);
    }
    void bytesHost2Net(void) // 主机 -> 网络
    {
        NetHeader.bytesHost2Net();
    }
    void bytesNet2Host(void) // 网络 -> 主机
    {
        NetHeader.bytesNet2Host();
    }
    int bytes(void) // 注意此函数只有在转到本地字节序后才可调用
    {
        return (sizeof(NetHeader) + sizeof(unsigned char)*NetHeader.PayloadSize);
    }
}SPxNextReturn;
}

namespace NRxRecordMessage
{
typedef struct NRxRecordMsgHead // 32 字节
{
    unsigned int head; // 0 报文头 0x1A3B5C7D
    unsigned int time0; // 4 时统时间0: b31-24 预留; b23-16 时; b15-8 分; b7-0 秒.

    unsigned int time1; // 8 时统时间1: 纳秒, 单位：1ns.
    unsigned char sendNode; // 12 发方节点号
    unsigned char recvNode; // 13 收方节点号
    unsigned short msgLen; // 14 报文长度

    unsigned short msgType; // 16 报文识别符, 内部自定报文使用高位(> 256), 低位作为对外接口时使用
    unsigned char compressFlag; // 18 压缩标记, 定义如下, 默认: 原始数据; 未压缩; 网络字节序
    // b0 data format 0, 原始数据; 1, 压缩数据.
    // b1-b4 data compress format 0000, 数据未压缩; 0001, 数据使用 qt sdk 压缩
    // b5 bytes order 0, 网络字节序; 1, 本地字节序
    unsigned short counter; // 19 计数器, 各类报文独自编号
    unsigned char stationNum; // 21 站点号, 用于雷达站点标志
    unsigned char systemNum; // 22 系统号
    unsigned char equipNum; // 23 设备号

    unsigned char res1[8]; // 24 备用

    NRxRecordMsgHead(void)
    {
        memset(this, 0, sizeof(*this));
        head = 0x1A3B5C7D;
    }
    void bytesHost2Net(void) // 主机 -> 网络
    {
        head = htonl(head); // 0 报文头
        time0 = htonl(time0); // 4 时统时间0
        time1 = htonl(time1); // 8 时统时间1
        msgLen = htons(msgLen); // 14 报文长度
        msgType = htons(msgType); // 16 报文识别符
        counter = htons(counter); // 19 计数器
    }
    void bytesNet2Host(void) // 网络 -> 主机
    {
        head = ntohl(head); // 0 报文头
        time0 = ntohl(time0); // 4 时统时间0
        time1 = ntohl(time1); // 8 时统时间1
        msgLen = ntohs(msgLen); // 14 报文长度
        msgType = ntohs(msgType); // 16 报文识别符
        counter = ntohs(counter); // 19 计数器
    }
}NRxRecordMsgHead;
typedef struct NRxRecordMsgEnd // 8 字节
{
    unsigned int CRC; // 0 校验码
    unsigned int end; // 4 信息尾 0xAAAABBBB

    NRxRecordMsgEnd(void)
    {
        CRC = 0;
        end = 0xAAAABBBB;
    }
    void bytesHost2Net(void) // 主机 -> 网络
    {
        CRC = htonl(CRC);
        end = htonl(end);
    }
    void bytesNet2Host(void) // 网络 -> 主机
    {
        CRC = ntohl(CRC);
        end = ntohl(end);
    }
    bool check() const
    {
        return (CRC == 0 && end == 0xAAAABBBB);
    }
}NRxRecordMsgEnd;
// 64 位整数表示的 ms
typedef struct tms64
{
    unsigned int tHigh; // 0 64位整数的高32位
    unsigned int tLow; // 4 64位整数的低32位

    tms64(void)
        : tHigh(0)
        , tLow(0)
    {
    }
    tms64(unsigned int high, unsigned int low)
        : tHigh(high)
        , tLow(low)
    {
    }
    void bytesHost2Net(void) // 主机 -> 网络
    {
        tHigh = htonl(tHigh); // 0 64位整数的高32位
        tLow = htonl(tLow); // 4 64位整数的低32位
    }
    void bytesNet2Host(void) // 网络 -> 主机
    {
        tHigh = ntohl(tHigh); // 0 64位整数的高32位
        tLow = ntohl(tLow); // 4 64位整数的低32位
    }
    unsigned long long toUint64() const
    {
        return ( ((unsigned long long)(tHigh) << 32) + tLow );
    }
    double toDouble() const
    {
        return double( ((unsigned long long)(tHigh) << 32) + tLow );
    }
    void fromDouble(double t)
    {
        const unsigned long long maxT32 = 0xFFFFFFFF;
        unsigned long long t64 = (unsigned long long)(t);
        tLow = (t64 & maxT32);
        tHigh = ((t64 >> 32) & maxT32);
    }
    void addMs(int ms)
    {
        const unsigned long long maxT32 = 0xFFFFFFFF;
        unsigned long long t = this->toUint64();
        t += ms;
        tLow = (t & maxT32);
        tHigh = ((t >> 32) & maxT32);
    }
    friend long long operator-(const tms64& t1_64, const tms64& t2_64)
    {
        unsigned long long t1 = t1_64.toUint64();
        unsigned long long t2 = t2_64.toUint64();
        if(t1 >= t2)
        {
            return (long long)(t1 - t2);
        }
        else
        {
            return -(long long)(t2 - t1);
        }
    }
    friend std::ostream& operator<<(std::ostream& out, const tms64& t64)
    {
        unsigned long long t = t64.toUint64();
        out << "t = " << t << std::endl;
        return out;
    }
}tms64;
// 视频数据格式
typedef struct NRxRecordMsgVideoHead   // 72字节+变长
{
    tms64 refTime;         // 0 相对时间, 单位: 1ms

    unsigned int absTime;  // 8 绝对时间, 单位: 1ms
    unsigned int res0;     // 12 预留（用于1970时间的微秒数）

    unsigned short azi;    // 16 方位, 单位: 360.f / 65536.f
    unsigned short PRT;    // 18 重复周期
    unsigned int sampleCellsNum;    // 20 当前脉冲采样单元数

    unsigned short sampleCellSize;  // 24 采样距离，LSB：0.01米
    unsigned char dataFlag ;        // 26 数据标识 b7-5视频幅度量纲. 0dB; 1线性映射; 2约定非线性映射.
    unsigned char mapPreLowerDB;    // 27 线性映射参数0，线性映射时有效. 映射前幅度下限
    unsigned char mapPreUpperDB;    // 28 线性映射参数0，线性映射时有效. 映射前幅度上限
    // val = 0 (if DB<= mapPreLowerDB)
    // val = 2^n - 1 (if DB>= mapPreUpperDB)
    // val = (DB-mapPreLowerDB) / (mapPreUpperDB-mapPreLowerDB) * (2^n - 1)
    unsigned char res1;   // 29 预留
    unsigned short res2;  // 30 备用

    int longitude;        // 32 导航信息，经度，LSB：1/10000分
    int latitude;         // 36 纬度，LSB：1/10000分

    int altitude;         // 40 高度，LSB：1米
    unsigned short absPlatformCourse ;   // 44 绝对航向，LSB：360/65536，0xFFFF表示绝对航向和绝对航速无效
    unsigned short absPlatformVelocity;  // 46 绝对航速，LSB：0.1m/s

    unsigned short relPlatformCourse;    // 48 相对航向，LSB：360/65536，0xFFFF表示相对航向和相对航速无效
    unsigned short relPlatformVelocity;  // 50 相对航速，LSB：0.1m/s
    short headSway;         // 52 首摇；单位：360度/32768  默认值0
    short rollSway;         // 54 横摇；单位：360度/32768  默认值0

    short pitchSway;        // 56 纵摇；单位：360度/32768  默认值0
    unsigned char scanType; // 58 扫描方式
    // b7: 0,固定平台;1,移动平台. b6-5预留
    // b4-0: 0顺时针环扫, 1逆时针环扫, 2顺时针机械扇扫, 3逆时针机械扇扫,4顺时针单向电扫, 5逆时针单向电扫, 6随机扫描, 7定位, 8停车首, 9手轮.
    unsigned char res3;     // 59 备用
    unsigned int res4;      // 60 备用

    unsigned int res5;      // 64 备用
    unsigned int res6;      // 68 备用

//    unsigned char MTPPacketData[gMaxSampleCellNum];   // 72 视频数据

    NRxRecordMsgVideoHead(void)
    {
        memset(this, 0, sizeof(*this));
    }
    void bytesHost2Net(void) // 主机 -> 网络
    {
        refTime.bytesHost2Net();         // 0 相对时间, 单位: 1ms
        absTime = htonl(absTime);  // 8 绝对时间, 单位: 1ms
        res0 = htonl(res0);        // 12 预留（用于1970时间的微秒数）
        azi = htons(azi);    // 16 方位, 单位: 360.f / 65536.f
        PRT = htons(PRT);    // 18 重复周期
        sampleCellsNum = htonl(sampleCellsNum);    // 20 当前脉冲采样单元数
        sampleCellSize = htons(sampleCellSize);    // 24 采样距离，LSB：0.01米
        res2 = htons(res2);   // 30 备用
        longitude = htonl(longitude);        // 32 导航信息，经度，LSB：1/10000分
        latitude = htonl(latitude);          // 36 纬度，LSB：1/10000分
        altitude = htonl(altitude);          // 40 高度，LSB：1米
        absPlatformCourse = htons(absPlatformCourse);   // 44 绝对航向，LSB：360/65536，0xFFFF表示绝对航向和绝对航速无效
        absPlatformVelocity = htons(absPlatformVelocity);  // 46 绝对航速，LSB：0.1m/s
        relPlatformCourse = htons(relPlatformCourse);      // 48 相对航向，LSB：360/65536，0xFFFF表示相对航向和相对航速无效
        relPlatformVelocity = htons(relPlatformVelocity);  // 50 相对航速，LSB：0.1m/s
        headSway = htons(headSway);                        // 52 首摇；单位：360度/32768  默认值0
        rollSway = htons(rollSway);                        // 54 横摇；单位：360度/32768  默认值0
        pitchSway = htons(pitchSway);        // 56 纵摇；单位：360度/32768  默认值0
        res4 = htonl(res4);       // 60 备用
        res5 = htonl(res5);       // 64 备用
        res6 = htonl(res6);       // 68 备用
    }
    void bytesNet2Host(void) // 网络 -> 主机
    {
        refTime.bytesNet2Host();         // 0 相对时间, 单位: 1ms
        absTime = ntohl(absTime);  // 8 绝对时间, 单位: 1ms
        res0 = ntohl(res0);        // 12 预留（用于1970时间的微秒数）
        azi = ntohs(azi);    // 16 方位, 单位: 360.f / 65536.f
        PRT = ntohs(PRT);    // 18 重复周期
        sampleCellsNum = ntohl(sampleCellsNum);    // 20 当前脉冲采样单元数
        sampleCellSize = ntohs(sampleCellSize);    // 24 采样距离，LSB：0.01米
        res2 = ntohs(res2);   // 30 备用
        longitude = ntohl(longitude);        // 32 导航信息，经度，LSB：1/10000分
        latitude = ntohl(latitude);          // 36 纬度，LSB：1/10000分
        altitude = ntohl(altitude);          // 40 高度，LSB：1米
        absPlatformCourse = ntohs(absPlatformCourse);   // 44 绝对航向，LSB：360/65536，0xFFFF表示绝对航向和绝对航速无效
        absPlatformVelocity = ntohs(absPlatformVelocity);  // 46 绝对航速，LSB：0.1m/s
        relPlatformCourse = ntohs(relPlatformCourse);      // 48 相对航向，LSB：360/65536，0xFFFF表示相对航向和相对航速无效
        relPlatformVelocity = ntohs(relPlatformVelocity);  // 50 相对航速，LSB：0.1m/s
        headSway = ntohs(headSway);                        // 52 首摇；单位：360度/32768  默认值0
        rollSway = ntohs(rollSway);                        // 54 横摇；单位：360度/32768  默认值0
        pitchSway = ntohs(pitchSway);        // 56 纵摇；单位：360度/32768  默认值0
        res4 = ntohl(res4);       // 60 备用
        res5 = ntohl(res5);       // 64 备用
        res6 = ntohl(res6);       // 68 备用
    }
    int bytes()
    {
        return sizeof(*this);
    }
}NRxRecordMsgVideoHead;

// 原始视频网络报文
typedef struct NRxRecordVideo
{
    NRxRecordMsgHead msghead;
    NRxRecordMsgVideoHead videohead;
    // NRxRecordMsgEnd end;

    void bytesHost2Net(void) // 主机 -> 网络
    {
        msghead.bytesHost2Net();
        videohead.bytesHost2Net();
        // end.bytesHost2Net();
    }
    void bytesNet2Host(void) // 网络 -> 主机
    {
        msghead.bytesNet2Host();
        videohead.bytesNet2Host();
        // end.bytesNet2Host();
    }
    int bytes() // 注意此函数只有在转到本地字节序后才可调用
    {
        return sizeof(NRxRecordMsgHead) + videohead.bytes();
    }
}NRxRecordVideo;
}

// 导航 CZ2102 20221203
namespace NaviIf {

#define NaviIfTag_VidDtc (0x300) // 检测后视频
#define NaviIfTAGDET_THR (0x1010) // 检测门限
#define NaviIfTAGDET_LOG (0x1020) // 检测记录 tag
#define NaviIfTAG_HEAD (0x1A3B5C7D) // 报文通用头

typedef struct NaviHead    //32字节
{
    unsigned int msgHead; // 0x1A3B5C7D
    unsigned int time; // 时统时间  b31-24 预留;b23-16 时;b15-8 分;b7-0 秒
    unsigned int radarTime; // 雷达时间  纳秒  LSB：1ns
    unsigned char srcNode; // 发方节点号
    unsigned char dstNode; // 收方节点号
    unsigned short msgLen; // 报文长度
    unsigned short msgID; // 报文标识  内部自定报文使用高位8位;低位作为对外接口使用;0x2000 雷达控制命令 0x3000 雷达控制命令  0x4000 信号处理控制命令  //dx20220810
    unsigned char compFlag; // 压缩标记  b0 : dataformat: 0 未压缩; 1压缩;b1 - b4 : data compress format :0000 未压缩; 0001 数据使用 qtsdk压缩;b5 byte order : 0 网络字节序; 1 本地字节序
    unsigned short msgCnt; // 报文计数
    unsigned char siteNum; // 站点号
    unsigned char sysNum; // 系统号
    unsigned char deviceNum; // 设备号
    unsigned char res[8]; //预留

    NaviHead() {
        memset(this, 0, sizeof(*this));
    }
    void bytesHost2Net(void) {
        msgHead = htonl(msgHead); // 0 报文头
        time = htonl(time); // 4 时统时间0
        radarTime = htonl(radarTime); // 8 时统时间1
        // unsigned char sendNode; // 12 发方节点号
        // unsigned char recvNode; // 13 收方节点号
        msgLen = htons(msgLen); // 14 报文长度
        msgID = htons(msgID); // 16 报文识别符
        // unsigned char compressFlag; // 18 压缩标记
        msgCnt = htons(msgCnt); // 19 计数器
        // unsigned char stationNum; // 21 站点号, 用于雷达站点标志
        // unsigned char systemNum; // 22 系统号
        // unsigned char equipNum; // 23 设备号
        // unsigned char res1[8]; // 24 备用
    }
    void bytesNet2Host(void) {
        msgHead = ntohl(msgHead); // 0 报文头
        time = ntohl(time); // 4 时统时间0
        radarTime = ntohl(radarTime); // 8 时统时间1
        // unsigned char sendNode; // 12 发方节点号
        // unsigned char recvNode; // 13 收方节点号
        msgLen = ntohs(msgLen); // 14 报文长度
        msgID = ntohs(msgID); // 16 报文识别符
        // unsigned char compressFlag; // 18 压缩标记
        msgCnt = ntohs(msgCnt); // 19 计数器
        // unsigned char stationNum; // 21 站点号, 用于雷达站点标志
        // unsigned char systemNum; // 22 系统号
        // unsigned char equipNum; // 23 设备号
        // unsigned char res1[8]; // 24 备用
    }
}NaviHead;

typedef struct NaviEnd // 8 字节
{
    unsigned int CRC; // 0 校验码
    unsigned int end; // 4 信息尾 0xAAAABBBB

    NaviEnd() {
        memset(this,0,sizeof(*this));
        CRC = 0;
        end = 0xAAAABBBB;
    }
    void bytesNet2Host() {
        CRC = ntohl(CRC);
        end = ntohl(end);
    }
}NaviEnd;

// RH03A 惯导结构体
typedef struct NaviINSInfo {
    unsigned char Sync1;        	 // 同步字 0xAA
    unsigned char Sync2;			 // 同步字 0x55
    unsigned char MessageID;       // 消息ID
    unsigned char SourceID;		 // 发送方ID

    unsigned short Sequence;		 // 发送序列号
    unsigned short MassageLength;  // 消息字节长度

    long long  Lon;			 // 经度 东经为正 lsb = 0.000000001°

    long long  Lat;			 // 纬度 东经为正 lsb = 0.000000001°

    int Height;			 // 海拔高度 lsb = 0.001米

    unsigned char FixType;		 // BD定位状态 (0: 定位无效; 1: 单点定位; 2: 伪差分定位; 3: 双频定位; 4: 载波相位差分定位; 6: 北斗B3定位)
    unsigned char Sats;			 // BD卫星数
    unsigned char HeadingValid;    // BD定向状态(4: 定向有效; 0: 定向无效)
    unsigned char HeadingSats;	 // 定向卫星数

    unsigned short Heading;		 // BD定向角度 lsb = 0.01°

    unsigned int Year;            //年
    unsigned char month; //月
    unsigned char day; //日
    unsigned char hour; //时
    unsigned char min; //分

    unsigned int ms;//毫秒

    short Roll;			 // 滚转角 lsb = 0.01° range = ±180°
    short Pitch;			 // 俯仰角 lsb = 0.01° range = ±180°

    short Yaw;			 // 航向角 lsb = 0.01° range = ±180°

    int PosN;			 // 北向相对位置(0.01m)

    int PosE;			 // 东向相对位置(0.01m)

    short PosD;			 // 相对高度(向下为正, lsb = 0.01m)
    short VelN;			 // 北向速度(0.001m/s)

    short VelE;			 // 东向速度(0.001m/s)
    short VelD;			 // 垂直速度(向下为正, lsb = 0.001m/s)

    unsigned int crc;
}NaviINSInfo;

typedef struct NaviINSInfoCZ2102 {
    unsigned char Sync1;        	 // 同步字 0xAA
    unsigned char Sync2;			 // 同步字 0x55
    unsigned char MessageID;       // 消息ID
    unsigned char SourceID;		 // 发送方ID
    unsigned short Sequence;		 // 发送序列号
    unsigned short MassageLength;  // 消息字节长度

    long long  Lon;			 // 经度 东经为正 lsb = 0.000000001°

    long long  Lat;			 // 纬度 东经为正 lsb = 0.000000001°

    int Height;			 // 海拔高度 lsb = 0.001米
    unsigned char FixType;		 // BD定位状态 (0: 定位无效; 1: 单点定位; 2: 伪差分定位; 3: 双频定位; 4: 载波相位差分定位; 6: 北斗B3定位)
    unsigned char Sats;			 // BD卫星数
    unsigned char HeadingValid;    // BD定向状态(4: 定向有效; 0: 定向无效)
    unsigned char HeadingSats;	 // 定向卫星数

    unsigned short Heading;		 // BD定向角度 lsb = 0.01°
    unsigned short GpsWeek;		 // Gps周数
    unsigned int GpsMs;			 // Gps周内ms数

    short Roll;			 // 滚转角 lsb = 0.01° range = ±180°
    short Pitch;			 // 俯仰角 lsb = 0.01° range = ±180°
    short Yaw;			 // 航向角 lsb = 0.01° range = ±180°
    short RollRate;		 // 滚转角速度 lsb = 0.001°/s

    short PitchRate;		 // 俯仰角速度 lsb = 0.001°/s
    short YawRate;		 // 航向角速度 lsb = 0.001°/s
    int PosN;			 // 北向相对位置(0.01m)

    int PosE;			 // 东向相对位置(0.01m)
    short PosD;			 // 相对高度(向下为正, lsb = 0.01m)
    short VelN;			 // 北向速度(0.001m/s)

    short VelE;			 // 东向速度(0.001m/s)
    short VelD;			 // 垂直速度(向下为正, lsb = 0.001m/s)
    unsigned int crc;

    NaviINSInfoCZ2102(void) {
        memset(this,0,sizeof(NaviINSInfo));
        Sync1 = 0xAA;        	 // 同步字 0xAA
        Sync2 = 0x55;			 // 同步字 0x55
        MessageID = 0;       // 消息ID
        SourceID = 0;		 // 发送方ID
    //    Sequence;		 // 发送序列号
        MassageLength=sizeof(NaviINSInfo);  // 消息字节长度
    }

    void NetToHostEndian(const NaviINSInfoCZ2102& src) {
        Sync1      =src. Sync1;        	 // 同步字 0xAA
        Sync2	    =src. Sync2;			 // 同步字 0x55
        MessageID  =src. MessageID;       // 消息ID
        SourceID	=src. SourceID;		 // 发送方ID
        Sequence=ntohs(src.Sequence);		 // 发送序列号
        MassageLength=ntohs(src.MassageLength);  // 消息字节长度

        Lon=htonll(src.Lon);			 // 经度 东经为正 lsb = 0.000000001°

        Lat=htonll(src.Lon);			 // 纬度 东经为正 lsb = 0.000000001°

        Height=ntohl(src.Height);		 // 海拔高度 lsb = 0.001米

        FixType		=src.FixType;		 // BD定位状态 (0: 定位无效; 1: 单点定位; 2: 伪差分定位; 3: 双频定位; 4: 载波相位差分定位; 6: 北斗B3定位)
        Sats		=src.Sats;			 // BD卫星数
        HeadingValid=src.HeadingValid;    // BD定向状态(4: 定向有效; 0: 定向无效)
        HeadingSats	=src.HeadingSats;	 // 定向卫星数

        Heading=ntohs(src.Heading);		 // BD定向角度 lsb = 0.01°
        GpsWeek=ntohs(src.GpsWeek);		 // Gps周数
        GpsMs=ntohl(src.GpsMs);     	 // Gps周内ms数

        Roll	 =ntohs(src.Roll); 			 // 滚转角 lsb = 0.01° range = ±180°
        Pitch	 =ntohs(src.Pitch); 			 // 俯仰角 lsb = 0.01° range = ±180°
        Yaw	 =ntohs(src.Yaw); 			 // 航向角 lsb = 0.01° range = ±180°
        RollRate=ntohs(src.RollRate);		 // 滚转角速度 lsb = 0.001°/s

        PitchRate	 =ntohs(src.PitchRate)	;  	 // 俯仰角速度 lsb = 0.001°/s
        YawRate	 =ntohs(src.YawRate	); 	 // 航向角速度 lsb = 0.001°/s
        PosN	 =ntohl(src.PosN	);			 // 北向相对位置(0.01m)

        PosE	 =ntohl(src.PosE	);		 // 东向相对位置(0.01m)
        PosD	 =ntohs(src.PosD	); 		 // 相对高度(向下为正, lsb = 0.01m)
        VelN	 =ntohs(src.VelN	); 		 // 北向速度(0.001m/s)

        VelE	 =ntohs(src.VelE	);			 // 东向速度(0.001m/s)
        VelD	 =ntohs(src.VelD	);			 // 垂直速度(向下为正, lsb = 0.001m/s)
        crc	 =ntohl(src.crc	);
    }

    void HostToNetEndian(const NaviINSInfoCZ2102& src) {
        Sync1      =src. Sync1;        	 // 同步字 0xAA
        Sync2	    =src. Sync2;			 // 同步字 0x55
        MessageID  =src. MessageID;       // 消息ID
        SourceID	=src. SourceID;		 // 发送方ID
        Sequence=htons(src.Sequence);		 // 发送序列号
        MassageLength=htons(src.MassageLength);  // 消息字节长度
        Lon=htonll(src.Lon);			 // 经度 东经为正 lsb = 0.000000001°

        Lat=htonll(src.Lon);			 // 纬度 东经为正 lsb = 0.000000001°

        Height=htonl(src.Height);		 // 海拔高度 lsb = 0.001米

        FixType		=src.FixType;		 // BD定位状态 (0: 定位无效; 1: 单点定位; 2: 伪差分定位; 3: 双频定位; 4: 载波相位差分定位; 6: 北斗B3定位)
        Sats		=src.Sats;			 // BD卫星数
        HeadingValid=src.HeadingValid;    // BD定向状态(4: 定向有效; 0: 定向无效)
        HeadingSats	=src.HeadingSats;	 // 定向卫星数

        Heading=htons(src.Heading);		 // BD定向角度 lsb = 0.01°
        GpsWeek=htons(src.GpsWeek);		 // Gps周数
        GpsMs=htonl(src.GpsMs);     	 // Gps周内ms数

        Roll	 =htons(src.Roll); 			 // 滚转角 lsb = 0.01° range = ±180°
        Pitch	 =htons(src.Pitch); 			 // 俯仰角 lsb = 0.01° range = ±180°
        Yaw	 =htons(src.Yaw); 			 // 航向角 lsb = 0.01° range = ±180°
        RollRate=htons(src.RollRate);		 // 滚转角速度 lsb = 0.001°/s

        PitchRate	 =htons(src.PitchRate)	;  	 // 俯仰角速度 lsb = 0.001°/s
        YawRate	 =htons(src.YawRate	); 	 // 航向角速度 lsb = 0.001°/s
        PosN	 =htonl(src.PosN	);			 // 北向相对位置(0.01m)

        PosE	 =htonl(src.PosE	);		 // 东向相对位置(0.01m)
        PosD	 =htons(src.PosD	); 		 // 相对高度(向下为正, lsb = 0.01m)
        VelN	 =htons(src.VelN	); 		 // 北向速度(0.001m/s)

        VelE	 =htons(src.VelE	);			 // 东向速度(0.001m/s)
        VelD	 =htons(src.VelD	);			 // 垂直速度(向下为正, lsb = 0.001m/s)
        crc	 =htonl(src.crc	);
    }
}NaviINSInfoCZ2102;

// 4 字节 1 个视频值
typedef struct NaviVidHead //72字节
{
    unsigned int ulRelTime_H; // 相对时间高32 LSB:1ms
    unsigned int ulRelTime_L; // 相对时间低32 LSB:1ms

    unsigned int ulAbsTime; // 绝对时间 LSB:1ms
    unsigned int ulRES0; // 预留（用于1970时间的微妙数）

    unsigned short usAzi; // 方位 0~65535
    unsigned short usPRT; // 重复周期时间	LSB:1us
    unsigned int ulSamplUnitNum; // 采样单元数

    unsigned short ulSamplUnitSize; // 采样单大小 LSB:0.01m
    unsigned char ucDataFlag; // 数据标识 b7-5 视频幅度量纲 : 0 : db 1: 线性映射 2: 约定非线性映射
    unsigned char ucLinerMapPara_F; // 线性映射参数 映射前幅度下限（线性映射时有效）
    unsigned char ucLinerMapPara_U; // 线性映射参数 映射前幅度上限（线性映射时有效） 映射前幅度上限（线性映射时有效）val = 0 (if DB <= mapPreLowerDB)  val = 2 ^ n - 1(if DB >= mapPreUpperDB)  val = (DB - mapPreLowerDB) / (mapPreUpperDB - mapPreLowerDB) * (2 ^ n - 1)
    unsigned char ucRES1; // 预留
    unsigned short usRES2; // 预留

    unsigned int ulLon; // 经度 LSB:0.0001分 // not used
    unsigned int ulLat; // 纬度 LSB:0.0001分 // not used

    unsigned int ulHeight; // 高度 LSB:1米 // not used
    unsigned short usAbsCourse; // 绝对航向 0-65534, 0xFFFF表示绝对航向和绝对航数无效
    unsigned short usAbsSpeed; // 绝对航速 LSB:0.1m/s

    unsigned short usRelCourse; // 相对航向 0-65534, 0xFFFF表示绝对航向和绝对航数无效
    unsigned short usRelSpeed; // 相对航速 LSB:0.1m/s
    short sBowShake; // 首摇 LSB:360/32768 默认值0
    short sRolling; // 横摇 LSB:360/32768 默认值0

    short sPitching; // 纵摇 LSB:360/32768 默认值0
    unsigned char ucScanMode; // 扫描方式 b7: 0,固定平台;1,移动平台; b6-5 预留 b4 - 0: 0 顺时针环扫, 1 逆时针环扫, 2 顺时针机械扇扫, 3 逆时针机械扇扫, 4 顺时针单向电扫, 5 逆时针单向电扫, 6 随机扫描, 7 定位, 8 停车首, 9 手轮
    unsigned char ucRES3; // 预留
    unsigned short ele0; //
    unsigned short ele1; //
    unsigned int ulRES5; // 预留
    unsigned int ulRES6; // 预留
    unsigned int INSInfo[18];//惯导信息, NaviINSInfo

    NaviVidHead() {
        memset(this, 0, sizeof(*this));
    }
    void bytesNet2Host() {
        ulRelTime_H = ntohl(ulRelTime_H); // 相对时间高32 LSB:1ms
        ulRelTime_L = ntohl(ulRelTime_L); // 相对时间低32 LSB:1ms
        ulAbsTime = ntohl(ulAbsTime); // 绝对时间 LSB:1ms
        ulRES0 = ntohl(ulRES0); // 预留（用于1970时间的微妙数）
        usAzi = ntohs(usAzi); // 方位 0~65535
        usPRT = ntohs(usPRT); // 重复周期时间	LSB:1us
        ulSamplUnitNum = ntohl(ulSamplUnitNum); // 采样单元数
        ulSamplUnitSize = ntohs(ulSamplUnitSize); // 采样单大小 LSB:0.01m
        //    unsigned char ucDataFlag; // 数据标识 b7-5 视频幅度量纲 : 0 : db 1: 线性映射 2: 约定非线性映射
        //    unsigned char ucLinerMapPara_F; // 线性映射参数 映射前幅度下限（线性映射时有效）
        //    unsigned char ucLinerMapPara_U; // 线性映射参数 映射前幅度上限（线性映射时有效） 映射前幅度上限（线性映射时有效）val = 0 (if DB <= mapPreLowerDB)  val = 2 ^ n - 1(if DB >= mapPreUpperDB)  val = (DB - mapPreLowerDB) / (mapPreUpperDB - mapPreLowerDB) * (2 ^ n - 1)
        //    unsigned char ucRES1; // 预留
        //    unsigned short usRES2; // 预留

        ulLon = ntohl(ulLon); // 经度 LSB:0.0001分
        ulLat = ntohl(ulLat); // 纬度 LSB:0.0001分
        ulHeight = ntohl(ulHeight); // 高度 LSB:1米
        usAbsCourse = ntohs(usAbsCourse); // 绝对航向 0-65534, 0xFFFF表示绝对航向和绝对航数无效
        usAbsSpeed = ntohs(usAbsSpeed); // 绝对航速 LSB:0.1m/s
        usRelCourse = ntohs(usRelCourse); // 相对航向 0-65534, 0xFFFF表示绝对航向和绝对航数无效
        usRelSpeed = ntohs(usRelSpeed); // 相对航速 LSB:0.1m/s
        sBowShake = ntohs(sBowShake); // 首摇 LSB:360/32768 默认值0
        sRolling = ntohs(sRolling); // 横摇 LSB:360/32768 默认值0
        sPitching = ntohs(sPitching); // 纵摇 LSB:360/32768 默认值0
        //    unsigned char ucScanMode; // 扫描方式 b7: 0,固定平台;1,移动平台; b6-5 预留 b4 - 0: 0 顺时针环扫, 1 逆时针环扫, 2 顺时针机械扇扫, 3 逆时针机械扇扫, 4 顺时针单向电扫, 5 逆时针单向电扫, 6 随机扫描, 7 定位, 8 停车首, 9 手轮
        //    unsigned char ucRES3; // 预留
        //    unsigned int ulRES4; // 预留
        //    unsigned int ulRES5; // 预留
        //    unsigned int ulRES6; // 预留
        for(int ii =0;ii<18;ii++)
        {
            INSInfo[ii] = ntohl(INSInfo[ii]);
        }
    }
}NaviVidHead;
}

namespace RH03AIf {

#define RH03ATAG_SECPLOTS (0x12C) // 扇区点迹标志

typedef struct RH03ADtcThr
{
    NaviIf::NaviHead head;
    unsigned char thr;
    NaviIf::NaviEnd end;

public:
    RH03ADtcThr(void) {
        head.msgHead = NaviIfTAG_HEAD;
        head.msgLen = sizeof(*this);
        head.msgID = NaviIfTAGDET_THR;
    }

    void bytesNet2Host(void) {
        head.bytesNet2Host();
        end.bytesNet2Host();
    }
}RH03ADtcThr;

typedef struct RH03ADtcLogInputVid
{
    NaviIf::NaviHead head;

    unsigned char dataType; // 记录数据类型: 0 记录检测视频; 1 记录惯导数据
    unsigned char logOn; // 记录开关: 0 关; 1 开
    char res0[2];

    NaviIf::NaviEnd end;

public:
    RH03ADtcLogInputVid(void) {
        head.msgHead = NaviIfTAG_HEAD;
        head.msgLen = sizeof(*this);
        head.msgID = NaviIfTAGDET_LOG;
    }

    void bytesNet2Host(void) {
        head.bytesNet2Host();
        end.bytesNet2Host();
    }
}RH03ADtcLogInputVid;

// 64 位整数表示的 ms
struct tms64
{
    unsigned int tHigh; // 0 64位整数的高32位
    unsigned int tLow; // 4 64位整数的低32位

public:
    tms64(void)
        : tHigh(0)
        , tLow(0)
    {
    }

    tms64(unsigned int high, unsigned int low)
        : tHigh(high)
        , tLow(low)
    {
    }

    void bytesHost2Net(void) // 主机 -> 网络
    {
        tHigh = htonl(tHigh); // 0 64位整数的高32位
        tLow = htonl(tLow); // 4 64位整数的低32位
    }

    void bytesNet2Host(void) // 网络 -> 主机
    {
        tHigh = ntohl(tHigh); // 0 64位整数的高32位
        tLow = ntohl(tLow); // 4 64位整数的低32位
    }

    unsigned long long toUint64() const
    {
        return ( ((unsigned long long)(tHigh) << 32) + tLow );
    }

    double toDouble() const
    {
        return double( ((unsigned long long)(tHigh) << 32) + tLow );
    }

    void fromDouble(double t)
    {
        const unsigned long long maxT32 = 0xFFFFFFFF;
        unsigned long long t64 = (unsigned long long)(t);
        tLow = (t64 & maxT32);
        tHigh = ((t64 >> 32) & maxT32);
    }

    void addMs(int ms)
    {
        const unsigned long long maxT32 = 0xFFFFFFFF;
        unsigned long long t = this->toUint64();
        t += ms;
        tLow = (t & maxT32);
        tHigh = ((t >> 32) & maxT32);
    }
};

static const unsigned int gMaxPlotInSector = 300;

// 点迹数据格式
typedef struct RH03AIfDtcPlot // 72 字节
{
    unsigned int refTime0; // 0 相对时间 高32位, LSB: 1ms
    unsigned int refTime1; // 4 相对时间 低32位, LSB: 1ms

    unsigned int absTime; // 8 绝对时间, 单位: 1ms
    unsigned int res0; // 12 备用

    unsigned int attrValidFlag; // 16 非必填字段是否有效. 0, 无效; 1, 有效
    // b0 仰角; b1 幅度; b2 多普勒速度; b3 主通道比例; b4 多普勒通道数; b5 背景属性;
    // b6 用途属性; b7 信杂噪比; b8 饱和度; b9 点迹类型; b10 点迹质量; b11 杂波等级.
    unsigned short plotNo; // 20 点迹流水号
    unsigned short azi; // 22 方位, 单位: 360.f / 65536.f

    unsigned int dis; // 24 距离, 单位: 1m
    unsigned short ele; // 28 仰角, 单位: 360.f / 65536.f
    unsigned short amp; // 30 幅度
    unsigned short disSpan; // 32 距离展宽, 单位: 1m

    unsigned short aziSpan; // 34 方位展宽, 单位: 360.f / 65536.f
    unsigned short epNum; // 36 EP 数
    unsigned char doppChannel; // 38 多普勒速度, 单位: XX9A (速度 + 50) / 100 * 128;
    unsigned char doppRatio; // 39 主通道比例, 量化为 0-255
    unsigned char doppNum; // 40 多普勒通道数

    unsigned char background; // 41 背景属性, 0 未知; 1 噪声区; 2 海杂波区; 3 气象杂波区; 4 干扰区; 5 地杂波
    unsigned char task; // 42 用途属性, 0 未知; 1 军港; 2 民港; 3 锚区; 4 航道
    unsigned char scnr; // 43 信杂噪比, 0-255
    unsigned char sat; // 44 饱和度, 量化为 0-255
    unsigned char type; // 45 点迹类型, 0 未知; 1 距离副瓣; 2 方位副瓣
    unsigned char quality; // 46 点迹质量, 0 不合格; 1 合格
    unsigned char clutterLevel; // 47 杂波等级, 0 低; 1 高

    unsigned int disStart; // 48 点迹距离起始, 单位: 0.01m
    unsigned int disEnd; // 52 点迹距离终止, 单位: 0.01m

    unsigned int aziStart; // 56 方位距离起始, 单位: 360.f / 65536.f
    unsigned int aziEnd; // 60 方位距离终止, 单位: 360.f / 65536.f

    unsigned char res1[8]; // 64 备用

    //    unsigned short aziStart; // 56 方位距离起始, 单位: 360.f / 65536.f
    //    unsigned short aziEnd; // 58 方位距离终止, 单位: 360.f / 65536.f
    //    unsigned char res1[4]; // 60 备用
    //    unsigned char res2[8]; // 64 备用

public:
    RH03AIfDtcPlot(void)
    {
        memset(this, 0, sizeof(*this));
    }

    void bytesHost2Net(void) // 主机 -> 网络
    {
        refTime0 = htonl(refTime0); // 0 相对时间 高32位, LSB: 1ms
        refTime1 = htonl(refTime1); // 4 相对时间 低32位, LSB: 1ms
        absTime = htonl(absTime); // 8 绝对时间
        res0 = htonl(res0); // 12 备用
        attrValidFlag = htonl(attrValidFlag); // 16 非必填字段是否有效
        plotNo = htons(plotNo); // 20 点迹流水号
        azi = htons(azi); // 22 方位
        dis = htonl(dis); // 24 距离
        ele = htons(ele); // 28 仰角
        amp = htons(amp); // 30 幅度
        disSpan = htons(disSpan); // 32 距离展宽
        aziSpan = htons(aziSpan); // 34 方位展宽
        epNum = htons(epNum); // 36 EP 数
        // unsigned char doppChannel; // 38 多普勒速度
        // unsigned char doppRatio; // 39 主通道比例
        // unsigned char doppNum; // 40 多普勒通道数
        // unsigned char background; // 41 背景属性
        // unsigned char task; // 42 用途属性
        // unsigned char scnr; // 43 信杂噪比
        // unsigned char sat; // 44 饱和度
        // unsigned char type; // 45 点迹类型
        // unsigned char quality; // 46 点迹质量
        // unsigned char clutterLevel; // 47 杂波等级
        disStart = htonl(disStart); // 48 点迹距离起始
        disEnd = htonl(disEnd); // 52 点迹距离终止
        aziStart = htonl(aziStart); // 56 方位距离起始
        aziEnd = htonl(aziEnd); // 60 方位距离终止
        // unsigned char res1[8]; // 64 备用
    }

    void bytesNet2Host(void) // 网络 -> 主机
    {
        refTime0 = ntohl(refTime0); // 0 相对时间 高32位, LSB: 1ms
        refTime1 = ntohl(refTime1); // 4 相对时间 低32位, LSB: 1ms
        absTime = ntohl(absTime); // 8 绝对时间
        res0 = ntohl(res0); // 12 备用
        attrValidFlag = ntohl(attrValidFlag); // 16 非必填字段是否有效
        plotNo = ntohs(plotNo); // 20 点迹流水号
        azi = ntohs(azi); // 22 方位
        dis = ntohl(dis); // 24 距离
        ele = ntohs(ele); // 28 仰角
        amp = ntohs(amp); // 30 幅度
        disSpan = ntohs(disSpan); // 32 距离展宽
        aziSpan = ntohs(aziSpan); // 34 方位展宽
        epNum = ntohs(epNum); // 36 EP 数
        // unsigned char doppChannel; // 38 多普勒速度
        // unsigned char doppRatio; // 39 主通道比例
        // unsigned char doppNum; // 40 多普勒通道数
        // unsigned char background; // 41 背景属性
        // unsigned char task; // 42 用途属性
        // unsigned char scnr; // 43 信msgType杂噪比
        // unsigned char sat; // 44 饱和度
        // unsigned char type; // 45 点迹类型
        // unsigned char quality; // 46 点迹质量
        // unsigned char clutterLevel; // 47 杂波等级
        disStart = ntohl(disStart); // 48 点迹距离起始
        disEnd = ntohl(disEnd); // 52 点迹距离终止
        aziStart = ntohl(aziStart); // 56 方位距离起始
        aziEnd = ntohl(aziEnd); // 60 方位距离终止
        // unsigned char res1[8]; // 64 备用
    }
}RH03AIfDtcPlot;

// 88 + 48 * gMaxPlotInSector 字节
typedef struct RH03AIfDtcSectorBody
{
    unsigned char sectorNo; // 0 扇区号
    unsigned short plotNum; // 1 当前扇区内点迹数
    unsigned char sectorNum; // 3 扇区总数
    unsigned char res0[4]; // 4 备用

    unsigned int startRelTime0; // /8, 扇区前沿相对时间 高32位, LSB: 1ms
    unsigned int startRelTime1; // 12, 扇区前沿相对时间 低32位, LSB: 1ms

    unsigned int endRelTime0; // 16, 扇区后沿相对时间 高32位, LSB: 1ms
    unsigned int endRelTime1; // 22, 扇区后沿相对时间 低32位, LSB: 1ms

    unsigned int startAbsTime; // 24 扇区前沿绝对时间, 单位: 1ms
    unsigned int res1; // 28 备用

    unsigned int stopAbsTime; // 32 扇区后沿绝对时间, 单位: 1ms
    unsigned int res2; // 36 备用

    unsigned short shipAzi2Ground; // 40 对地航向 单位: 360.f / 65536.f
    unsigned short shipVel2Ground; // 42 对地航速 单位: 0.1m/s
    unsigned short shipAzi2Water; // 44 对水航向 单位: 360.f / 65536.f
    unsigned short shipVel2Water; // 46 对水航速 单位: 0.1m/s

    int longitude; // 48 经度 单位: 1 / 10000 分
    int latitude; // 52 纬度 单位: 1 / 10000 分

    short height; // 56 高度 单位: 1m
    short headSway; // 58 首摇 单位: 360.f / 65536.f
    short rollSway; // 60 横摇 单位: 360.f / 65536.f
    short pitchSway; // 62 纵摇 单位: 360.f / 65536.f

    unsigned int sigProCtrl; // 64 信号处理方式开关 b0 该字段是否有效 0 无效; 1, 有效;
    // b1-3,杂波抑制：000 脉冲积累(默认), 001 MTI, 010 MTD, 011 增强MTD, 100 无. 其他预留.
    // b4-b32:预留
    unsigned short antennaScanSpeed; // 68 天线扫描速度 单位: 0.1 deg/s
    unsigned char antennaWorkMode; // 70 天线工作方式
    // b7: 0, 固定平台;1, 移动平台. b6-5预留    b4-0: 0顺时针环扫, 1逆时针环扫, 2顺时针机械扇扫, 3逆时针机械扇扫,4顺时针单向电扫, 5逆时针单向电扫, 6随机扫描, 7定位, 8停车首, 9手轮.
    unsigned char res3; // 71 备用

    unsigned short antennaScanRange; // 72 天线扇扫范围, 单位: 360.f / 65536.f
    unsigned short antennaScanCenter; // 74 天线扇扫中心, 单位: 360.f / 65536.f
    unsigned char res4[4]; // 76 备用

    unsigned char res5[8]; // 80 备用

    RH03AIfDtcPlot plot[gMaxPlotInSector]; // 88 plotNum 个点, 变长

public:
    RH03AIfDtcSectorBody(void)
    {
        memset(this, 0, sizeof(*this));
    }

    void bytesHost2Net(void) // 主机 -> 网络
    {
        // unsigned char sectorNo; // 0 扇区号
        // unsigned char res0[5]; // 3 备用
        startRelTime0 = htonl(startRelTime0); // /8, 扇区前沿相对时间 高32位, LSB: 1ms
        startRelTime1 = htonl(startRelTime1); // 12, 扇区前沿相对时间 低32位, LSB: 1ms
        endRelTime0 = htonl(endRelTime0); // 16, 扇区后沿相对时间 高32位, LSB: 1ms
        endRelTime1 = htonl(endRelTime1); // 22, 扇区后沿相对时间 低32位, LSB: 1ms
        startAbsTime = htonl(startAbsTime); // 24 扇区前沿绝对时间
        res1 = htonl(res1); // 28 备用
        stopAbsTime = htonl(stopAbsTime); // 32 扇区后沿绝对时间
        res2 = htonl(res2); // 36 备用
        shipAzi2Ground = htons(shipAzi2Ground); // 40 对地航向
        shipVel2Ground = htons(shipVel2Ground); // 42 对地航速
        shipAzi2Water = htons(shipAzi2Water); // 44 对水航向
        shipVel2Water = htons(shipVel2Water); // 46 对水航速
        longitude = htonl(longitude); // 48 经度
        latitude = htonl(latitude); // 52 纬度
        height = htons(height); // 56 高度
        headSway = htons(headSway); // 58 首摇
        rollSway = htons(rollSway); // 60 横摇
        pitchSway = htons(pitchSway); // 62 纵摇
        sigProCtrl = htonl(sigProCtrl); // 64 信号处理方式开关
        antennaScanSpeed = htons(antennaScanSpeed); // 68 天线扫描速度
        // unsigned char antennaWorkMode; // 70 天线工作方式
        // unsigned char res1; // 71 备用
        antennaScanRange = htons(antennaScanRange); // 72 天线扇扫范围
        antennaScanCenter = htons(antennaScanCenter); // 74 天线扇扫中心
        // unsigned char res2[4]; // 76 备用
        // unsigned char res2[64]; // 80 备用

        if(plotNum <= gMaxPlotInSector) // 88 plotNum 个点, 变长
        {
            for(int plotIdx = 0; plotIdx < plotNum; ++plotIdx)
            {
                plot[plotIdx].bytesHost2Net();
            }
        }
        else
        {
            std::cout << "plots num overflow before trans2Net: " << plotNum;
            plotNum = 0;
        }

        plotNum = htons(plotNum); // 1 当前扇区内点迹数
    }

    void bytesNet2Host(void) // 网络 -> 主机
    {
        // unsigned char sectorNo; // 0 扇区号
        plotNum = ntohs(plotNum); // 1 当前扇区内点迹数
        // unsigned char res0[5]; // 3 备用
        startRelTime0 = ntohl(startRelTime0); // /8, 扇区前沿相对时间 高32位, LSB: 1ms
        startRelTime1 = ntohl(startRelTime1); // 12, 扇区前沿相对时间 低32位, LSB: 1ms
        endRelTime0 = ntohl(endRelTime0); // 16, 扇区后沿相对时间 高32位, LSB: 1ms
        endRelTime1 = ntohl(endRelTime1); // 22, 扇区后沿相对时间 低32位, LSB: 1ms
        startAbsTime = ntohl(startAbsTime); // 24 扇区前沿绝对时间
        res1 = ntohl(res1); // 28 备用
        stopAbsTime = ntohl(stopAbsTime); // 34 扇区后沿绝对时间
        res2 = ntohl(res2); // 36 备用
        shipAzi2Ground = ntohs(shipAzi2Ground); // 40 对地航向
        shipVel2Ground = ntohs(shipVel2Ground); // 42 对地航速
        shipAzi2Water = ntohs(shipAzi2Water); // 44 对水航向
        shipVel2Water = ntohs(shipVel2Water); // 46 对水航速
        longitude = ntohl(longitude); // 48 经度
        latitude = ntohl(latitude); // 52 纬度
        height = ntohs(height); // 56 高度
        headSway = ntohs(headSway); // 58 首摇
        rollSway = ntohs(rollSway); // 60 横摇
        pitchSway = ntohs(pitchSway); // 62 纵摇
        sigProCtrl = ntohl(sigProCtrl); // 64 信号处理方式开关
        antennaScanSpeed = ntohs(antennaScanSpeed); // 68 天线扫描速度
        // unsigned char antennaWorkMode; // 70 天线工作方式
        // unsigned char res1; // 71 备用
        antennaScanRange = ntohs(antennaScanRange); // 72 天线扇扫范围
        antennaScanCenter = ntohs(antennaScanCenter); // 74 天线扇扫中心
        // unsigned char res2[4]; // 76 备用
        // unsigned char res2[64]; // 80 备用

        if(plotNum <= gMaxPlotInSector) // 88 plotNum 个点, 变长
        {
            for(int plotIdx = 0; plotIdx < plotNum; ++plotIdx)
            {
                plot[plotIdx].bytesNet2Host();
            }
        }
        else
        {
            std::cout << "Recieve plots num overflow: " << plotNum;
//            throw msgException(DtcPlotsNumWrong, msg.str());
        }
    }

    int bytes(void) const
    {
        return (sizeof(RH03AIfDtcSectorBody) - (gMaxPlotInSector - plotNum) * sizeof(RH03AIfDtcPlot));
    }
}RH03AIfDtcSectorBody;


// 扇区点迹网络报文
typedef struct RH03AIfDtcSector
{
    NaviIf::NaviHead head;
    RH03AIfDtcSectorBody body;
    NaviIf::NaviEnd end;

public:
    RH03AIfDtcSector(void)
    {
        head.msgLen = sizeof(*this); // need refill before send out
        head.msgID = RH03ATAG_SECPLOTS;
    }

    void bytesHost2Net(void) // 主机 -> 网络
    { // 变长报文, 对 end 转字节序会出错
        head.bytesHost2Net();
        body.bytesHost2Net();
    }

    void bytesNet2Host(void) // 网络 -> 主机
    {
        head.bytesNet2Host();
        body.bytesNet2Host();
    }

    int bytes(void) // 注意此函数只有在转到本地字节序后才可调用
    {
        return (sizeof(NaviIf::NaviHead) + body.bytes() + sizeof(NaviIf::NaviEnd));
    }
}RH03AIfDtcSector;

}

namespace YR18If {

typedef struct YR18CAT240
{
    unsigned char msgFormat; // 0, 报文格式标志, 填 240, CAT240 格式
    unsigned short msgLen; // 1, 报文长度, 16 进制
    unsigned short FSPEC; // 3, FSPEC 标志, 发端为 0xE7A8
    unsigned char SAC; // 5, System Area Code, 发端为 0
    unsigned char SIC; // 6, System Identification Code, 发端为 0
    unsigned char msgType; // 7, 报文类型标识, 发端为 2 表示视频数据

    unsigned int msgCounter; // 8, 报文计数器
    unsigned short aziStart; // 12, 视频起始方位 LSB: 360 / 65536
    unsigned short aziEnd; // 14, 视频终止方位 LSB: 360 / 65536

    unsigned int startRG; // 16, 雷达距离单元起始偏移
    unsigned int cellDur; // 20, 雷达分辨率 10^{-15}s

    unsigned char spare; // 24, 预留位 0
    unsigned char resolution; // 25, 视频解析率, 发端填 4, 8bit 视频
    unsigned short nbVB; // 26, 视频有效数值范围, 发端填 1024
    unsigned char nbCells0; // 28, 有效距离单元数, 发端填 1024
    unsigned short nbCells1; // 29, 有效距离单元数, 发端填 1024
    unsigned char res0; // 31, 预留

    unsigned char data[1024]; // 32, 有效距离单元数, 发端填 1024

    unsigned char time0; // 绝对时间, UTC 时间 LSB: 1/128 s
    unsigned short time1; // 绝对时间, UTC 时间

    void bytesNet2Host() {
//        unsigned char msgFormat; // 0, 报文格式标志, 填 240, CAT240 格式
        msgLen = ntohs(msgLen); // 1, 报文长度, 16 进制
        FSPEC = ntohs(FSPEC); // 3, FSPEC 标志, 发端为 0xE7A8
//        unsigned char SAC; // 5, System Area Code, 发端为 0
//        unsigned char SIC; // 6, System Identification Code, 发端为 0
//        unsigned char msgType; // 7, 报文类型标识, 发端为 2 表示视频数据

        msgCounter = ntohl(msgCounter); // 8, 报文计数器
        aziStart = ntohs(aziStart); // 12, 视频起始方位
        aziEnd = ntohs(aziEnd); // 14, 视频终止方位

        startRG = ntohl(startRG); // 16, 雷达距离单元起始偏移
        cellDur = ntohl(cellDur); // 20, 雷达分辨率 10^{-15}s

//        unsigned char spare; // 24, 预留位 0
//        unsigned char resolution; // 25, 视频解析率, 发端填 4, 8bit 视频
        nbVB = ntohs(nbVB); // 26, 视频有效数值范围, 发端填 1024
        nbCells1 = ntohs(nbCells1); // 28, 有效距离单元数, 发端填 1024

//        unsigned char data[1024]; // 28, 有效距离单元数, 发端填 1024

//        unsigned char time0; // 绝对时间, UTC 时间 LSB: 1/128 s
//        unsigned short time1; // 绝对时间, UTC 时间
    }
} YR18CAT240;

}

namespace RPSIf {
#define RPS_MAX_PLOT_NUM 1750 // 最大点迹数量
#define RPS_PLOT_SEAAIRFLAG 0x1234 // 空海点迹标志

struct tagPlotDisplay
{
    unsigned int unFlag; // 判定点迹源, 0xa5b6 - 空; 0x1234 - 海

    unsigned int unSectorSn; // sec id 0-31

    unsigned int unWorkMode; // 工作方式

    unsigned int unPlotNum;

    unsigned int unStartTime; // 起始时间 0.1ms

    unsigned int unStopTime; // 终止时间 0.1ms

    int nShip_V_Deg; // 纵摇

    int nShip_H_Deg; // 横摇

    unsigned int unShipCourse; // 航向

    unsigned int unShipVelocity; // 航速

    unsigned int unAziCode[RPS_MAX_PLOT_NUM]; // 点迹方位码 0.01°

    unsigned int unRange[RPS_MAX_PLOT_NUM]; // 点迹距离 1m

    int unEleCode[RPS_MAX_PLOT_NUM]; // 点迹仰角 0.01°

    unsigned int unAmp[RPS_MAX_PLOT_NUM]; // 点迹幅度 0-65536

    unsigned int unTime[RPS_MAX_PLOT_NUM]; // 点迹时间 0.1ms

    unsigned int unFusionFlag[RPS_MAX_PLOT_NUM]; // 点迹关联标志

    unsigned int unAziSpan[RPS_MAX_PLOT_NUM]; // 点迹方位展宽 0.01°

    unsigned int unDistSpan[RPS_MAX_PLOT_NUM]; // 点迹距离展宽 0.01m

    unsigned int unStripIdx[RPS_MAX_PLOT_NUM]; // 距离网格号

    tagPlotDisplay()
    {
        init();
    }
    void init()
    {
        memset(this, 0, sizeof(tagPlotDisplay));
        unFlag = RPS_PLOT_SEAAIRFLAG;
    }
};

}

namespace CAT240 {

typedef struct CAT240Video
{
    unsigned char msgFormat; // 0, 报文格式标志, 填 240, CAT240 格式
    unsigned short msgLen; // 1, 报文长度, 16 进制
    unsigned short FSPEC; // 3, FSPEC 标志, 发端 &0xE380 == 0xE380
    unsigned char SAC; // 5, System Area Code
    unsigned char SIC; // 6, System Identification Code
    unsigned char msgType; // 7, 报文类型标识, 发端为 2 表示视频数据

    unsigned int msgCounter; // 8, 报文计数器
    unsigned short aziStart; // 12, 视频起始方位 LSB: 360 / 65536
    unsigned short aziEnd; // 14, 视频终止方位 LSB: 360 / 65536

    unsigned int startRG; // 16, 雷达距离单元起始偏移
    unsigned int cellDur; // 20, 雷达分辨率, 0x0800 == (msgFSPEC & 0x0800), 10^{-9}s; 0x0800 != (msgFSPEC & 0x0800), 10^{-15}s

    unsigned char spare; // 24, 预留位 0
    unsigned char resolution; // 25, 视频解析率, 发端填 4, 8bit 视频
    unsigned short nbVB; // 26, 视频有效数值范围
    unsigned char nbCells0; // 28, 有效距离单元数
    unsigned short nbCells1; // 29, 有效距离单元数
    unsigned char rep; // 31, 后续视频 data block 数量

    // video block:
    // low data volume 4bytes, medium data volume 64bytes, high data volume 258bytes

    // unsigned char time0; // 绝对时间, UTC 时间 LSB: 1/128 s
    // unsigned short time1; // 绝对时间, UTC 时间

    void bytesNet2Host() {
        //        unsigned char msgFormat; // 0, 报文格式标志, 填 240, CAT240 格式
        msgLen = ntohs(msgLen); // 1, 报文长度, 16 进制
        FSPEC = ntohs(FSPEC); // 3, FSPEC 标志, 发端为 0xE7A8
        //        unsigned char SAC; // 5, System Area Code, 发端为 0
        //        unsigned char SIC; // 6, System Identification Code, 发端为 0
        //        unsigned char msgType; // 7, 报文类型标识, 发端为 2 表示视频数据

        msgCounter = ntohl(msgCounter); // 8, 报文计数器
        aziStart = ntohs(aziStart); // 12, 视频起始方位
        aziEnd = ntohs(aziEnd); // 14, 视频终止方位

        startRG = ntohl(startRG); // 16, 雷达距离单元起始偏移
        cellDur = ntohl(cellDur); // 20, 雷达分辨率 10^{-15}s

        //        unsigned char spare; // 24, 预留位 0
        //        unsigned char resolution; // 25, 视频解析率, 发端填 4, 8bit 视频
        nbVB = ntohs(nbVB); // 26, 视频有效数值范围, 发端填 1024
        nbCells1 = ntohs(nbCells1); // 28, 有效距离单元数, 发端填 1024

        //        unsigned char data[1024]; // 28, 有效距离单元数, 发端填 1024

        //        unsigned char time0; // 绝对时间, UTC 时间 LSB: 1/128 s
        //        unsigned short time1; // 绝对时间, UTC 时间
    }
} CAT240Video;

}

#pragma pack()// 恢复之前的对齐
#endif // OTHERINTERFACE_H
