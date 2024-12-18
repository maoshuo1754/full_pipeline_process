#ifndef MYSTRUCT_H
#define MYSTRUCT_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <arpa/inet.h>

#define SLOT_SIZE (4096*1024)//缓冲区槽大小
#define SLOT_NUM (32)//缓冲区槽个数

#define VEDIODATASIZE (25000)
#define HEAD_FLAG (0x7E7E)
#define END_FLAG (0xAAAA)
#define MAX_DISTANCE_ELEMENT_NUMBER 7500

//数据缓冲区
typedef struct _BaseRingBufferInfo {
    std::mutex mtLock;//保护锁
    std::condition_variable mtCdtR;//读数据的条件变量
    std::atomic<int> readOffset;//读偏移量
    std::atomic<int> writeOffset;//写偏移量
    std::atomic<int> count;//缓存计数
    char* dataBuf;//RingBuf
    uint32_t* dataSize;//缓冲区数据大小
}*ptrBaseRingBuffer, BaseRingBuffer;

#pragma pack(1) // 内存中 1 字节对齐

//报文头结构体
struct  _tagMsgHead
{
    unsigned short               usHead;                 //信息头 0x7E7E
    unsigned int                 ulTime;
    unsigned char                ucSendNode;             //发方节点号
    unsigned char                ucRecvNode;             //收方节点号
    unsigned short               usDragLength;           //信息总长度 包括报文头报文尾
    unsigned char                ucDragCAT;              //信息识别符
    unsigned char                ucInSureMark;           //信息确认标识
    unsigned char                ucRes1;                 //信息单元备用
    unsigned char                ucRes2;                 //信息单元备用
    unsigned short               usRes3;                 //信息单元备用

    _tagMsgHead()
    {
        memset(this,0,sizeof(*this));
        usHead = HEAD_FLAG;
    }
};
//报文报文尾
struct _tagMsgEnd
{
    unsigned short              usCRC;                  //CRC校验码
    unsigned short              usEnd;                  //信息尾 0xAAAA

    _tagMsgEnd()
    {
        memset(this,0,sizeof(*this));
        usEnd = END_FLAG;
    }
};

struct _EntireMessageInfo
{
    unsigned int   TrigleFlag;			// [0000-0001]同步头：0xD8D80606
    unsigned int   ServoFlag;			// [0002-0003]同步头：0xF4F4F4F4
    unsigned char RecvPoint; 			//雷达站点号,6a:0;19:1~255
    unsigned char PackKind;				//包数据种类,原始IQ-0;脉压后IQ-1；脉压后IQ＋求模-2;原始求模视频-3；检测后视频-4；点迹-5；
    unsigned char ResWord1[6];			//备用
    unsigned short AziVal;				// [0004]方位，单位：360/65536
    unsigned short DisCellNum;			//采样距离单元数
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
    unsigned char usPulsegroupPara[11];//脉组处理参数
    unsigned char ContAreaDet[2];	   //联通域检测
    unsigned short ShipAzi1;			//航向1
    unsigned short ShipVel1;			//航速1
    unsigned short ShipAzi2;			//航向２
    unsigned short ShipVel2;			//航速2
    int longitude;						//经度;单位:1/10000分
    int latitude;						//纬度;单位:1/10000分
    short height;						//高度;单位:1米
    unsigned char ResWord2[6];			//保留字
    unsigned int EchoFlag1;			// 零距开始：0xA5A51234
    unsigned int EchoFlag2;			//零距开始：0xA5A51234
    unsigned int EchoFlag3;			//流水号，临时用，后删除

    _EntireMessageInfo()
    {
        memset(this,0,sizeof(*this));
        TrigleFlag = 0xD8D80606;
        ServoFlag = 0xF4F4F4F4;
        EchoFlag1 = 0xA5A51234;
        EchoFlag2 = 0xA5A51234;
    }
};

//视频数据格式
//struct VideoMessageInfo
//{
//    _tagMsgHead  st_tagMsgHead;
//    _EntireMessageInfo  m_MessageInfo;//主动定义报文头
//    unsigned char data[VEDIODATASIZE];      //回波数据
//    _tagMsgEnd st_tagMsgEnd;

//};

struct _ControlInfo
{
    unsigned int oriVideoSend_Enable;//原始视频发送使能
    unsigned int detVideoSend_Enable;//检测视频发送使能
    unsigned int iqSend_Enable;//IQ对外发送使能
    unsigned int iqRecord_Enable;//IQ本地记录使能
    unsigned int iqRecord_Compress;//IQ记录是否压缩
    unsigned int pulseAcc_Enable;//脉冲滑窗积累使能
    unsigned int pulseAcc_Num;//脉冲滑窗积累脉冲数
    unsigned int aziResamp_Enable;//方位重采样使能
    unsigned int aziResamp_Num;//方位重采样方位总数

    unsigned int pulseSW_Num;//MTD滑窗数
    unsigned int pulseMTD_Num;//MTD脉冲数
    unsigned int pulseOut_Idx;//MTD方位输出脉冲号
    unsigned int pulseMTD_Enable;//MTD使能
    unsigned int dopCoefIdx;//滤波器系数选择
    unsigned int zeroDop_Enable;//零速滤波使能
    unsigned int cfarMode;//CFAR模式：0关闭,1CA,2SO,3GO,4SE,otherAT
    unsigned int cfarThreshold;//CFAR门限0.1
    unsigned int cfarThreshold_0;//CFAR门限0.1 0通道
    int iqAziAdj;//IQ路方位修正0.01°
    int iqDisAdj;//IQ路距离修正1m
    int detAziAdj;//检测路方位修正0.01°
    int detDisAdj;//检测路距离修正1m

    _ControlInfo()
    {
        oriVideoSend_Enable = 1;
        detVideoSend_Enable = 0;
        iqSend_Enable = 0;
        iqRecord_Enable = 0;
        iqRecord_Compress = 0;
        pulseAcc_Enable = 1;
        pulseAcc_Num = 5;
        aziResamp_Enable = 1;
        aziResamp_Num = 2048;

        pulseSW_Num = 6;
        pulseMTD_Num = 32;
        pulseOut_Idx = pulseMTD_Num / 2;
        pulseMTD_Enable = 1;
        dopCoefIdx = 5;
        zeroDop_Enable = 1;
        cfarMode = 3;
        cfarThreshold = 120;
        cfarThreshold_0 = 120;
        iqAziAdj = 0;
        iqDisAdj = 0;
        detAziAdj = 0;
        detDisAdj = 0;
    }
};

//视频转发操控命令
struct VideoTransControl
{
    _tagMsgHead  st_tagMsgHead;
    _ControlInfo st_ControlInfo;
    _tagMsgEnd st_tagMsgEnd;
};

struct XX6AHead
{
    unsigned short TrigleFlag1;                     // 主触发标识1:0xD8D8
    unsigned short TrigleFlag2;                     // 主触发标识2:0x0606
    unsigned short ServoFlag1;                      // 伺服数据头1:0xF4F4
    unsigned short ServoFlag2;                      // 伺服数据头2:0xF4F4
    unsigned short ServoAzi;                        // 伺服方位,LSB=360/65536
    unsigned short ShipDirect;                      // 航向,LSB=360/65536
    unsigned short Time1;                           // 时统63~48,高8位:天,低8位:时
    unsigned short Time2;                           // 时统47~32,高8位:分,低8位:秒
    unsigned short Time3;                           // 时统31~16,纳秒的高16位
    unsigned short Time4;                           // 时统15~0,纳秒的低16位
    unsigned short ShipVel;                         // 航速,LSB=360/65536
    unsigned short VerSway;                         // 纵摇,LSB=360/65536
    unsigned short HorSway;                         // 横摇,LSB=360/65536
    unsigned short ServoEle;                        // 伺服俯仰,LSB=360/65536
    unsigned short TrueAzi;                         // 真方位,LSB=360/65536
    unsigned short Longitude;                       // 平台经度,LSB=360/65536
    unsigned short Latitude;                        // 平台纬度,LSB=360/65536改为主触发计数
    unsigned short PulseCnt;                        // Altitude海拔高度,LSB=1m改为脉冲计数
    unsigned short WorkMode;                        // 工作方式
    unsigned short FreqCode;                        // 频率代码
    unsigned short PRTLen;                          // 重复周期,LSB=1us改为主触发距离单元长度
    unsigned short Resv;                            // 备用
    unsigned short EchoFlag1;                       // 回波数据头1:0xA5A5
    unsigned short EchoFlag2;                       // 回波数据头2:0x1234



};
static const int s_nXX6AHeadLenSB = sizeof(XX6AHead);		// 重排后报文的长度,单位:字节
static const int s_nXX6AHeadLenDB = s_nXX6AHeadLenSB >> 1;	// 重排后报文的长度,单位:单字
static const int s_nXX6AHeadLenFB = s_nXX6AHeadLenSB >> 2;	// 重排后报文的长度,单位:双字

struct XX6AParaInfo
{
    unsigned int unSyncHead; 				 		// 同步报文头:0xA5A51234
    unsigned int unMsgCount; 				 		// 重排后包计数
    unsigned int unCurrTime;                                            // 当前时间, LSB = ms
    unsigned int unAziCode;                                             // 当前方位, LSB = 360/65536
    unsigned int unPulseNum; 				 		// 脉冲总数
    unsigned int unBurstNum;                        // 脉组总数
    unsigned int unBurstCnt;                        // 脉组计数
    unsigned int unWorkMode;                        // 工作方式
    unsigned int unPRTLen;                          // 主触发距离单元长度
    unsigned int unProcessEnSVC;					// 舰速补偿
    unsigned int unProcessEnMTD;					// 动目标检测/非相参
    unsigned int unDopCoefIdx;						// 多普勒滤波器系数序号
    unsigned int unProcessEnZSF;					// 零速滤波, 0通道系数置零
    unsigned int unChannelNum;						// 多普勒滤波后通道数
    unsigned int unProcessEnCFAR;					// 恒虚警方式
    unsigned int unThreshold;						// 恒虚警检测门限
    unsigned int unThreshold_0;						// 恒虚警检测门限0通道
    unsigned int unEnMTDVideoSend;					// MTD视频发送开关
    unsigned int unEnDetVideoSend;					// 检测视频发送开关
    int nLongitude;                                 //经度;单位:1/10000分
    int nLatitude;                                  //纬度;单位:1/10000分
};
static const int s_nXX6AParaInfoLenSB = sizeof(XX6AParaInfo);		// 重排后报文的长度,单位:字节
static const int s_nXX6AParaInfoLenDB = s_nXX6AParaInfoLenSB >> 1;	// 重排后报文的长度,单位:单字
static const int s_nXX6AParaInfoLenFB = s_nXX6AParaInfoLenSB >> 2;	// 重排后报文的长度,单位:双字

//udp数据拼包包头
struct UdpDataHead
{
    unsigned int unSyncHead; 				 		// 同步报文头:0x55AA0724
    unsigned int unPacketCnt;                       // 小包计数, 判断是否丢包
    unsigned int unMtpCnt;                          // 触发标识, 每个主触发＋1
    unsigned int unLittlePacketCnt;                 // 触发内小包计数
    unsigned int unIQLength;                        // 数据长度
    unsigned int unResv[3];                         // 预留
};
static const int s_nUdpDataHeadLenSB = sizeof(UdpDataHead);		// 单位:字节

//平台经纬度信息
struct PlatformInfo
{
    unsigned short              Head;                   //报文头 0x7E7E
    unsigned int                TimeStamp;              //报文发送时间ms,LSB=1
    unsigned char               SendPoint;              //发方节点号
    unsigned char               RecvPoint;              //收方节点号
    unsigned short              DragLength;             //报文总长度0x1C
    unsigned char               DragCAT;                //报文识别符0x61
    unsigned char               InSureMark;             //应答标识0x00
    unsigned short              usState;                //状态及数据有效标识
    short                       sRelSpeed;              //相对航速,量纲:0.01节,取值范围[-12~40]
    short                       sAbsSpeed;              //绝对航速,量纲:0.01节,取值范围[-40~40]
    unsigned short              usTrackAngle;           //航迹向角,量纲:180/2^15度,取值范围[0~360)
    short                       sEastSpeed;             //东向速度,量纲:0.01节,取值范围[-40~40]
    short                       sNorthSpeed;            //北向速度,量纲:0.01节,取值范围[-40~40]
    int                         lLongitude;             //经度,量纲:90/2^30度,取值范围(-180~180]
    int                         lLatitude;              //纬度,量纲:90/2^30度,取值范围(-90~90]
    short                       sUpSpeed;               //深沉速度,量纲:0.01m/s
    unsigned long long          ullTime;                //时间,世界标准时或世界协调时或地方时
    short                       sAvgTrueWindSpeed;      //平均真风速,量纲:0.01m/s
    short                       sAvgTrueWindDirect;     //平均真风向,量纲:180/2^14度,取值范围[0~360)
    short                       sAvgRelWindSpeed;       //平均相对风速,量纲:0.01m/s,取值范围[0~70]
    short                       sAvgRelWindDirect;      //平均相对风向,量纲:180/2^14度,取值范围[0~360)
    short                       sTemperature;           //温度,量纲:0.1摄氏度,取值范围[-40~50]
    unsigned short              usHumidity;             //相对湿度,单位:%RH,取值范围[0~100]
    unsigned short              usPressure;             //大气压,单位:hpa,取值范围[600~1100]
    unsigned short              usOceanSpeed;           //海流速度
    unsigned short              usOceanDirect;          //海流流向
    unsigned int                ulOceanDepth;           //海深,量纲:0.01m,取值范围[1~2000]
    unsigned short              usOceanCondition;       //海况
    unsigned short              usZTflag;               //姿态有效标识 1无效 0有效
    int                         lHeadingAngle;          //艏向角,量纲:0.01度,取值范围[0~360)
    int                         lPitchAngle;            //纵摇角,量纲:0.01度,取值范围[-30~30]
    int                         lRollAngle;             //横摇角,量纲:0.01度,取值范围[-45~45]
    unsigned short              CRC;                    //CRC校验码
    unsigned short              End;                    //报文尾0xAAAA
};

#include <cstdint>
typedef uint32_t UINT32;
typedef uint16_t UINT16;
typedef uint8_t UINT8;
typedef int32_t INT32;
typedef unsigned int UINT;
typedef int16_t INT16;
typedef int8_t INT8;

typedef struct tagNRX_COMMON_HEADER
{
    UINT32 dwHEADER;									//����ͷ	0xF1A2B4C8
    UINT16 wVERSION;									//Э��汾��	���ݹ�������ͷ�ı����ʷ��ţ�ĿǰΪ0
    UINT16 wCOUNTER;									//������	���౨�Ķ��Լ���
    UINT32 dwTxSecondTime;								//����ʱ��1	32λUTC����ʾ�롣�ɷ�����ȡ����ʱ����д��
    UINT32 dwTxMicroSecondTime;							//����ʱ��2	��ʾ�����µ�΢����
    UINT16 wMsgTotalLen;								//������������ͷ�����ݡ�����β�����ֽ��������Ϊ64K�ֽ�.FPGA���͵����ݴ�����0.
    UINT16 wMsgFlag;									//����ʶ���	�ⲿ����ʹ��0-255���ڲ�����ʹ��256-65535
    UINT16 wRadarID;									//�״�ID	Ӧ�ó����Զ���
    UINT8  bytTxNodeNumber;								//�����ڵ��	ϵͳΪ�ܹ����ͻ�������ݵ���Ӳ��ʵ�����ڵ��
    UINT8  bytRxNodeNumber;								//�շ��ڵ��
    UINT8  bytDataFlag;									//���ݱ��	b7-4����ѹ����ǡ�0x0, ����δѹ��; 0x1, ����ʹ�� qt sdk ѹ��;����������.

    UINT8 bytRecChannel;								//��¼����ͨ����	��¼�ط�ʹ�õ�ͨ���ţ��������ֶ�ͨ������
    UINT16 wReserved0;									//Ԥ��	��0
    UINT16 wReserved1;									//Ԥ��	��0
    UINT16 wResesrved2;									//Ԥ��	��0
}NRX_COMMON_HEADER;

typedef struct tagNRX_RadarVideo_Head
{
    UINT32 dwSyncHeader;							//ͬ��ͷ��0xA5A61234
    UINT32 dwVideoLen;								//�״���Ƶ�����ѹ�����ֽ���������������ͷ���״���Ƶͷ������β�ĳ���
    UINT16 wHeadLen;								//�״���Ƶͷ����,���������ݣ���128.
    UINT16 wEncodeFormat;							//0 8λ��Ƶ;1 16λ��Ƶ.;2 8λ��Ƶ + 8λ����;3 16λ��Ƶ + 16λ����;4 8λ��Ƶ + 8λ���� + 8λ�ٶ� + 8λԤ��. �ٶ��ò����ʾ.
    //100~127��ר�����ڲ�����ʹ��. 100��32λ���������ȣ�dB�� + ͨ���ٶ�chnnalSpeed  101��32λ���������ȣ�dB�� + 32λ������������dB�� + ͨ���ٶ�chnnalSpeed
    //102��32λ���������ȣ�dB�� + 32λ������������dB�� + 32λ�������ٶȣ�m / s��
    UINT8 bytPulseMode;								//������Ϸ�ʽ,0��������  1�������岹ä	2��MTD��ͨ��  3��1�� + ���
    UINT8 bytSubPulseNumber;						//���������,�����������������	���磺�������޲�ä���壬��ֵ��1��������1��ä����ֵ��2��	1��������16�������壬��ֵ��17��
    UINT8 bytSubPulseCount;							//��ǰ�����������е���������ţ�[0, n)
    UINT8 bytReserved0;								//Ԥ������ʾ������Ϸ�ʽ
    UINT32 dwTxAbsSecondTime;						//����ʱ��1 32λUTC����ʾ��� 0 - 86399999ms���ɹ�������ͷ�е����ݱ�ǵ�b3λ������
    UINT32 dwTxAbsMicroSecondTime;					//����ʱ��2 ��ʾ�����µ�΢������ ��Ч���ɹ�������ͷ�е����ݱ�ǵ�b3λ������
    UINT32 dwTxRelMilliSecondTime_H;				//���ʱ�� ��32λ����λ: 1ms
    UINT32 dwTxRelMilliSecondTime_L;				//���ʱ�� ��32λ����λ: 1ms
    UINT32 dwSigBWHz;								//�źŴ���,LSB��1Hz��ȫF��ʾ��Ч
    UINT32 dwSampleFreqHz;							//������,LSB��1Hz
    UINT16 wAziCode;								//��λ 16λ���룬360��/65536
    UINT16 wPulseWidth0p1us;						//���� LSB��0.1us��ȫF��ʾ��Ч
    UINT16 wPRT0p1us;								//PRT LSB��0.1us��ȫF��ʾ��Ч
    INT16 nZeroPointPos;							//��ʼ��Ԫ��� ��0�����뵥Ԫ������ĵ�Ԫ��
    UINT32 dwSampleElementNumber;					//������Ԫ����
    UINT32 dwReserved1;								//Ԥ��
    UINT8 bytReserved2;								//Ԥ��
    UINT8 bytPIM_Flag;								//PIM_Flag 0: ԭʼ. 1: ����. 2: ���.	0xFF��Ч.
    UINT8 bytDataFlag;								//���ݱ�ʶ b7-5��Ƶ��������. 0dB; 1����ӳ��; 2Լ��������ӳ��.b4 - 0Ԥ��
    UINT8 bytLinearMapLowPara;						//����ӳ����� ����ӳ��ʱ��Ч. ӳ��ǰ��������
    UINT8 bytLinearMapHighPara;						//����ӳ����� ����ӳ��ʱ��Ч. ӳ��ǰ��������	val = 0 (if DB <= mapPreLowerDB) 		val = 2 ^ n - 1 (if DB >= mapPreUpperDB) 		val = (DB - mapPreLowerDB) / (mapPreUpperDB - mapPreLowerDB) * (2 ^ n - 1)
    UINT8 bytReserved3;								//Ԥ��
    UINT16 wDataSrc;								//����Դ �����ļ���ͨ����λ��ķ�ʽʹ��.���֧��16������Դ.	1:Simple Test		2 : Scenario Generator		4 : Replay From Recording		8 : Receive By UDP		2 ^ (4:15)Ԥ��
    INT32 nLongitude;								//���� LSB��1/10000�֣�181�ȱ�ʾ��Ч
    INT32 nLatitude;								//γ�� LSB��1/10000�֣�91�ȱ�ʾ��Ч
    INT16 nAltitude;								//�߶� LSB��1�ף�Ĭ����0
    UINT16 wAbsCourse;								//���Ժ��� LSB��360/65536��Ĭ����0
    UINT16 wAbsCruiseSpeed;							//���Ժ��� LSB��0.1m/s��Ĭ����0
    UINT16 wRelCourse;								//��Ժ���	LSB��360/65536��Ĭ����0
    UINT16 wRelCruiseSpeed;							//��Ժ���	LSB��0.1m/s��Ĭ����0
    INT16 nHeadAngle;								//��ҡ	��λ��360��/32768  Ĭ��ֵ0
    INT16 nRoll;									//��ҡ	��λ��360��/32768  Ĭ��ֵ0
    INT16 nPitch;									//��ҡ	��λ��360��/32768  Ĭ��ֵ0
    UINT8 bytScanMode;								//ɨ�跽ʽ	b7: ��ʾ��������ɨ�跽ʽ�Ƿ���Ч. 1��Ч0��Ч.b6: 0, �̶�ƽ̨; 1, �ƶ�ƽ̨.	b5Ԥ��.	b4 - 0: 0˳ʱ�뻷ɨ, 1��ʱ�뻷ɨ, 2˳ʱ���е��ɨ, 3��ʱ���е��ɨ, 4˳ʱ�뵥���ɨ, 5��ʱ�뵥���ɨ, 6���ɨ��, 7��λ, 8ͣ����, 9����. b4 - 0 = 31��ʾ��Ч
    UINT8 bytReserved4;								//RES4	Ԥ��
    UINT16 wAntennaScanSpeed;						//����ɨ���ٶ�	��λ: 0.1 deg/s����Чʱ��0.
    UINT16 wFanScanFrontAngle;						//������ɨǰ��	��λ: 360.f / 65536.f����Чʱ��0.
    UINT16 wFanScanBackAngle;						//������ɨ����	��λ: 360.f / 65536.f����Чʱ��0.
    INT32 nChannelSpeed;							//ͨ���ٶ�	�ٶ��ò����ʾ��LSB = 0.1m/s  0xFFFFʱ��Ч
    UINT16 wChannelCount;							//ͨ�����	0xFFʱ��Ч
    UINT16 wReserved5;								//RES5	Ԥ��
    UINT8 wReserved6;								//RES6[16]
    UINT32 dwReserved7;								//RES7	����βԤ��
    UINT32 dwMsgTailFlag;							//����β	0xB5B65678
}RX_RadarVideo_Head;

typedef struct tagNRX_COMMON_TAIL
{
    UINT dwCheckSum;
    UINT16 wTail1;
    UINT16 wTail2;
}NRX_COMMON_TAIL;

struct VideoToNRXGUI
{
    NRX_COMMON_HEADER CommonHeader;
    RX_RadarVideo_Head RadarVideoHeader;
    UINT8 bytVideoData[MAX_DISTANCE_ELEMENT_NUMBER];
    NRX_COMMON_TAIL CommonTail;

    VideoToNRXGUI()
    {
        memset(&CommonHeader,0, sizeof(CommonHeader));
        CommonHeader.dwHEADER = htonl(0xF1A2B4C8);
        CommonHeader.wVERSION = htons(0);
        CommonHeader.wMsgTotalLen = htons(sizeof(CommonHeader) + sizeof(RadarVideoHeader)+sizeof(bytVideoData)+sizeof(CommonTail));
        CommonHeader.wMsgFlag = htons(0x0103);
        CommonHeader.wRadarID = htons(0x0012);
        CommonHeader.bytTxNodeNumber = 0x11;
        CommonHeader.bytRxNodeNumber = 0x22;
        CommonHeader.bytDataFlag = 0x08;
        CommonHeader.bytRecChannel = 0;

        memset(&RadarVideoHeader,0, sizeof(RadarVideoHeader));
        RadarVideoHeader.dwSyncHeader = htonl(0xa5a61234);
        RadarVideoHeader.dwVideoLen = htonl(sizeof(bytVideoData));
        RadarVideoHeader.wHeadLen = htons(128);
        RadarVideoHeader.wEncodeFormat = htons(0);
        RadarVideoHeader.bytPulseMode = 0;
        RadarVideoHeader.bytSubPulseNumber = 1;
        RadarVideoHeader.bytSubPulseCount = 0;

        RadarVideoHeader.dwSigBWHz = htonl(6e6);
        RadarVideoHeader.dwSampleFreqHz = htonl(31.25e6);
        RadarVideoHeader.wPulseWidth0p1us = htons(0xffff);
        RadarVideoHeader.wPRT0p1us = htons(0xffff);
        RadarVideoHeader.nZeroPointPos = htons(0);
        RadarVideoHeader.dwSampleElementNumber =htonl( sizeof(bytVideoData));
        RadarVideoHeader.bytPIM_Flag = 0;
        RadarVideoHeader.bytDataFlag = 0x40;
        RadarVideoHeader.bytLinearMapLowPara = 0;
        RadarVideoHeader.bytLinearMapHighPara = 0xff;

        RadarVideoHeader.wDataSrc = htons(8);

        RadarVideoHeader.nLongitude = htonl(181 * 60 * 10000);
        RadarVideoHeader.nLatitude = htonl(91 * 60 * 10000);
        RadarVideoHeader.nAltitude = htons(0);

        RadarVideoHeader.bytScanMode = 0x86;	//0x10000110

        RadarVideoHeader.dwMsgTailFlag = htonl(0xB5B65678);

        CommonTail.dwCheckSum = htonl(0);
        CommonTail.wTail1 = htons(0xABCD);
        CommonTail.wTail2 = htons(0xEF89);
    }
};

#pragma pack()// 恢复之前的对齐
#endif // MYSTRUCT_H
