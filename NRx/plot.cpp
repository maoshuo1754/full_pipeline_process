#include "plot.h"
#include <sstream>
#include "include/NRxPIM.h"
#include "include/NRxParamsMgr.h"
#include "libs/include/timer/Timer.h"
#include "MyStruct.h"
#include "arpa/inet.h"
#include <include/otherInterface.h>
#include <../Config.h>

static const int32 TEMP_MAX_SECTOR_NUM(32);
static const int32 MaxPlotNumInPlotDataSet(1024);
using namespace ::libParam;
// PlotDet Param
static const string tagName{"PlotDet"};

struct PlotDetConfig {
    string plotdetEnableName{"plotdetEnable"};
    string plotdetFusionModeName{"plotdetFusionMode"};
    string plotdetDisGridWidName{"plotdetDisGridWid"};
    string plotdetMinEchoAmpName{"plotdetMinEchoAmp"};
    string plotdetMaxEchoAmpName{"plotdetMaxEchoAmp"};
    string plotdetDisDetMName{"plotdetDisDetM"};
    string plotdetDisDetNName{"plotdetDisDetN"};
    string plotdetMinAziWidName{"plotdetMinAziWid"};
    string plotdetMaxAziWidName{"plotdetMaxAziWid"};
    string plotdetMinDisWidName{"plotdetMinDisWid"};
    string plotdetMaxDisWidName{"plotdetMaxDisWid"};
    string plotdetMinCellNumName{"plotdetMinCellNum"};
    string plotdetMaxCellNumName{"plotdetMaxCellNum"};
    string plotdetNoiDecThrName{"plotdetNoiDecThr"};
};
static const PlotDetConfig paramName;
static const string tag1Name{"SlbeDet"};
struct SlbeDetConfig {
    string slbedetEnableName{"slbedetEnable"};
    string slbedetGridAziWidName{"slbedetGridAziWid"};
    string slbedetGridDisWidName{"slbedetGridDisWid"};
    string slbedetMlobeAmpThrName{"slbedetMlobeAmpThr"};
    string slbedetDisSlobeDisWidName{"slbedetDisSlobeDisWid"};
    string slbedetDisSlobeDisRangeName{"slbedetDisSlobeDisRange"};
    string slbedetDisSlobeAziRangeName{"slbedetDisSlobeAziRange"};
    string slbedetMinMainDisSideRatioName{"slbedetMinMainDisSideRatio"};
    string slbedetMaxMainDisSideRatioName{"slbedetMaxMainDisSideRatio"};
    string slbedetAziSlobeAziWidName{"slbedetAziSlobeAziWid"};
    string slbedetAziSlobeDisRangeName{"slbedetAziSlobeDisRange"};
    string slbedetFirstAziSlobePosName{"slbedetFirstAziSlobePos"};
    string slbedetSecondAziSlobePosName{"slbedetSecondAziSlobePos"};
    string slbedetMinMainAziSideRatio_0Name{"slbedetMinMainAziSideRatio_0"};
    string slbedetMaxMainAziSideRatio_0Name{"slbedetMaxMainAziSideRatio_0"};
    string slbedetMinMainAziSideRatio_1Name{"slbedetMinMainAziSideRatio_1"};
    string slbedetMaxMainAziSideRatio_1Name{"slbedetMaxMainAziSideRatio_1"};
    string slbedetMinMainAziSideRatio_2Name{"slbedetMinMainAziSideRatio_2"};
    string slbedetMaxMainAziSideRatio_2Name{"slbedetMaxMainAziSideRatio_2"};
};
static const SlbeDetConfig param1Name;


Plot::Plot()
        : usSectorPlotNum(0), usPreSectorId(255), dPrePulseAzi(-1.0), bCWScanNorthSign(false),
          bACWScanNorthSign(false) {
    m_pimAziDim = 180;//2048
    pimRangeSamples = 10240;
    PrePlotBuff = new sTempPlotBuff[pimRangeSamples];
    CurPlotBuff = new sTempPlotBuff[pimRangeSamples];
    PreDisDetInfo = new int32[pimRangeSamples];
    CurDisDetInfo = new int32[pimRangeSamples];
    FindConGridIdx = new int32[pimRangeSamples];
    memset(PreDisDetInfo, -1, sizeof(int32) * pimRangeSamples);
    memset(CurDisDetInfo, -1, sizeof(int32) * pimRangeSamples);
    memset(FindConGridIdx, -1, sizeof(int32) * pimRangeSamples);
    PlotInfoBuff = new NRxIf::NRxPlot[MaxPlotInSector];
    usCurPulDetSign = new uint16[pimRangeSamples];
    dCurPulseData = new double[pimRangeSamples];
    // 副瓣点迹判决用
    CurSecPlotBuff.reserve(BUFF_MAX_SECTOR_GRID_NUM);
    PreSecPlotBuff.reserve(BUFF_MAX_SECTOR_GRID_NUM);
    CurSecGridPlotNum.resize(BUFF_MAX_SECTOR_GRID_NUM, 0);
    PreSecGridPlotNum.resize(BUFF_MAX_SECTOR_GRID_NUM, 0);
    for (uint32 idx = 0; idx < BUFF_MAX_SECTOR_GRID_NUM; idx++) {
        std::vector<NRxIf::NRxPlot> SecPlotBuffRef;
        SecPlotBuffRef.reserve(MAX_SECTOR_GRID_PLOT_NUM);
        for (uint32 idx1 = 0; idx1 < MAX_SECTOR_GRID_PLOT_NUM; idx1++) {
            NRxIf::NRxPlot SecPlotBuffRef_ref;
            SecPlotBuffRef.push_back(SecPlotBuffRef_ref);
        }
        CurSecPlotBuff.push_back(SecPlotBuffRef);
        PreSecPlotBuff.push_back(SecPlotBuffRef);
    }
    nrx8bitPulse = nullptr;
    nrx8bitPulse = new NRx8BitPulse();
    nrx8bitPulse->CreateData(1024 * 64);
    m_nrx8bitBuf = nullptr;
    m_nrx8bitBuf = new char[1024 * 64];
    memset(m_nrx8bitBuf, 0, 1024 * 64);
    m_outbuf = nullptr;
    m_outbuf = new char[1024 * 64];
    memset(m_outbuf, 0, 1024 * 64);
    mtdLowBound = 0;
    mtdHighBound = INT16_MAX;
//    udpSender::addUdpSender("ConvPlot", "0.0.0.0", 0);
    outfile = std::ofstream("plots/plots.txt", std::ios::app);

    plot_id = 0;
}

Plot::~Plot() {
    if (PrePlotBuff != nullptr) {
        delete[] PrePlotBuff;
        PrePlotBuff = nullptr;
    }
    if (CurPlotBuff != nullptr) {
        delete[] CurPlotBuff;
        CurPlotBuff = nullptr;
    }
    if (PreDisDetInfo != nullptr) {
        delete[] PreDisDetInfo;
        PreDisDetInfo = nullptr;
    }
    if (CurDisDetInfo != nullptr) {
        delete[] CurDisDetInfo;
        CurDisDetInfo = nullptr;
    }
    if (FindConGridIdx != nullptr) {
        delete[] FindConGridIdx;
        FindConGridIdx = nullptr;
    }
    if (PlotInfoBuff != nullptr) {
        delete[] PlotInfoBuff;
        PlotInfoBuff = nullptr;
    }
    if (usCurPulDetSign != nullptr) {
        delete[] usCurPulDetSign;
        usCurPulDetSign = nullptr;
    }
    if (dCurPulseData != nullptr) {
        delete[] dCurPulseData;
        dCurPulseData = nullptr;
    }
    if (nrx8bitPulse != nullptr) {
        delete[] nrx8bitPulse;
        nrx8bitPulse = nullptr;
    }
    if (m_nrx8bitBuf != nullptr) {
        delete[] m_nrx8bitBuf;
        m_nrx8bitBuf = nullptr;
    }
    if (m_outbuf != nullptr) {
        delete[] m_outbuf;
        m_outbuf = nullptr;
    }
    outfile.close();
}

void Plot::MainFun(char *dataBuf, unsigned int dataSize, int *speeds) {
    // 将dataBuf转为NRx8bitPulse
    XX92NRx8bit(dataBuf);

    // 点迹凝聚使用速度信息
    PlotConv(nrx8bitPulse, speeds, NFFT);

}

void Plot::setSocket(int socket, sockaddr_in addr) {
    memset(&remotePlotAddr, 0, sizeof(remotePlotAddr));
    remotePlotAddr.sin_family = AF_INET;
    remotePlotAddr.sin_port = htons(remote_plot_port);  // 0x2001是8192  0x2002岁8194
    remotePlotAddr.sin_addr.s_addr = inet_addr(remote_plot_ip.c_str());

    localSocket = socket;
}


void Plot::XX92NRx8bit(char *xx9buf) {
    auto* pVideoToNRXGUI = reinterpret_cast<VideoToNRXGUI*>(xx9buf);

    auto* header = (NRxIfHeader *)m_nrx8bitBuf;
    // fill in head
    header->head = NRxIfHead;
    header->protocol = NRxProtolVerion;
    static unsigned short uscounter = 0; // 发送流水号
    header->counter = uscounter++;

//    gettimeofday(&tv, nullptr);
    header->time = pVideoToNRXGUI->CommonHeader.dwTxSecondTime;
    header->microSecs = pVideoToNRXGUI->CommonHeader.dwTxMicroSecondTime;

//    header->msgBytes; // fill before snd
    header->tag = NRxIfTag_VidOri;
    header->rdID = 0;
    header->sendID = 0;
    header->rcvID = 0;
    header->cpr = 0; // 自定义时间
    header->rdrChnID = 0;
    header->res1 = 0;
    header->res2 = 0;
    header->res3 = 0;

    auto* vidInfo = (NRxVidInfo *) (header + 1);
    vidInfo->vidSyncHead = 0xA5A61234;
    vidInfo->vidFormat = 0; // process only support 8bit - 20221203
    vidInfo->pulseCombineMode = 0;
    vidInfo->subPulseNum = 1;
    vidInfo->subPulseNo = 0;

    //绝对时间用XX9时统时间
    // vidInfo->absTime0 = (uint32) xx9VidMsgInfo.Time1;
    // vidInfo->absTime1 = 0;

    vidInfo->absTime0 = pVideoToNRXGUI->CommonHeader.dwTxSecondTime;
    vidInfo->absTime1 = pVideoToNRXGUI->CommonHeader.dwTxMicroSecondTime;
    //自守时
    double dUTCMicroSec = vidInfo->absTime0 * 1e6 + vidInfo->absTime1;

    static double dUTCMicroSecPre = dUTCMicroSec;
    static double dRelTime = 0;
    double dDeltaTime = (dUTCMicroSec - dUTCMicroSecPre) / 1e3;
    dRelTime += dDeltaTime;
    vidInfo->relTime0 = getHL32bit(dRelTime).second;
    vidInfo->relTime1 = getHL32bit(dRelTime).first;
    dUTCMicroSecPre = dUTCMicroSec;

    vidInfo->bandWidth = ntohl(pVideoToNRXGUI->RadarVideoHeader.dwSigBWHz);
    // vidInfo->sampleRate = 5000000; // 5MHz
    vidInfo->sampleRate = ntohl(pVideoToNRXGUI->RadarVideoHeader.dwSampleFreqHz);
    vidInfo->azi = ntohs(pVideoToNRXGUI->RadarVideoHeader.wAziCode);
//    printf("[Azi] %d\n", vidInfo->azi);
    vidInfo->pulseWidth = 0xFFFF;
    vidInfo->prt = 2500;
    vidInfo->startCellNo = 0;
    vidInfo->cellNum = htonl(pVideoToNRXGUI->RadarVideoHeader.dwVideoLen);
    vidInfo->PIMFlag = 0xFF;

    vidInfo->dataSource = 8; // udp source
//    vidInfo->longitude = pVideoToNRXGUI->RadarVideoHeader.nLongitude;
//    vidInfo->latitude = pVideoToNRXGUI->RadarVideoHeader.nLatitude;
    vidInfo->longitude = 0;
    vidInfo->latitude = 0;

//    switch (xx9Video->st_tagMsgHead.ucSendNode) {
//        case 1://DD
//            vidInfo->longitude = (112 * 60 + 40 + 20.f / 60) * 10000;
//            vidInfo->latitude = (16 * 60 + 39 + 42.f / 60) * 10000;
//            break;
//        case 2://ZJ
//            vidInfo->longitude = (111 * 60 + 11 + 47.f / 60) * 10000;
//            vidInfo->latitude = (15 * 60 + 46 + 56.f / 60) * 10000;
//            break;
//        default:
//            break;
//    }
    vidInfo->high = 0;

    vidInfo->scanType = (1 << 7) & 10;
    // vidInfo->servoScanSpeed = 300;
    vidInfo->servoScanSpeed = 13635;
    vidInfo->servoStartAzi = 0;
    vidInfo->servoEndAzi = 0;
    vidInfo->channelSpeed = 0xFFFF;
    vidInfo->channelNo = 0xFF;
    vidInfo->vidSyncTail = 0xB5B65678;

    auto* transedData = (unsigned char *) (vidInfo + 1);
    memcpy(transedData, pVideoToNRXGUI->bytVideoData, vidInfo->cellNum);

    auto* end = (NRxIfEnd *) (transedData + vidInfo->cellNum);
    end->CRC = 0;
    end->end1 = NRxIfEnd1;
    end->end2 = NRxIfEnd2;

    vidInfo->vidLength = vidInfo->cellNum;
    header->msgBytes = sizeof(NRxIfHeader) + sizeof(NRxIfEnd) + sizeof(NRxVidInfo) + vidInfo->cellNum;

    nrx8bitPulse->SetIfHeader(*header);
    nrx8bitPulse->SetVidInfo(*vidInfo);
    nrx8bitPulse->SetData(transedData);
    nrx8bitPulse->SetIfEnd(*end);
}


void Plot::PlotConv(NRx8BitPulse *res_a, int *speed, size_t speedLength) {
//    cout << speed[326] << endl;
    // 是否测试时间
//    bool bIsTestTime = NRxObj::isTestTime();
//    static uint32 uiProPulseAll = 0;
//    static double dProTimeAll = 0.0;
//    if (bIsTestTime)
//    {
//        StaticTimer::tick();
//    }

    plot2DParam.bPlotDet_Enable = true;

    /* 主要参数设置 */
    plot2DParam.uiFusionMode = 0;       // 凝聚方式：0质心 1几何中心
    // plot2DParam.dDisGridWid = getIntParam(tagName, paramName.plotdetDisGridWidName);     // 保留参数，距离格子宽度（米），为了节约计算量，可以粗略画格子
    plot2DParam.uiMinEchoAmp = 0;       // 最小回波幅度
    plot2DParam.uiMaxEchoAmp = 255;     // 最大回波幅度
    plot2DParam.uiDisDet_M = 3;    // 距离检测，M
    plot2DParam.uiDisDet_N = 4;    // 距离检测，N
    plot2DParam.dMinAziWid = 1;         // 点迹最小方位展宽
    plot2DParam.dMaxAziWid = 360;       // 点迹最大方位展宽
    plot2DParam.dMinDisWid = 1;        // 点迹最小距离展宽
    plot2DParam.dMaxDisWid = 10000;    // 点迹最大距离展宽
    plot2DParam.uiMinCellNum = 3;        // 点迹最小距离单元数
    plot2DParam.uiMaxCellNum = 65535;     // 点迹最大距离单元数

    plot2DParam.usNosJugThr = 37;

    plot2DParam.bSlbeDet_Enable = 0;
    plot2DParam.dGridAziWid = 11.25;      // 格子方位宽度
    plot2DParam.dGridDisWid = 5000;       // 格子距离宽度
    plot2DParam.iMlobeAmpThr = 136;       // 大于该门限的认为可能存在副瓣点迹
    plot2DParam.dDisSlobeDisWid = 10800;      // 距离副瓣距离延伸范围（副瓣点迹一般位于上下终点）
    plot2DParam.dDisSlobeDisRange = 2000;     // 距离副瓣距离波动范围，米，以上下延伸的终点为中心
    plot2DParam.dDisSlobeAziRange = 0.60;      // 距离副瓣方位波动范围，度，以上下延伸的终点为中心
    plot2DParam.iMinMainDisSideRatio = 55;    // 距离主副比下限
    plot2DParam.iMaxMainDisSideRatio = 136;   // 距离主副比上限
    plot2DParam.dAziSlobeAziWid = 360;     // 方位副瓣方位延伸范围（延伸范围内皆有可能）
    plot2DParam.dAziSlobeDisRange = 300;     // 方位副瓣距离波动范围
    plot2DParam.dFirstAziSlobePos = 2.0;       // 第1方位副瓣位置
    plot2DParam.dSecondAziSlobePos = 6.0;      // 第2方位副瓣位置
    plot2DParam.iMinMainAziSideRatio_0 = 82;     // 方位主副比下限(0~2deg)
    plot2DParam.iMaxMainAziSideRatio_0 = 191;    // 方位主副比上限(0~2deg)
    plot2DParam.iMinMainAziSideRatio_1 = 82;     // 方位主副比下限(2~6deg)
    plot2DParam.iMaxMainAziSideRatio_1 = 191;    // 方位主副比上限(2~6deg)
    plot2DParam.iMinMainAziSideRatio_2 = 82;     // 方位主副比下限(>6deg)
    plot2DParam.iMaxMainAziSideRatio_2 = 191;    // 方位主副比上限(>6deg)


    /* 获取当前脉冲数据 */
    NRx8BitPulse *res_b(nullptr);
    NRx8BitPulse *res_c(nullptr);

    /* 过正北判断 */
    ScanOverNorthJudge(res_a);
    /* 距离检测 */
    DisDetCov(res_a, res_b, res_c, speed);
    /* 方位检测 */
    AziDetCov(res_a);
    /* 点迹检测 */
    PlotsDetect(res_a);
    /* 副瓣标记 */
    SidelobePlotDet(res_a);
    /* 点迹发送 */
    PlotNetSend(res_a);
    /* 格子重置 */
    for (int32 idx = 0; idx < pimRangeSamples; idx++) {
        PreDisDetInfo[idx] = CurDisDetInfo[idx];
        CurDisDetInfo[idx] = -1;
    }
    /* 重新赋值 */
    dPrePulseAzi = dCurPulseAzi;

//    if (bIsTestTime)
//    {
//        StaticTimer::tock();
//        double dProTime = StaticTimer::timeDuration();
//        dProTimeAll = dProTimeAll + dProTime;
//        uiProPulseAll++;
//        if ((uiProPulseAll % m_pimAziDim) == 0)
//            std::cout << "PlotDet Process pulse num is " << uiProPulseAll << ", process time all is " << dProTimeAll << std::endl;
//    }
}

/*******************************************************************
*   函数名称：ScanOverNorthJudge
*   功能：   扫描过正北判断
*   输入：
*       无
*   输出：
*       无
*   返回：
*       无
*******************************************************************/
void Plot::ScanOverNorthJudge(NRx8BitPulse *curPulse) {
    dCurPulseAzi = curPulse->vidinfo.azi * 360.0 / 65536.0;
    if (dPrePulseAzi < 0) {
        dPrePulseAzi = dCurPulseAzi;
    }
    if ((dPrePulseAzi > 180.0) && (dCurPulseAzi < 180.0)) {
        // 顺时针过正北判断
        bCWScanNorthSign = true;
        bACWScanNorthSign = false;
    } else if ((dCurPulseAzi > 180.0) && (dPrePulseAzi < 180.0)) {
        // 逆时针过正北判断
        bACWScanNorthSign = true;
        bCWScanNorthSign = false;
    } else {
        bCWScanNorthSign = false;
        bACWScanNorthSign = false;
    }
}

/*******************************************************************
*   函数名称：SidelobePlotDet
*   功能：   副瓣点迹检测
*   输入：
*       无
*   输出：
*       无
*   返回：
*       无
*******************************************************************/
void Plot::SidelobePlotDet(NRx8BitPulse *curPulse) {
    unsigned int uiSectorAziNums = 65536 / TEMP_MAX_SECTOR_NUM;
    usCurSectorId = curPulse->vidinfo.azi / uiSectorAziNums;
    if (usPreSectorId != 255) {
        if (usCurSectorId != usPreSectorId) {
            // 将当前扇区格子存入上一扇区格子，并将当前扇区格子重置
            NRxIf::NRxPlot PlotGridRef;
            for (unsigned int idx = 0; idx < BUFF_MAX_SECTOR_GRID_NUM; idx++) {
                for (unsigned int idx1 = 0; idx1 < MAX_SECTOR_GRID_PLOT_NUM; idx1++) //
                {
                    PreSecPlotBuff[idx][idx1] = CurSecPlotBuff[idx][idx1];
                    CurSecPlotBuff[idx][idx1] = PlotGridRef;
                }
                PreSecGridPlotNum[idx] = CurSecGridPlotNum[idx];
                CurSecGridPlotNum[idx] = 0;
            }
            // 将当前扇区点迹存入当前扇区格子
            unsigned int uiPlotGridIdx(0);
            unsigned int uiPlotGridNum(0);
            for (unsigned int idx = 0; idx < usSectorPlotNum; idx++) {
                uiPlotGridIdx = PlotInfoBuff[idx].dis / plot2DParam.dGridDisWid;
                uiPlotGridNum = CurSecGridPlotNum[uiPlotGridIdx];
                CurSecPlotBuff[uiPlotGridIdx][uiPlotGridNum] = PlotInfoBuff[idx];
                CurSecGridPlotNum[uiPlotGridIdx]++;
            }

            if (!plot2DParam.bSlbeDet_Enable)
                return;

            // 对当前扇区点迹内部进行距离方位副瓣判决
            unsigned int uiCurGridPlotNum(0);
            NRxIf::NRxPlot CurGridPlotInfo;
            double dCurPlotAzi(0.0);
            double dCurPlotDis(0.0);
            double dMaxExtendDis(0.0);
            double dMinExtendDis(0.0);
            unsigned int uiStartGridIdx(0);
            unsigned int uiEndGridIdx(0);
            unsigned int uiFindGridPlotNum(0);
            NRxIf::NRxPlot FindGridPlotInfo;
            double dFindPlotAzi(0.0);
            double dFindPlotDis(0.0);
            for (unsigned int idx = 0; idx < BUFF_MAX_SECTOR_GRID_NUM; idx++) {
                uiCurGridPlotNum = CurSecGridPlotNum[idx];
                for (unsigned int idx1 = 0; idx1 < uiCurGridPlotNum; idx1++) {
                    CurGridPlotInfo = CurSecPlotBuff[idx][idx1];
                    // 判断如果点迹幅度大于一定门限，则认为可能存在副瓣点迹
                    if (CurGridPlotInfo.amp > plot2DParam.iMlobeAmpThr) {
                        // 先找距离副瓣点迹，远距离副瓣（上方）
                        dCurPlotAzi = CurGridPlotInfo.azi * 360.0 / 65535.0;
                        dCurPlotDis = (double) CurGridPlotInfo.dis;
                        dMaxExtendDis = dCurPlotDis + plot2DParam.dDisSlobeDisWid + plot2DParam.dDisSlobeDisRange;
                        dMinExtendDis = dCurPlotDis + plot2DParam.dDisSlobeDisWid - plot2DParam.dDisSlobeDisRange;
                        uiEndGridIdx = dMaxExtendDis / plot2DParam.dGridDisWid;
                        uiEndGridIdx = min(uiEndGridIdx, BUFF_MAX_SECTOR_GRID_NUM - 1);
                        if (dMinExtendDis < 0) {
                            uiStartGridIdx = 0;
                        } else {
                            uiStartGridIdx = dMinExtendDis / plot2DParam.dGridDisWid;
                        }
                        for (unsigned int idx2 = uiStartGridIdx; idx2 <= uiEndGridIdx; idx2++) {
                            uiFindGridPlotNum = CurSecGridPlotNum[idx2];
                            for (unsigned int idx3 = 0; idx3 < uiFindGridPlotNum; idx3++) {
                                FindGridPlotInfo = CurSecPlotBuff[idx2][idx3];
                                dFindPlotAzi = FindGridPlotInfo.azi * 360.0 / 65535.0;
                                dFindPlotDis = (double) FindGridPlotInfo.dis;
                                if (std::abs(dFindPlotAzi - dCurPlotAzi) < plot2DParam.dDisSlobeAziRange) {
                                    if (FindGridPlotInfo.amp <
                                        (CurGridPlotInfo.amp - plot2DParam.iMinMainDisSideRatio)) {
                                        if (FindGridPlotInfo.amp >
                                            (CurGridPlotInfo.amp - plot2DParam.iMaxMainDisSideRatio)) {
                                            if ((dFindPlotDis < dMaxExtendDis) && (dFindPlotDis > dMinExtendDis)) {
                                                CurSecPlotBuff[idx2][idx3].plotRetain = 2;
                                                CurSecPlotBuff[idx2][idx3].plotType = 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // 先找距离副瓣点迹，近距离副瓣（下方）
                        dMaxExtendDis = dCurPlotDis - plot2DParam.dDisSlobeDisWid + plot2DParam.dDisSlobeDisRange;
                        dMinExtendDis = dCurPlotDis - plot2DParam.dDisSlobeDisWid - plot2DParam.dDisSlobeDisRange;
                        if (dMaxExtendDis < 0) {
                            uiEndGridIdx = 0;
                        } else {
                            uiEndGridIdx = dMaxExtendDis / plot2DParam.dGridDisWid;
                        }
                        if (dMinExtendDis < 0) {
                            uiStartGridIdx = 0;
                        } else {
                            uiStartGridIdx = dMinExtendDis / plot2DParam.dGridDisWid;
                        }
                        for (unsigned int idx2 = uiStartGridIdx; idx2 <= uiEndGridIdx; idx2++) {
                            uiFindGridPlotNum = CurSecGridPlotNum[idx2];
                            for (unsigned int idx3 = 0; idx3 < uiFindGridPlotNum; idx3++) {
                                FindGridPlotInfo = CurSecPlotBuff[idx2][idx3];
                                dFindPlotAzi = FindGridPlotInfo.azi * 360.0 / 65535.0;
                                dFindPlotDis = (double) FindGridPlotInfo.dis;
                                if (std::abs(dFindPlotAzi - dCurPlotAzi) < plot2DParam.dDisSlobeAziRange) {
                                    if (FindGridPlotInfo.amp <
                                        (CurGridPlotInfo.amp - plot2DParam.iMinMainDisSideRatio)) {
                                        if (FindGridPlotInfo.amp >
                                            (CurGridPlotInfo.amp - plot2DParam.iMaxMainDisSideRatio)) {
                                            if ((dFindPlotDis < dMaxExtendDis) && (dFindPlotDis > dMinExtendDis)) {
                                                CurSecPlotBuff[idx2][idx3].plotRetain = 2;
                                                CurSecPlotBuff[idx2][idx3].plotType = 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // 在找方位副瓣点迹
                        dMaxExtendDis = dCurPlotDis + plot2DParam.dAziSlobeDisRange;
                        dMinExtendDis = dCurPlotDis - plot2DParam.dAziSlobeDisRange;
                        uiEndGridIdx = dMaxExtendDis / plot2DParam.dGridDisWid;
                        uiEndGridIdx = min(uiEndGridIdx, BUFF_MAX_SECTOR_GRID_NUM - 1);
                        if (dMinExtendDis < 0) {
                            uiStartGridIdx = 0;
                        } else {
                            uiStartGridIdx = dMinExtendDis / plot2DParam.dGridDisWid;
                        }
                        for (unsigned int idx2 = uiStartGridIdx; idx2 <= uiEndGridIdx; idx2++) {
                            uiFindGridPlotNum = CurSecGridPlotNum[idx2];
                            for (unsigned int idx3 = 0; idx3 < uiFindGridPlotNum; idx3++) {
                                FindGridPlotInfo = CurSecPlotBuff[idx2][idx3];
                                dFindPlotAzi = FindGridPlotInfo.azi * 360.0 / 65535.0;
                                dFindPlotDis = (double) FindGridPlotInfo.dis;
                                if (std::abs(dFindPlotAzi - dCurPlotAzi) < plot2DParam.dAziSlobeAziWid) {
                                    if (FindGridPlotInfo.amp <
                                        (CurGridPlotInfo.amp - plot2DParam.iMinMainAziSideRatio_0)) {
                                        if (FindGridPlotInfo.amp >
                                            (CurGridPlotInfo.amp - plot2DParam.iMaxMainAziSideRatio_0)) {
                                            if ((dFindPlotDis < dMaxExtendDis) && (dFindPlotDis > dMinExtendDis)) {
                                                CurSecPlotBuff[idx2][idx3].plotRetain = 2;
                                                CurSecPlotBuff[idx2][idx3].plotType = 2;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // 与上一扇区点迹进行联合判决，先以上一扇区为参考
            for (unsigned int idx = 0; idx < BUFF_MAX_SECTOR_GRID_NUM; idx++) {
                uiCurGridPlotNum = PreSecGridPlotNum[idx];
                for (unsigned int idx1 = 0; idx1 < uiCurGridPlotNum; idx1++) {
                    CurGridPlotInfo = PreSecPlotBuff[idx][idx1];
                    // 判断如果点迹幅度大于一定门限，则认为可能存在副瓣点迹
                    if (CurGridPlotInfo.amp > plot2DParam.iMlobeAmpThr) {
                        // 先找距离副瓣点迹，远距离副瓣（上方）
                        dCurPlotAzi = CurGridPlotInfo.azi * 360.0 / 65535.0;
                        dCurPlotDis = (double) CurGridPlotInfo.dis;
                        dMaxExtendDis = dCurPlotDis + plot2DParam.dDisSlobeDisWid + plot2DParam.dDisSlobeDisRange;
                        dMinExtendDis = dCurPlotDis + plot2DParam.dDisSlobeDisWid - plot2DParam.dDisSlobeDisRange;
                        uiEndGridIdx = dMaxExtendDis / plot2DParam.dGridDisWid;
                        uiEndGridIdx = min(uiEndGridIdx, BUFF_MAX_SECTOR_GRID_NUM - 1);
                        if (dMinExtendDis < 0) {
                            uiStartGridIdx = 0;
                        } else {
                            uiStartGridIdx = dMinExtendDis / plot2DParam.dGridDisWid;
                        }
                        for (unsigned int idx2 = uiStartGridIdx; idx2 <= uiEndGridIdx; idx2++) {
                            uiFindGridPlotNum = CurSecGridPlotNum[idx2];
                            for (unsigned int idx3 = 0; idx3 < uiFindGridPlotNum; idx3++) {
                                FindGridPlotInfo = CurSecPlotBuff[idx2][idx3];
                                dFindPlotAzi = FindGridPlotInfo.azi * 360.0 / 65535.0;
                                dFindPlotDis = (double) FindGridPlotInfo.dis;
                                if (std::abs(dFindPlotAzi - dCurPlotAzi) < plot2DParam.dDisSlobeAziRange) {
                                    if (FindGridPlotInfo.amp <
                                        (CurGridPlotInfo.amp - plot2DParam.iMinMainDisSideRatio)) {
                                        if (FindGridPlotInfo.amp >
                                            (CurGridPlotInfo.amp - plot2DParam.iMaxMainDisSideRatio)) {
                                            if ((dFindPlotDis < dMaxExtendDis) && (dFindPlotDis > dMinExtendDis)) {
                                                CurSecPlotBuff[idx2][idx3].plotRetain = 2;
                                                CurSecPlotBuff[idx2][idx3].plotType = 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // 先找距离副瓣点迹，近距离副瓣（下方）
                        dMaxExtendDis = dCurPlotDis - plot2DParam.dDisSlobeDisWid + plot2DParam.dDisSlobeDisRange;
                        dMinExtendDis = dCurPlotDis - plot2DParam.dDisSlobeDisWid - plot2DParam.dDisSlobeDisRange;
                        if (dMaxExtendDis < 0) {
                            uiEndGridIdx = 0;
                        } else {
                            uiEndGridIdx = dMaxExtendDis / plot2DParam.dGridDisWid;
                        }
                        if (dMinExtendDis < 0) {
                            uiStartGridIdx = 0;
                        } else {
                            uiStartGridIdx = dMinExtendDis / plot2DParam.dGridDisWid;
                        }
                        for (unsigned int idx2 = uiStartGridIdx; idx2 <= uiEndGridIdx; idx2++) {
                            uiFindGridPlotNum = CurSecGridPlotNum[idx2];
                            for (unsigned int idx3 = 0; idx3 < uiFindGridPlotNum; idx3++) {
                                FindGridPlotInfo = CurSecPlotBuff[idx2][idx3];
                                dFindPlotAzi = FindGridPlotInfo.azi * 360.0 / 65535.0;
                                dFindPlotDis = (double) FindGridPlotInfo.dis;
                                if (std::abs(dFindPlotAzi - dCurPlotAzi) < plot2DParam.dDisSlobeAziRange) {
                                    if (FindGridPlotInfo.amp <
                                        (CurGridPlotInfo.amp - plot2DParam.iMinMainDisSideRatio)) {
                                        if (FindGridPlotInfo.amp >
                                            (CurGridPlotInfo.amp - plot2DParam.iMaxMainDisSideRatio)) {
                                            if ((dFindPlotDis < dMaxExtendDis) && (dFindPlotDis > dMinExtendDis)) {
                                                CurSecPlotBuff[idx2][idx3].plotRetain = 2;
                                                CurSecPlotBuff[idx2][idx3].plotType = 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // 在找方位副瓣点迹
                        dMaxExtendDis = dCurPlotDis + plot2DParam.dAziSlobeDisRange;
                        dMinExtendDis = dCurPlotDis - plot2DParam.dAziSlobeDisRange;
                        uiEndGridIdx = dMaxExtendDis / plot2DParam.dGridDisWid;
                        uiEndGridIdx = min(uiEndGridIdx, BUFF_MAX_SECTOR_GRID_NUM - 1);
                        if (dMinExtendDis < 0) {
                            uiStartGridIdx = 0;
                        } else {
                            uiStartGridIdx = dMinExtendDis / plot2DParam.dGridDisWid;
                        }
                        for (unsigned int idx2 = uiStartGridIdx; idx2 <= uiEndGridIdx; idx2++) {
                            uiFindGridPlotNum = CurSecGridPlotNum[idx2];
                            for (unsigned int idx3 = 0; idx3 < uiFindGridPlotNum; idx3++) {
                                FindGridPlotInfo = CurSecPlotBuff[idx2][idx3];
                                dFindPlotAzi = FindGridPlotInfo.azi * 360.0 / 65535.0;
                                dFindPlotDis = (double) FindGridPlotInfo.dis;
                                if (std::abs(dFindPlotAzi - dCurPlotAzi) < plot2DParam.dAziSlobeAziWid) {
                                    if (FindGridPlotInfo.amp <
                                        (CurGridPlotInfo.amp - plot2DParam.iMinMainAziSideRatio_0)) {
                                        if (FindGridPlotInfo.amp >
                                            (CurGridPlotInfo.amp - plot2DParam.iMaxMainAziSideRatio_0)) {
                                            if ((dFindPlotDis < dMaxExtendDis) && (dFindPlotDis > dMinExtendDis)) {
                                                CurSecPlotBuff[idx2][idx3].plotRetain = 2;
                                                CurSecPlotBuff[idx2][idx3].plotType = 2;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // 与上一扇区点迹进行联合判决，后以当前扇区为参考
            for (unsigned int idx = 0; idx < BUFF_MAX_SECTOR_GRID_NUM; idx++) {
                uiCurGridPlotNum = CurSecGridPlotNum[idx];
                for (unsigned int idx1 = 0; idx1 < uiCurGridPlotNum; idx1++) {
                    CurGridPlotInfo = CurSecPlotBuff[idx][idx1];
                    // 判断如果点迹幅度大于一定门限，则认为可能存在副瓣点迹
                    if (CurGridPlotInfo.amp > plot2DParam.iMlobeAmpThr) {
                        // 先找距离副瓣点迹，远距离副瓣（上方）
                        dCurPlotAzi = CurGridPlotInfo.azi * 360.0 / 65535.0;
                        dCurPlotDis = (double) CurGridPlotInfo.dis;
                        dMaxExtendDis = dCurPlotDis + plot2DParam.dDisSlobeDisWid + plot2DParam.dDisSlobeDisRange;
                        dMinExtendDis = dCurPlotDis + plot2DParam.dDisSlobeDisWid - plot2DParam.dDisSlobeDisRange;
                        uiEndGridIdx = dMaxExtendDis / plot2DParam.dGridDisWid;
                        uiEndGridIdx = min(uiEndGridIdx, BUFF_MAX_SECTOR_GRID_NUM - 1);
                        if (dMinExtendDis < 0) {
                            uiStartGridIdx = 0;
                        } else {
                            uiStartGridIdx = dMinExtendDis / plot2DParam.dGridDisWid;
                        }
                        for (unsigned int idx2 = uiStartGridIdx; idx2 <= uiEndGridIdx; idx2++) {
                            uiFindGridPlotNum = PreSecGridPlotNum[idx2];
                            for (unsigned int idx3 = 0; idx3 < uiFindGridPlotNum; idx3++) {
                                FindGridPlotInfo = PreSecPlotBuff[idx2][idx3];
                                dFindPlotAzi = FindGridPlotInfo.azi * 360.0 / 65535.0;
                                dFindPlotDis = (double) FindGridPlotInfo.dis;
                                if (std::abs(dFindPlotAzi - dCurPlotAzi) < plot2DParam.dDisSlobeAziRange) {
                                    if (FindGridPlotInfo.amp <
                                        (CurGridPlotInfo.amp - plot2DParam.iMinMainDisSideRatio)) {
                                        if (FindGridPlotInfo.amp >
                                            (CurGridPlotInfo.amp - plot2DParam.iMaxMainDisSideRatio)) {
                                            if ((dFindPlotDis < dMaxExtendDis) && (dFindPlotDis > dMinExtendDis)) {
                                                PreSecPlotBuff[idx2][idx3].plotRetain = 2;
                                                PreSecPlotBuff[idx2][idx3].plotType = 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // 先找距离副瓣点迹，近距离副瓣（下方）
                        dMaxExtendDis = dCurPlotDis - plot2DParam.dDisSlobeDisWid + plot2DParam.dDisSlobeDisRange;
                        dMinExtendDis = dCurPlotDis - plot2DParam.dDisSlobeDisWid - plot2DParam.dDisSlobeDisRange;
                        if (dMaxExtendDis < 0) {
                            uiEndGridIdx = 0;
                        } else {
                            uiEndGridIdx = dMaxExtendDis / plot2DParam.dGridDisWid;
                        }
                        if (dMinExtendDis < 0) {
                            uiStartGridIdx = 0;
                        } else {
                            uiStartGridIdx = dMinExtendDis / plot2DParam.dGridDisWid;
                        }
                        for (unsigned int idx2 = uiStartGridIdx; idx2 <= uiEndGridIdx; idx2++) {
                            uiFindGridPlotNum = PreSecGridPlotNum[idx2];
                            for (unsigned int idx3 = 0; idx3 < uiFindGridPlotNum; idx3++) {
                                FindGridPlotInfo = PreSecPlotBuff[idx2][idx3];
                                dFindPlotAzi = FindGridPlotInfo.azi * 360.0 / 65535.0;
                                dFindPlotDis = (double) FindGridPlotInfo.dis;
                                if (std::abs(dFindPlotAzi - dCurPlotAzi) < plot2DParam.dDisSlobeAziRange) {
                                    if (FindGridPlotInfo.amp <
                                        (CurGridPlotInfo.amp - plot2DParam.iMinMainDisSideRatio)) {
                                        if (FindGridPlotInfo.amp >
                                            (CurGridPlotInfo.amp - plot2DParam.iMaxMainDisSideRatio)) {
                                            if ((dFindPlotDis < dMaxExtendDis) && (dFindPlotDis > dMinExtendDis)) {
                                                PreSecPlotBuff[idx2][idx3].plotRetain = 2;
                                                PreSecPlotBuff[idx2][idx3].plotType = 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // 在找方位副瓣点迹
                        dMaxExtendDis = dCurPlotDis + plot2DParam.dAziSlobeDisRange;
                        dMinExtendDis = dCurPlotDis - plot2DParam.dAziSlobeDisRange;
                        uiEndGridIdx = dMaxExtendDis / plot2DParam.dGridDisWid;
                        uiEndGridIdx = min(uiEndGridIdx, BUFF_MAX_SECTOR_GRID_NUM - 1);
                        if (dMinExtendDis < 0) {
                            uiStartGridIdx = 0;
                        } else {
                            uiStartGridIdx = dMinExtendDis / plot2DParam.dGridDisWid;
                        }
                        for (unsigned int idx2 = uiStartGridIdx; idx2 <= uiEndGridIdx; idx2++) {
                            uiFindGridPlotNum = PreSecGridPlotNum[idx2];
                            for (unsigned int idx3 = 0; idx3 < uiFindGridPlotNum; idx3++) {
                                FindGridPlotInfo = PreSecPlotBuff[idx2][idx3];
                                dFindPlotAzi = FindGridPlotInfo.azi * 360.0 / 65535.0;
                                dFindPlotDis = (double) FindGridPlotInfo.dis;
                                if (std::abs(dFindPlotAzi - dCurPlotAzi) < plot2DParam.dAziSlobeAziWid) {
                                    if (FindGridPlotInfo.amp <
                                        (CurGridPlotInfo.amp - plot2DParam.iMinMainAziSideRatio_0)) {
                                        if (FindGridPlotInfo.amp >
                                            (CurGridPlotInfo.amp - plot2DParam.iMaxMainAziSideRatio_0)) {
                                            if ((dFindPlotDis < dMaxExtendDis) && (dFindPlotDis > dMinExtendDis)) {
                                                PreSecPlotBuff[idx2][idx3].plotRetain = 2;
                                                PreSecPlotBuff[idx2][idx3].plotType = 2;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

        }
    }
}

/*******************************************************************
*   函数名称：PlotNetSend
*   功能：   点迹网络发送
*   输入：
*       无
*   输出：
*       无
*   返回：
*       无
*******************************************************************/
void Plot::PlotNetSend(NRx8BitPulse *curPulse) {
    uint32 uiSectorAziNums = 65536 / TEMP_MAX_SECTOR_NUM;
    usCurSectorId = curPulse->vidinfo.azi / uiSectorAziNums;
    if (usPreSectorId != 255) {
        if (usCurSectorId != usPreSectorId) {
            // 统计上一格子点迹总数
            uint16 usPreGridPlotNum(0);
            for (uint32 idx = 0; idx < BUFF_MAX_SECTOR_GRID_NUM; idx++) {
                usPreGridPlotNum += PreSecGridPlotNum[idx];
            }

//            if (usPreGridPlotNum > 0)
            {
//                int32 iSecCnt = m_plotDataSet->GetSecCnt();
//                if (iSecCnt < TEMP_MAX_SECTOR_NUM)// 扇区缓存未满
                {
//                    m_plotDataSet->ClearPlots();
                    // 构造NRxIfHeader
                    NRxIfHeader *header = (NRxIfHeader *) m_outbuf;
//                    std::cout << "[header address] " << (void*) header << std::endl;
                    // fill in head
                    header->head = NRxIfHead;
                    header->protocol = NRxProtolVerion;
                    static unsigned short uscounter = 0; // 发送流水号
                    header->counter = uscounter++;
                    time_t curtm;
                    ::time(&curtm); // get current time
                    std::tm *localTime = std::localtime(&curtm);
                    header->time = curtm;

                    timeval now{};
                    gettimeofday(&now, nullptr);
                    header->microSecs = now.tv_usec;

                    //    header->msgBytes; // fill before snd
                    header->tag = NRxIfTag_SectorPlots;
                    header->rdID = 0;
                    header->sendID = 0;
                    header->rcvID = 0;
                    header->cpr = 0; // 自定义时间
                    header->rdrChnID = 0;
                    header->res1 = 0;
                    header->res2 = 0;
                    header->res3 = 0;


                    /* ************************************************************ */
                    // 构造NRxSectorInfo
                    NRxIf::NRxSectorInfo *sechead = (NRxSectorInfo *) (header + 1);
//                    std::cout << "[sechead address] " << (void*) sechead << std::endl;
                    sechead->secNo = presechead.secNo;
                    sechead->secNum = presechead.secNum;
//                    sechead->plotsNum = usPreGridPlotNum;
                    sechead->startAbsTime0 = presechead.startAbsTime0;
                    sechead->startAbsTime1 = presechead.startAbsTime1;
                    sechead->endAbsTime0 = presechead.endAbsTime0;
                    sechead->endAbsTime1 = presechead.endAbsTime1;
                    sechead->startRelTime0 = presechead.startRelTime0;
                    sechead->startRelTime1 = presechead.startRelTime1;
                    sechead->endRelTime0 = presechead.endRelTime0;
                    sechead->endRelTime1 = presechead.endRelTime1;
                    sechead->longitude = presechead.longitude;
                    sechead->latitude = presechead.latitude;
                    sechead->height = presechead.height;
                    sechead->absCourse = presechead.absCourse;
                    sechead->absVel = presechead.absVel;
                    sechead->relCourse = presechead.relCourse;
                    sechead->relVel = presechead.relVel;
                    sechead->headSway = presechead.headSway;
                    sechead->rollSway = presechead.rollSway;
                    sechead->pitchSway = presechead.pitchSway;
                    sechead->scanType = presechead.scanType;
                    sechead->servoScanSpeed = presechead.servoScanSpeed;
                    sechead->servoStartAzi = presechead.servoStartAzi;
                    sechead->servoEndAzi = presechead.servoEndAzi;
//                    m_plotDataSet->SetSectorInfo(sechead);

                    // 构造NRxPlot
                    size_t plot_count = 0;
                    /* ************************************************************ */
                    for (uint32 idx = 0; idx < BUFF_MAX_SECTOR_GRID_NUM; idx++) {
                        for (uint32 idx1 = 0; idx1 < PreSecGridPlotNum[idx]; idx1++) {
                            // if (abs(PreSecPlotBuff[idx][idx1].maxDoppDiff) < 65 ) {
                            //     continue;
                            // }
                            // if (PreSecPlotBuff[idx][idx1].dis < 2700 || PreSecPlotBuff[idx][idx1].dis > 11000) {
                            //     continue;
                            // }
                            // if (PreSecPlotBuff[idx][idx1].azi < 52/360.0*65536 || PreSecPlotBuff[idx][idx1].azi > 168/360.0*65536) {
                            //     continue;
                            // }
                            // if (checkPolyArea(PreSecPlotBuff[idx][idx1].azi, PreSecPlotBuff[idx][idx1].dis)) {
                            //     continue;
                            // }
                            NRxIf::NRxPlot *plotinfo = (NRxPlot *) (sechead + 1) + plot_count;
                            plot_count++;

//                            printf("[plotinfo address %d] ", plot_count);
//                            std::cout << (void*) plotinfo << std::endl;

                            if (curPulse->ifheader.cpr & 0x8)//1, 自定义: 绝对时间1表示0-86399999ms, 绝对时间2无效填0.
                            {
                                plotinfo->absTime0 = PreSecPlotBuff[idx][idx1].absTime0 * 1000 +
                                                     PreSecPlotBuff[idx][idx1].absTime1 / 1000;
                                plotinfo->absTime1 = 0;
                            } else//0, 32位UTC.
                            {
                                plotinfo->absTime0 = PreSecPlotBuff[idx][idx1].absTime0;
                                plotinfo->absTime1 = PreSecPlotBuff[idx][idx1].absTime1;
                            }
                            plotinfo->relTime0 = PreSecPlotBuff[idx][idx1].relTime0;
                            plotinfo->relTime1 = PreSecPlotBuff[idx][idx1].relTime1;
                            //std::cout << "relTime0: " << plotinfo.relTime0 << ", relTime1: " << plotinfo.relTime1 <<std::endl;
                            plotinfo->dis = PreSecPlotBuff[idx][idx1].dis/* * LIGHT_AMEND_COEF*/;
                            plotinfo->azi = PreSecPlotBuff[idx][idx1].azi;

//                            printf("plotinfo dis= %d  plotinfo azi= %d\n", plotinfo->dis, plotinfo->azi);
                            plotinfo->ele = PreSecPlotBuff[idx][idx1].ele;
                            plotinfo->validAttr = PreSecPlotBuff[idx][idx1].validAttr;
                            plotinfo->amp = PreSecPlotBuff[idx][idx1].amp;
                            plotinfo->BaGAmp = PreSecPlotBuff[idx][idx1].BaGAmp;
                            plotinfo->ThrAmp = PreSecPlotBuff[idx][idx1].ThrAmp;
                            plotinfo->SNR = PreSecPlotBuff[idx][idx1].SNR;
                            plotinfo->areaType = PreSecPlotBuff[idx][idx1].areaType;
                            plotinfo->ep = PreSecPlotBuff[idx][idx1].ep;
                            plotinfo->disStart = PreSecPlotBuff[idx][idx1].disStart/* * LIGHT_AMEND_COEF*/;
                            plotinfo->disEnd = PreSecPlotBuff[idx][idx1].disEnd/* * LIGHT_AMEND_COEF*/;
                            plotinfo->aziStart = PreSecPlotBuff[idx][idx1].aziStart;
                            plotinfo->aziEnd = PreSecPlotBuff[idx][idx1].aziEnd;
                            // TODO 判断点迹是否在MTD上下界内，若不在，doppVel和maxDoppDiff填0xFFFF

                            plotinfo->doppVel = PreSecPlotBuff[idx][idx1].doppVel;
                            plotinfo->maxDoppDiff = PreSecPlotBuff[idx][idx1].maxDoppDiff;
                            // plotinfo->amp = PreSecPlotBuff[idx][idx1].maxDoppDiff;
//                            cout << plotinfo->dis << " " << plotinfo->azi << " doppVel:" << PreSecPlotBuff[idx][idx1].doppVel << endl;
                            if (velocityCoalescenceMethod == 0) {
                                plotinfo->amp = abs(PreSecPlotBuff[idx][idx1].doppVel);
                            }
                            else {
                                plotinfo->amp = PreSecPlotBuff[idx][idx1].maxDoppDiff;
                            }
                            /* null ********************************************************
                            plotinfo.saturate = PreSecPlotBuff[idx][idx1].saturate;
                            plotinfo.SNR = PreSecPlotBuff[idx][idx1].SNR;
                            plotinfo.plotType = PreSecPlotBuff[idx][idx1].plotType;
                            plotinfo.plotConfLv = PreSecPlotBuff[idx][idx1].plotConfLv;
                            ************************************************************* */
                            plotinfo->plotRetain = PreSecPlotBuff[idx][idx1].plotRetain;
                            plotinfo->plotType = PreSecPlotBuff[idx][idx1].plotType;
                            plotinfo->plot_id = this->plot_id;
                            this->plot_id++;
                            // 将点迹写入文件
                            // writePlotTxt(plotinfo);

                            plotinfo->HostToNetEndian(*plotinfo);
//                            m_plotDataSet->AddPlots(&plotinfo, 1);
                        }
                    }


                    sechead->plotsNum = plot_count;
                    sechead->HostToNetEndian(*sechead);

                    // 构造NRxIfEnd
                    NRxIfEnd *end = (NRxIfEnd *) ((NRxPlot *) (sechead + 1) + plot_count);
//                    std::cout << "[end address] " << (void*) end << std::endl;
                    end->CRC = 0;
                    end->end1 = NRxIfEnd1;
                    end->end2 = NRxIfEnd2;
                    end->HostToNetEndian(*end);

                    uint16 usMsgBytes = sizeof(NRxIfHeader) + sizeof(NRxSectorInfo) + sizeof(NRxPlot) * plot_count +
                                        sizeof(NRxIfEnd);
                    header->msgBytes = usMsgBytes;

//                    printf("[Plot count] %d\n", plot_count);
//                    printf("[Total bytes] %d\n", header->msgBytes);

                    header->HostToNetEndian(*header);

                    // 发送凝聚后的点迹信息
                    auto sendRes = sendto(localSocket, m_outbuf, usMsgBytes, 0, (sockaddr *)&remotePlotAddr, sizeof(remotePlotAddr));

                    if (sendRes < 0) {
                        std::cerr << "sendto() in Plot failed!" << std::endl;
                    } else {
//                        cout << "Plot send success" << endl;
                    }

                    //printf(">> Plot: send success!\n");

//                    m_plotDataSet->FinishWrite();
                }
            }

            presechead.secNo = (char) usPreSectorId;
            presechead.secNum = TEMP_MAX_SECTOR_NUM;
            presechead.plotsNum = usSectorPlotNum;
            presechead.startAbsTime0 = SecStartAbsTime0;
            presechead.startAbsTime1 = SecStartAbsTime1;
            presechead.endAbsTime0 = SecStopAbsTime0;
            presechead.endAbsTime1 = SecStopAbsTime1;
            presechead.startRelTime0 = SecStartRefTime0;
            presechead.startRelTime1 = SecStartRefTime1;
            presechead.endRelTime0 = SecStopRefTime0;
            presechead.endRelTime1 = SecStopRefTime1;
            presechead.longitude = curPulse->vidinfo.longitude;
            presechead.latitude = curPulse->vidinfo.latitude;
            presechead.height = curPulse->vidinfo.high;
            presechead.absCourse = curPulse->vidinfo.absCourse;
            presechead.absVel = curPulse->vidinfo.absVel;
            presechead.relCourse = curPulse->vidinfo.relCourse;
            presechead.relVel = curPulse->vidinfo.relVel;
            presechead.headSway = curPulse->vidinfo.headSway;
            presechead.rollSway = curPulse->vidinfo.rollSway;
            presechead.pitchSway = curPulse->vidinfo.pitchSway;
            presechead.scanType = curPulse->vidinfo.scanType;
            presechead.servoScanSpeed = curPulse->vidinfo.servoScanSpeed;
            presechead.servoStartAzi = curPulse->vidinfo.servoStartAzi;
            presechead.servoEndAzi = curPulse->vidinfo.servoEndAzi;

            // reset
            usSectorPlotNum = 0;
            SecStartRefTime0 = curPulse->vidinfo.relTime0;
            SecStartRefTime1 = curPulse->vidinfo.relTime1;
            SecStartAbsTime0 = curPulse->vidinfo.absTime0;
            SecStartAbsTime1 = curPulse->vidinfo.absTime1;
        } else {
            SecStopRefTime0 = curPulse->vidinfo.relTime0;
            SecStopRefTime1 = curPulse->vidinfo.relTime1;
            SecStopAbsTime0 = curPulse->vidinfo.absTime0;
            SecStopAbsTime1 = curPulse->vidinfo.absTime1;
        }
    } else {
//        m_plotDataSet->ClearPlots();

        SecStartRefTime0 = curPulse->vidinfo.relTime0;
        SecStartRefTime1 = curPulse->vidinfo.relTime1;
        SecStartAbsTime0 = curPulse->vidinfo.absTime0;
        SecStartAbsTime1 = curPulse->vidinfo.absTime1;
    }
    // 重新赋值
    usPreSectorId = usCurSectorId;
}

/*******************************************************************
*   函数名称：DisDetCov
*   功能：   距离向检测凝聚
*   输入：
*       无
*   输出：
*       无
*   返回：
*       无
*******************************************************************/
void Plot::DisDetCov(NRx8BitPulse *curPulse, NRx8BitPulse *curBaGAmp, NRx8BitPulse *curDetThr, int *speed) {
    /* ************************** 获取当前脉冲参数 ***************************** */
    double dCurPulseTime = (double) (((uint64) (curPulse->vidinfo.relTime0) << 32) + curPulse->vidinfo.relTime1);
    // std::cout << "dCurPulseTime" << dCurPulseTime << std::endl;
    double dCurPulseAbsTime;
    if (curPulse->ifheader.cpr & 0x8)//1, 自定义: 绝对时间1表示0-86399999ms, 绝对时间2无效填0.
    {
        dCurPulseAbsTime = (double) curPulse->vidinfo.absTime0 * 1000;
    } else//0, 32位UTC.
    {
//        time_t tm = curPulse->vidinfo.absTime0;
//        std::tm* localTime = std::localtime(&tm);
//        uint8 hour = localTime->tm_hour;
//        uint8 min = localTime->tm_min;
//        uint8 sec = localTime->tm_sec;
//        dCurPulseAbsTime = ((double)hour * 3600 + (double)min * 60 + (double)sec) * 1000000 +
//                ((double)curPulse->vidinfo.absTime1);
        dCurPulseAbsTime = ((double) curPulse->vidinfo.absTime0) * 1000000 +
                           ((double) curPulse->vidinfo.absTime1);
    }
    // std::cout << "dCurPulseAbsTime" << dCurPulseAbsTime << std::endl;
    double dSamCellSize = 3e8 / curPulse->vidinfo.sampleRate / 2;    // 距离单元大小
    uint32 uiSamCellNum = curPulse->vidinfo.cellNum;

    /* ******************** 建立数组保存检测标识和数据转换值 ********************** */
    for (uint32 idx = 0; idx < uiSamCellNum; idx++) {
        // 门限检测  对数值转换为功率值
        if ((curPulse->data[idx] > plot2DParam.uiMinEchoAmp) && (curPulse->data[idx] <= plot2DParam.uiMaxEchoAmp)) {
            dCurPulseData[idx] = std::pow(10, ((double) (curPulse->data[idx])) * 0.1f);
            // dCurPulseData[idx] = (double)(curPulse->data[idx]);
            usCurPulDetSign[idx] = 1;
        } else {
            dCurPulseData[idx] = 0.0;
            usCurPulDetSign[idx] = 0;
        }
    }

    /* *************************** 定义过程变量 ******************************* */
    sTempPlotBuff CurPlotBuff_Row;      // 当前距离格子
    bool bDisDetStart = false;          // 距离检测起始标识
    uint32 uiStartIdx = 0;              // 距离凝聚起始id
    uint32 uiEndIdx = 0;                // 距离凝聚结束id
    uint32 uiWinDetSum = 0;             // 距离检测窗过门限点数
    uint32 uiSumCellNum = 0;            // 过门限单元计数
    double dSumAmp = 0.0;               // 幅度和
    double dSumTimeAmp = 0.0;           // 时间幅度和
    double dSumAbsTimeAmp = 0.0;        // 时间幅度和
    double dSumAziAmp = 0.0;            // 方位幅度和
    double dSumDisAmp = 0.0;            // 距离幅度和
    double dSumBaGAmp = 0.0;            // 背景幅度和
    double dSumThrAmp = 0.0;            // 门限幅度和
    //速度过程变量
    double dSumSpeed = 0.0;
    double dMaxSpeed = -65536.0;
    double dPowerSum = 0.0;

    uint32 uiDisRowIdx = 0;          // 格子行号
    uint32 uiStartDisRowIdx = 0;     // 格子起始行号
    uint32 uiEndDisRowIdx = 0;       // 格子结束行号

    /* ***************** 初始化：计算第一个检测窗过门限个数 *********************** */
    for (uint32 idx0 = 0; idx0 < plot2DParam.uiDisDet_N; idx0++) {
        uiWinDetSum += usCurPulDetSign[idx0];
    }
    if (uiWinDetSum >= plot2DParam.uiDisDet_M) {
        bDisDetStart = true;    // 标记开始
        // 找开始地址
        for (uint32 idx0 = 0; idx0 < plot2DParam.uiDisDet_N; idx0++) {
            if (usCurPulDetSign[idx0] == 1) {
                uiStartIdx = idx0;
                break;
            }
        }
    }

    /* ******************* 开始滑窗检测 检测抽头线位于窗的尾部 ******************** */
    for (uint32 idx1 = plot2DParam.uiDisDet_N; idx1 < uiSamCellNum; idx1++) {
        // 过门限个数更新，滑动更新，减去老的加上新的
        uiWinDetSum = uiWinDetSum - usCurPulDetSign[idx1 - plot2DParam.uiDisDet_N] + usCurPulDetSign[idx1];
        // 满足距离起始门限 M/N 则起始  同时考虑滑窗至最后一个距离单元仍然满足起始准则
        if ((uiWinDetSum >= plot2DParam.uiDisDet_M) && (idx1 != uiSamCellNum - 1)) {
            if (!bDisDetStart)  // 距离检测未开始，则记下开始，并寻找初始地址序号
            {
                bDisDetStart = true;
                uiStartIdx = idx1 - plot2DParam.uiDisDet_N + 1;   // 赋予初值，防止为空
                for (uint32 idx2 = idx1 - plot2DParam.uiDisDet_N + 1; idx2 <= idx1; idx2++) {
                    if (usCurPulDetSign[idx2] == 1) {
                        uiStartIdx = idx2;
                        break;
                    }
                }
            } else  //距离检测已开始，则继续滑动
            {/*null*/}
        } else   // 不满足距离起始门限 M/N 则结束
        {
            if (bDisDetStart) // 如果已开始，则记下结束，并寻找结束地址
            {
                uiEndIdx = idx1 - plot2DParam.uiDisDet_N;   // 赋予初值，防止为空，考虑到M=1的情况
                for (uint32 idx2 = idx1; idx2 >= idx1 - plot2DParam.uiDisDet_N + 1; idx2--) {
                    if (usCurPulDetSign[idx2] == 1) {
                        uiEndIdx = idx2;
                        break;
                    }
                }
                /* ********************** 距离凝聚处理 **********************   */
                for (uint32 idx3 = uiStartIdx; idx3 <= uiEndIdx; idx3++) {
                    uiSumCellNum++;
                    dSumAmp += dCurPulseData[idx3];
                    dSumTimeAmp += dCurPulseTime * dCurPulseData[idx3];
                    dSumAbsTimeAmp += dCurPulseAbsTime * dCurPulseData[idx3];
                    dSumAziAmp += dCurPulseAzi * dCurPulseData[idx3];
                    dSumDisAmp += ((double) (idx3 + 1)) * dSamCellSize * dCurPulseData[idx3];  // idx3? or idx3+1?
                    // 凝聚速度信息
                    double tmp_speed = speed[idx3] / 100.0; // 质心法
                    dSumSpeed += tmp_speed * dCurPulseData[idx3]; // 求和
                    dPowerSum += dCurPulseData[idx3]; // 功率求和
                    dMaxSpeed = max(dMaxSpeed, abs(tmp_speed)); // 取大

                    if (curBaGAmp != nullptr) {
                        dSumBaGAmp += curBaGAmp->data[idx3];
                    }
                    if (curDetThr != nullptr) {
                        dSumThrAmp += curDetThr->data[idx3];
                    }
                }

                /* *********** 保留功能：计算当前凝聚点的距离格子号 **************** */
                // uiDisRowIdx = ((uint32)(uiStartIdx * dSamCellSize / plot2DParam.dDisGridWid));
                // uiStartDisRowIdx = ((uint32)(uiStartIdx * dSamCellSize / plot2DParam.dDisGridWid));
                // uiEndDisRowIdx = ((uint32)(uiEndIdx * dSamCellSize / plot2DParam.dDisGridWid));
                /* ********************************************************** */

                uiDisRowIdx = uiStartIdx;
                uiStartDisRowIdx = uiStartIdx;
                uiEndDisRowIdx = uiEndIdx;

                // 填写当前点迹格子填充标志
                for (uint32 idx4 = uiStartDisRowIdx; idx4 <= uiEndDisRowIdx; idx4++) {
                    CurDisDetInfo[idx4] = uiDisRowIdx;
                }
                // 将凝聚得到的临时点迹存入对应格子
                CurPlotBuff_Row.bProStart = true;
                CurPlotBuff_Row.bCWScanOverNorth = bCWScanNorthSign;
                CurPlotBuff_Row.bACWScanOverNorth = bACWScanNorthSign;
                CurPlotBuff_Row.ucGridType = 1;
                CurPlotBuff_Row.unGridIdx = uiDisRowIdx;
                CurPlotBuff_Row.unStartGridIdx = uiStartDisRowIdx;
                CurPlotBuff_Row.unEndGridIdx = uiEndDisRowIdx;
                CurPlotBuff_Row.dAmpSum = dSumAmp;
                CurPlotBuff_Row.dTimeAmpSum = dSumTimeAmp;
                CurPlotBuff_Row.dAbsTimeAmpSum = dSumAbsTimeAmp;
                CurPlotBuff_Row.dAziAmpSum = dSumAziAmp;
                CurPlotBuff_Row.dDisAmpSum = dSumDisAmp;
                CurPlotBuff_Row.dBaGAmpSum = dSumBaGAmp;
                CurPlotBuff_Row.dThrAmpSum = dSumThrAmp;
                CurPlotBuff_Row.unSamCellSum = uiSumCellNum;
                CurPlotBuff_Row.unStartAziIdx = curPulse->vidinfo.azi;
                CurPlotBuff_Row.unEndAziIdx = curPulse->vidinfo.azi;
                /* ****************************************************** */
                CurPlotBuff_Row.dStartAzi = dPrePulseAzi;   // 起始方位为上一方位
                if (dPrePulseAzi < 0)   // 程序刚开始运行，上方位不存在
                {
                    CurPlotBuff_Row.dStartAzi = dCurPulseAzi - (360.0 / m_pimAziDim);
                } else if (std::abs(dCurPulseAzi - dPrePulseAzi) > (360.0 / 1024.0))  // 方位突变
                {
                    CurPlotBuff_Row.dStartAzi = dCurPulseAzi - (360.0 / m_pimAziDim);
                }
                /* ****************************************************** */
                CurPlotBuff_Row.dEndAzi = dCurPulseAzi;
                CurPlotBuff_Row.unStartDisIdx = uiStartIdx;
                CurPlotBuff_Row.unEndDisIdx = uiEndIdx;
                CurPlotBuff_Row.dStartDis = ((double) (uiStartIdx)) * dSamCellSize;
                CurPlotBuff_Row.dEndDis = ((double) (uiEndIdx + 1)) * dSamCellSize;

                // TODO 赋值速度信息给当前点迹缓存
                CurPlotBuff_Row.dSumSpeed = dSumSpeed;
                CurPlotBuff_Row.dMaxSpeed = dMaxSpeed;
                CurPlotBuff_Row.dPowerSum = dPowerSum;

                CurPlotBuff[uiDisRowIdx] = CurPlotBuff_Row;

                /* ********************* 过程变量值重置 ************************ */
                dSumAmp = 0.0;
                dSumTimeAmp = 0.0;
                dSumAbsTimeAmp = 0.0;//修改解决绝对时间有误
                dSumAziAmp = 0.0;
                dSumDisAmp = 0.0;
                dSumBaGAmp = 0.0;
                dSumThrAmp = 0.0;
                uiSumCellNum = 0;
                // TODO 速度过程变量重置
                dSumSpeed = 0.0;
                dPowerSum = 0.0;
                dMaxSpeed = 0.0;
                bDisDetStart = false;
                /* ********************************************************** */
            } else // 如果未开始，则继续滑动
            {/* null */}
        }
    }
}

/*******************************************************************
*   函数名称：AziDetCov
*   功能：   方位向检测凝聚
*   输入：
*       无
*   输出：
*       无
*   返回：
*       无
*******************************************************************/
void Plot::AziDetCov(NRx8BitPulse *curPulse) {
    /* ************************* 定义过程变量 ********************************* */
    sTempPlotBuff sPlotBuffRef;               // 参考格子
    sTempPlotBuff CurPlotBuff_idx;            // 当前距离格子
    sTempPlotBuff PrePlotBuff_idx;            // 之前距离格子
    uint32 uiFindConGridSIdx;           // 寻找时的起始格子序号
    uint32 uiFindConGridEIdx;           // 寻找时的结束格子序号
    uint32 uiFindConGridNum;            // 寻找到的连通点数量
    uint32 uiFindConGridIdx_kk;         //
    uint32 uiNewGridIdx;                // 合并后新的行号

    /* ************************** 遍历每个格子 ******************************** */
    /* ！这里是遍历全部格子，也可以遍历当前脉冲的最远距离对应的格子 */
    double dSamCellSize = 3e8 / curPulse->vidinfo.sampleRate / 2;    // 距离单元大小
    uint32 uiSamCellNum = curPulse->vidinfo.cellNum;
    // uint32 uiMaxGridIdx = (uiSamCellNum + 1) * dSamCellSize / plot2DParam.dDisGridWid;
    uint32 uiMaxGridIdx = uiSamCellNum;

    for (uint32 idx = 0; idx < uiMaxGridIdx; idx++) {
        CurPlotBuff_idx = CurPlotBuff[idx];
        if (CurPlotBuff_idx.bProStart) {
            // 如果该当前格子内存在临时点迹，则寻找与之连通的历史格子点迹
            uiFindConGridSIdx = CurPlotBuff_idx.unStartGridIdx;   // 当前临时点迹占据的格子起始序号
            uiFindConGridEIdx = CurPlotBuff_idx.unEndGridIdx;     // 当前临时点迹占据的格子结束序号
            uiFindConGridNum = 0;
            for (uint32 idx1 = uiFindConGridSIdx; idx1 <= uiFindConGridEIdx; idx1++) {
                if (PreDisDetInfo[idx1] >= 0) {
                    if (uiFindConGridNum == 0) {
                        FindConGridIdx[uiFindConGridNum] = PreDisDetInfo[idx1];
                        uiFindConGridNum++;
                    } else {
                        if (PreDisDetInfo[idx1] != FindConGridIdx[uiFindConGridNum - 1]) {
                            FindConGridIdx[uiFindConGridNum] = PreDisDetInfo[idx1];
                            uiFindConGridNum++;
                        }
                    }
                }
            }
            // If find grid, combined grid.
            if (uiFindConGridNum > 0) {
                // combined pregrid.
                for (uint32 idx1 = 0; idx1 < uiFindConGridNum; idx1++) {
                    uiFindConGridIdx_kk = FindConGridIdx[idx1];
                    PrePlotBuff_idx = PrePlotBuff[uiFindConGridIdx_kk];
                    if (PrePlotBuff_idx.bProStart) {
                        CurPlotBuff_idx = GridCombine(CurPlotBuff_idx, PrePlotBuff_idx);
                        PrePlotBuff[uiFindConGridIdx_kk] = sPlotBuffRef;
                    } else {
                        // std::cout << "plot combine error! combine plot num is " << uiFindConGridNum << std::endl;
                    }
                }
                uiNewGridIdx = CurPlotBuff_idx.unGridIdx;
                PrePlotBuff[uiNewGridIdx] = CurPlotBuff_idx;

                for (uint32 idx1 = 0; idx1 < uiFindConGridNum; idx1++) {
                    // 重新标记合并后的序号
                    uiFindConGridIdx_kk = FindConGridIdx[idx1];
                    for (uint32 idx2 = 0; idx2 < uiMaxGridIdx; idx2++)//pimRangeSamples
                    {
                        // 需要优化Find，节省计算量
                        if (PreDisDetInfo[idx2] == uiFindConGridIdx_kk) {
                            PreDisDetInfo[idx2] = uiNewGridIdx;
                        }
                        if (CurDisDetInfo[idx2] == uiFindConGridIdx_kk) {
                            CurDisDetInfo[idx2] = uiNewGridIdx;
                        }
                    }
                }
            } else {
                PrePlotBuff[idx] = CurPlotBuff_idx;
                PrePlotBuff[idx].ucGridType = 0;
            }
        }
        CurPlotBuff[idx] = sPlotBuffRef; //当前格子重置
    }
}

/*******************************************************************
*   函数名称：PlotsDetect
*   功能：   点迹检测
*   输入：
*       无
*   输出：
*       无
*   返回：
*       无
*******************************************************************/
void Plot::PlotsDetect(NRx8BitPulse *curPulse) {
    /* ************************** 定义过程变量 ******************************** */
    sTempPlotBuff sPlotBuffRef;
    sTempPlotBuff PrePlotBuff_idx;
    uint16 usPlotIdx;   // 点迹流水号
    double dPlotTime;      // 点迹时刻
    double dPlotAbsTime;      // 点迹时刻
    double dPlotAzi;       // 点迹方位
    double dPlotDis;       // 点迹距离
    double dPlotAmp;       // 点迹幅度
    double dPlotBaGAmp;       // 点迹背景幅度
    double dPlotThrAmp;       // 点迹门限幅度
    double dPlotCellNum;   // 点迹过门限单元数
    double dPlotAziWid;    // 点迹方位展宽
    double dPlotDisWid;    // 点迹距离展宽
    double dPlotSAzi;      // 点迹开始方位
    double dPlotEAzi;      // 点迹结束方位
    double dPlotSDis;      // 点迹开始距离
    double dPlotEDis;      // 点迹结束距离
    // TODO 新增点迹速度信息、最大最小速度信息
    double dPlotSpeed;
    double dPlotMaxSpeed;
    double dPlotPowerSum;

    double dCurPulseAzi;
    double dCurPulseAzi_1 = (double) (curPulse->vidinfo.azi) * 360.0 / 65536.0;
    double dCurPulseAzi_2 = dCurPulseAzi_1 + 360.0;
    double dSamCellSize = 3e8 / curPulse->vidinfo.sampleRate / 2;    // 距离单元大小
    uint32 uiSamCellNum = curPulse->vidinfo.cellNum;
//    uint32 uiMaxGridIdx = (uiSamCellNum + 1) * dSamCellSize / plot2DParam.dDisGridWid;
    uint32 uiMaxGridIdx = uiSamCellNum;

    /* ************************** 遍历每个格子 ******************************** */
    /* ！这里是遍历全部格子，防止切换探测距离 */
    for (uint32 idx = 0; idx < uiMaxGridIdx; idx++)//pimRangeSamples
    {
        PrePlotBuff_idx = PrePlotBuff[idx];
        if (PrePlotBuff_idx.bProStart) {
            if (bCWScanNorthSign) {
                dCurPulseAzi = dCurPulseAzi_2;
            } else if ((!bCWScanNorthSign) && (PrePlotBuff_idx.bCWScanOverNorth)) {
                dCurPulseAzi = dCurPulseAzi_2;
            } else {
                dCurPulseAzi = dCurPulseAzi_1;
            }
            if (std::abs(dCurPulseAzi - PrePlotBuff_idx.dEndAzi) > 0.1) {
                if (usSectorPlotNum >= MaxPlotInSector) {
                    std::stringstream msg;
                    msg << "warning: current sector plot num over " << MaxPlotInSector;
//                    NRxLogger::logWarning(msg.str());

                    /* 异常信息 */
                    static int32 logInfoReportLevel = 0;
                    if (logInfoReportLevel >= 1) {
                        NRxSWStateInfo swStateInfo;
                        swStateInfo.head.msgBytes = sizeof(NRxSWStateInfo);
                        swStateInfo.head.sendID = NRxSW_ObjDet;
                        swStateInfo.head.rcvID = NRxSW_Disp;
                        swStateInfo.body.softwareID = NRxSW_ObjDet;
                        memcpy(swStateInfo.body.state, msg.str().data(), msg.str().size());
                        swStateInfo.HostToNetEndian(swStateInfo);
                        string dispIP = "239.168.6.189";
//                        string dispIP = getStrParam("RadarDataDistribution", "dispIP");
                        int32 dispHeartPort = 8200;
//                        int32 dispHeartPort = getIntParam("RadarDataDistribution", "dispHeartPort");
//                        udpSender::udpSendMsg(&swStateInfo, sizeof(NRxSWStateInfo), dispIP, dispHeartPort, "heart");
                        auto sendRes = sendto(localSocket, &swStateInfo, sizeof(NRxSWStateInfo), 0, (sockaddr *)&remotePlotAddr, sizeof(remotePlotAddr));

                        if (sendRes < 0) {
                            std::cerr << "sendto() in Plot failed!" << std::endl;
                        }
                    }
                    /************/
                    break;
                }
                // 检测点迹
                usPlotIdx = usSectorPlotNum;
                dPlotTime = PrePlotBuff_idx.dTimeAmpSum / PrePlotBuff_idx.dAmpSum;
                dPlotAbsTime = PrePlotBuff_idx.dAbsTimeAmpSum / PrePlotBuff_idx.dAmpSum;
                dPlotAzi = PrePlotBuff_idx.dAziAmpSum / PrePlotBuff_idx.dAmpSum;

                dPlotDis = PrePlotBuff_idx.dDisAmpSum / PrePlotBuff_idx.dAmpSum;
                dPlotAmp = 10 * std::log10(PrePlotBuff_idx.dAmpSum / PrePlotBuff_idx.unSamCellSum);
                dPlotBaGAmp = PrePlotBuff_idx.dBaGAmpSum / PrePlotBuff_idx.unSamCellSum;
                dPlotThrAmp = PrePlotBuff_idx.dThrAmpSum / PrePlotBuff_idx.unSamCellSum;
                dPlotCellNum = PrePlotBuff_idx.unSamCellSum;
                dPlotAziWid = std::abs(PrePlotBuff_idx.dEndAzi - PrePlotBuff_idx.dStartAzi);
                dPlotDisWid = PrePlotBuff_idx.dEndDis - PrePlotBuff_idx.dStartDis;
                dPlotSAzi = PrePlotBuff_idx.dStartAzi;
                dPlotEAzi = PrePlotBuff_idx.dEndAzi;
                dPlotSDis = PrePlotBuff_idx.dStartDis;
                dPlotEDis = PrePlotBuff_idx.dEndDis;
                dPlotSpeed = PrePlotBuff_idx.dSumSpeed;
                dPlotMaxSpeed = PrePlotBuff_idx.dMaxSpeed;
                dPlotPowerSum = PrePlotBuff_idx.dPowerSum;

                switch (plot2DParam.uiFusionMode) {
                    case 0:  // 质量中心
                        dPlotAzi = PrePlotBuff_idx.dAziAmpSum / PrePlotBuff_idx.dAmpSum;
                        if (dPlotAzi > 360.0) {
                            dPlotAzi = dPlotAzi - 360.0;
                        }
                        dPlotDis = PrePlotBuff_idx.dDisAmpSum / PrePlotBuff_idx.dAmpSum;
                        break;
                    case 1:  // 几何中心
                        dPlotAzi = (dPlotSAzi + dPlotEAzi) / 2;
                        if (dPlotAzi > 360.0) {
                            dPlotAzi = dPlotAzi - 360.0;
                        }
                        dPlotDis = (dPlotSDis + dPlotEDis) / 2;
                        break;
                    default:
                        break;
                }
                if (dPlotEAzi > 360.0) {
                    dPlotEAzi = dPlotEAzi - 360.0;
                }

                // 输出点迹
                if ((dPlotAziWid >= plot2DParam.dMinAziWid) && (dPlotAziWid <= plot2DParam.dMaxAziWid)) {
                    if ((dPlotDisWid >= plot2DParam.dMinDisWid) && (dPlotDisWid <= plot2DParam.dMaxDisWid)) {
                        if ((dPlotCellNum >= plot2DParam.uiMinCellNum) && (dPlotCellNum <= plot2DParam.uiMaxCellNum)) {
                            // double-ms  to  uint64-ms
                            const uint64 maxT32 = 0xFFFFFFFF;
                            uint64 t64ms = (uint64) (dPlotTime);
                            PlotInfoBuff[usSectorPlotNum].relTime0 = ((t64ms >> 32) & maxT32);
                            PlotInfoBuff[usSectorPlotNum].relTime1 = (t64ms & maxT32);

                            //
                            uint32 Time_Seconds = (dPlotAbsTime * 1e-6);
                            uint32 Time_USeconds = (dPlotAbsTime - Time_Seconds * 1e6);
                            PlotInfoBuff[usSectorPlotNum].absTime0 = Time_Seconds;
                            PlotInfoBuff[usSectorPlotNum].absTime1 = Time_USeconds;

                            PlotInfoBuff[usSectorPlotNum].dis = (int) (dPlotDis * LIGHT_AMEND_COEF);
                            PlotInfoBuff[usSectorPlotNum].azi = (dPlotAzi / 360.0) * 65536.f;
                            PlotInfoBuff[usSectorPlotNum].ele = 0;
                            PlotInfoBuff[usSectorPlotNum].validAttr = (0x1) | (0x1 << 1) | (0x1 << 2) | (0x1 << 3);
                            PlotInfoBuff[usSectorPlotNum].amp = (short) dPlotAmp;
                            PlotInfoBuff[usSectorPlotNum].BaGAmp = (short) dPlotBaGAmp;
                            PlotInfoBuff[usSectorPlotNum].ThrAmp = (short) dPlotThrAmp;
//                            std::cout << "BaGAmp is " << PlotInfoBuff[usSectorPlotNum].BaGAmp << std::endl;
//                            std::cout << "ThrAmp is " << PlotInfoBuff[usSectorPlotNum].ThrAmp << std::endl;
                            if (PlotInfoBuff[usSectorPlotNum].BaGAmp > plot2DParam.usNosJugThr) {
                                PlotInfoBuff[usSectorPlotNum].areaType = 2;
                            } else {
                                PlotInfoBuff[usSectorPlotNum].areaType = 1;
                            }
                            //std::cout << "areaType is " << (int)PlotInfoBuff[usSectorPlotNum].areaType << std::endl;
                            // 差值是否会超过8位
                            if (PlotInfoBuff[usSectorPlotNum].amp > PlotInfoBuff[usSectorPlotNum].BaGAmp) {
                                if (PlotInfoBuff[usSectorPlotNum].amp - PlotInfoBuff[usSectorPlotNum].BaGAmp > 255) {
                                    PlotInfoBuff[usSectorPlotNum].SNR = 255;
                                } else {
                                    PlotInfoBuff[usSectorPlotNum].SNR =
                                            PlotInfoBuff[usSectorPlotNum].amp - PlotInfoBuff[usSectorPlotNum].BaGAmp;
                                }
                            } else {
                                PlotInfoBuff[usSectorPlotNum].SNR = 0;
                            }
                            PlotInfoBuff[usSectorPlotNum].ep = (short) dPlotCellNum;
                            PlotInfoBuff[usSectorPlotNum].disStart = (int) (dPlotSDis * LIGHT_AMEND_COEF * 100);
                            PlotInfoBuff[usSectorPlotNum].disEnd = (int) (dPlotEDis * LIGHT_AMEND_COEF * 100);
                            PlotInfoBuff[usSectorPlotNum].aziStart = (dPlotSAzi / 360.0) * 65536.f;
                            PlotInfoBuff[usSectorPlotNum].aziEnd = (dPlotEAzi / 360.0) * 65536.f;
                            /* 缺省始 */
                            PlotInfoBuff[usSectorPlotNum].saturate = 0;
                            // TODO 根据局部变量填写NRxPlot对应的字段，注意单位
                            // cout << PlotInfoBuff[usSectorPlotNum].dis << " " << PlotInfoBuff[usSectorPlotNum].azi / 65536 * 360 << " cal speed:" << dPlotSpeed / dPlotPowerSum * 10 << endl;
                            PlotInfoBuff[usSectorPlotNum].doppVel =  dPlotSpeed / dPlotPowerSum * 100;
                            PlotInfoBuff[usSectorPlotNum].maxDoppDiff = dPlotMaxSpeed * 100.0;

                            PlotInfoBuff[usSectorPlotNum].plotType = 0;
                            PlotInfoBuff[usSectorPlotNum].plotConfLv = 0;
                            /* 缺省止 */
                            usSectorPlotNum++;
                        }
                    }
                }
                // 格子重置
                PrePlotBuff[idx] = sPlotBuffRef;
            }
        }
    }
}

/*******************************************************************
*   函数名称：GridCombine
*   功能：   格子合并
*   输入：
*       plotGrid_1，plotGrid_2
*   输出：
*       无
*   返回：
*       plotGrid_1
*******************************************************************/
sTempPlotBuff Plot::GridCombine(sTempPlotBuff plotGrid_1, sTempPlotBuff plotGrid_2) {
    /* ************************** 顺时针扫描过正北判断 ************************** */
    if ((plotGrid_1.ucGridType == 1) && (plotGrid_2.ucGridType == 0)) {
        // 当前格子与历史格子合并，合并后变成历史格子
        plotGrid_1.ucGridType = 0;
        if ((plotGrid_1.bCWScanOverNorth) && (!plotGrid_2.bCWScanOverNorth)) {
            // 当前格子过正北，历史格子未过正北，合并后历史格子过正北
            // plotGrid_1.bCWScanOverNorth = true;
            plotGrid_1.dStartAzi += 360.0;
            plotGrid_1.dEndAzi += 360.0;
            plotGrid_1.dAziAmpSum += 360.0 * plotGrid_1.dAmpSum;
        } else if ((!plotGrid_1.bCWScanOverNorth) && (plotGrid_2.bCWScanOverNorth)) {
            // 当前格子未过正北，历史格子过正北，合并后历史格子过正北
            plotGrid_1.bCWScanOverNorth = true;
            plotGrid_1.dStartAzi += 360.0;
            plotGrid_1.dEndAzi += 360.0;
            plotGrid_1.dAziAmpSum += 360.0 * plotGrid_1.dAmpSum;
        } else if ((plotGrid_1.bCWScanOverNorth) && (plotGrid_2.bCWScanOverNorth)) {
            // 当前格子过正北，历史格子过正北，合并后历史格子过正北
            // plotGrid_1.bCWScanOverNorth = true;
            plotGrid_1.dStartAzi += 360.0;
            plotGrid_1.dEndAzi += 360.0;
            plotGrid_1.dAziAmpSum += 360.0 * plotGrid_1.dAmpSum;
        } else {
            // 当前格子未过正北，历史格子未过正北，合并后历史格子未过正北
        }
    } else if ((plotGrid_1.ucGridType == 0) && (plotGrid_2.ucGridType == 0)) {
        // 历史格子1与历史格子2合并，合并后变成历史格子
        if ((!plotGrid_1.bCWScanOverNorth) && (plotGrid_2.bCWScanOverNorth)) {
            // 历史格子1未过正北，历史格子2过正北，合并后历史格子过正北
            plotGrid_1.bCWScanOverNorth = true;
        } else {
            // 历史格子1过正北，历史格子2未过正北，合并后历史格子过正北
            // 历史格子1过正北，历史格子2过正北，合并后历史格子过正北
            // 历史格子1未过正北，历史格子2未过正北，合并后历史格子未过正北
        }
    } else {
        // 历史格子与当前格子合并，暂不考虑
        // 当前格子与当前格子合并，暂不考虑
    }

    /* ************************** 逆时针扫描过正北判断 ************************** */
    //
    //
    /* ********************************************************************** */

    /* ***************************** 格子参数合并 ***************************** */
    // plotGrid_1.unGridIdx 不变
    plotGrid_1.dAmpSum += plotGrid_2.dAmpSum;
    plotGrid_1.dAziAmpSum += plotGrid_2.dAziAmpSum;
    plotGrid_1.dDisAmpSum += plotGrid_2.dDisAmpSum;
    plotGrid_1.dBaGAmpSum += plotGrid_2.dBaGAmpSum;
    plotGrid_1.dThrAmpSum += plotGrid_2.dThrAmpSum;
    plotGrid_1.unSamCellSum += plotGrid_2.unSamCellSum;
    plotGrid_1.dTimeAmpSum += plotGrid_2.dTimeAmpSum;
    plotGrid_1.dAbsTimeAmpSum += plotGrid_2.dAbsTimeAmpSum;

    // TODO 合并plotGrid_1和plotGrid_2的速度信息
    plotGrid_1.dSumSpeed += plotGrid_2.dSumSpeed;
    plotGrid_1.dPowerSum += plotGrid_2.dPowerSum;
    plotGrid_1.dMaxSpeed = max(plotGrid_1.dMaxSpeed, plotGrid_2.dMaxSpeed);

    if (plotGrid_1.dStartAzi > plotGrid_2.dStartAzi) {
        plotGrid_1.dStartAzi = plotGrid_2.dStartAzi;
        plotGrid_1.unStartAziIdx = plotGrid_2.unStartAziIdx;
    }
    if (plotGrid_1.dEndAzi < plotGrid_2.dEndAzi) {
        plotGrid_1.dEndAzi = plotGrid_2.dEndAzi;
        plotGrid_1.unEndAziIdx = plotGrid_2.unEndAziIdx;
    }
    if (plotGrid_1.dStartDis > plotGrid_2.dStartDis) {
        plotGrid_1.dStartDis = plotGrid_2.dStartDis;
        plotGrid_1.unStartDisIdx = plotGrid_2.unStartDisIdx;
    }
    if (plotGrid_1.dEndDis < plotGrid_2.dEndDis) {
        plotGrid_1.dEndDis = plotGrid_2.dEndDis;
        plotGrid_1.unEndDisIdx = plotGrid_2.unEndDisIdx;
    }
    if (plotGrid_1.unStartGridIdx > plotGrid_2.unStartGridIdx) {
        plotGrid_1.unStartGridIdx = plotGrid_2.unStartGridIdx;
    }
    if (plotGrid_1.unEndGridIdx < plotGrid_2.unEndGridIdx) {
        plotGrid_1.unEndGridIdx = plotGrid_2.unEndGridIdx;
    }
    return plotGrid_1;
}


void Plot::writePlotTxt(NRxPlot *pPlot) {

    if (!outfile.is_open()) {
        std::cerr << "写入点迹文件失败：文件无法打开。" << std::endl;
        return;
    }

    double absTime0 = pPlot->absTime0 / 1000.0;
    int hour = int(absTime0 / 3600);
    int minute = int((absTime0 - hour * 3600.0) / 60);
    double second = absTime0 - hour * 3600.0 - minute * 60.0;

    double degree = pPlot->azi / 65536.0 * 360.0;

    outfile << hour << ":" << minute << ":" << second << "\t" << absTime0 << "\t" << degree << "\t" << pPlot->dis << pPlot->doppVel << pPlot->maxDoppDiff << "\n";
}

bool Plot::checkPolyArea(uint16 azi, uint32 dis) {

    double ployArea_x1 = 2799 * cos((162.9) * M_PI / 180.0);
    double ployArea_y1 = 2799 * sin((162.9) * M_PI / 180.0);
    double ployArea_x2 = 4111 * cos((127.2) * M_PI / 180.0);
    double ployArea_y2 = 4111 * sin((127.2) * M_PI / 180.0);
    double k12 = (ployArea_y2 - ployArea_y1) / (ployArea_x2 - ployArea_x1);

    double ployArea_x3 = 2701 * cos((162.1) * M_PI / 180.0);
    double ployArea_y3 = 2701 * sin((162.1) * M_PI / 180.0);
    double ployArea_x4 = 4031 * cos((126.0) * M_PI / 180.0);
    double ployArea_y4 = 4031 * sin((126.0) * M_PI / 180.0);
    double k34 = (ployArea_y4 - ployArea_y3) / (ployArea_x4 - ployArea_x3);

    double ployArea_x0 = double(dis) * cos((double(azi) / 65536.0 * 360.0) * M_PI / 180.0);
    double ployArea_y0 = double(dis) * sin((double(azi) / 65536.0 * 360.0) * M_PI / 180.0);
    double k0_1 = (ployArea_y0 - ployArea_y1) / (ployArea_x0 - ployArea_x1);
    double k0_3 = (ployArea_y0 - ployArea_y3) / (ployArea_x0 - ployArea_x3);

    if (ployArea_x0 >= ployArea_x1 && ployArea_x0 <= ployArea_x2) {
        if (k0_1 > k12 && k0_3 < k34) {
            return true;
        }
    }
    return false;
}
