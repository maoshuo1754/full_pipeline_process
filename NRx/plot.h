#ifndef PLOT_H
#define PLOT_H

#include <include/NRxPIM.h>
#include <include/NRxType.h>
#include <include/NRxObj.h>
#include <vector>
#include <sys/time.h>

static const int32 MaxPlotInSector                (NRxMaxPlotsNum);

static const unsigned int BUFF_MAX_SECTOR_GRID_NUM (512);           // 副瓣点迹判决缓存方位扇区格子数
static const unsigned int MAX_SECTOR_GRID_PLOT_NUM (256);            // 副瓣点迹判决缓存方位扇区格子点迹数

/* 临时点迹缓存结构体 */
typedef struct sTempPlotBuff
{
    bool bProStart;             // 凝聚开始标志
    bool bCWScanOverNorth;      // 顺时针扫描过正北标识
    bool bACWScanOverNorth;     // 逆时针扫描过正北标识
    uint8 ucGridType;     // 格子类型，0：历史格子；1：当前格子
    uint32 unGridIdx;     // 格子行号
    uint32 unStartGridIdx; // 起始格子号
    uint32 unEndGridIdx;   // 结束格子号
    double dAmpSum;             // 幅度和
    double dTimeAmpSum;         // 时间幅度和(参考)
    double dAbsTimeAmpSum;      // 时间幅度和(参考)
    double dAziAmpSum;          // 方位幅度和
    double dDisAmpSum;          // 距离幅度和
    double dVelAmpSum;          // 速度幅度和
    double dBaGAmpSum;          // 背景幅度和
    double dThrAmpSum;          // 门限幅度和
    // TODO 新增速度过程变量
    double dSumSpeed; // 速度和
    double dMaxSpeed; // 最大速度
    double dPowerSum; // 幅度和
    double dMaxPower; // 最大幅度
    uint32 unSamCellSum;  // 过门限单元数
    uint32 unPulseSum;    // 脉冲计数
    uint32 unStartAziIdx; // 起始方位码
    uint32 unEndAziIdx;   // 终止方位码
    double dStartAzi;           // 起始方位
    double dEndAzi;             // 终止方位
    uint32 unStartDisIdx; // 起始距离码
    uint32 unEndDisIdx;   // 终止距离码
    double dStartDis;           // 起始距离
    double dEndDis;             // 终止距离
    double dMaxVel;             // 最大速度
    double dMinVel;             // 最小速度
    double dPrePulseAzi;        // 上一脉冲方位
    double dCurPulseAzi;        // 当前脉冲方位
    sTempPlotBuff()
    {
        bProStart = false;
        bCWScanOverNorth = false;
        bACWScanOverNorth = false;
        ucGridType = 0;
        unGridIdx = 0;
        unStartGridIdx = 0;
        unEndGridIdx = 0;
        dAmpSum = 0.0;
        dTimeAmpSum = 0.0;
        dAbsTimeAmpSum = 0.0;
        dAziAmpSum = 0.0;
        dDisAmpSum = 0.0;
        dVelAmpSum = 0.0;
        // TODO 新增速度过程变量
        dSumSpeed = 0.0;
        dMaxSpeed = -1.0;
        dMaxPower = -1.0;
        dPowerSum = -1.0;
        unSamCellSum = 0;
        unPulseSum = 0;
        unStartAziIdx = 0;
        unEndAziIdx = 0;
        dStartAzi = 0.0;
        dEndAzi = 0.0;
        unStartDisIdx = 0;
        unEndDisIdx = 0;
        dStartDis = 0.0;
        dEndDis = 0.0;
        dMaxVel = 0.0;
        dMinVel = 0.0;
        dPrePulseAzi = 0.0;
        dCurPulseAzi = 0.0;
    }
}sTempPlotBuff;

/* 点迹提取参数结构体 */
typedef struct sPlot2DParam
{
    bool bPlotDet_Enable;
    uint32 uiFusionMode;       // 凝聚方式：0质心 1几何中心
    // double dDisGridWid;         // 保留参数，距离格子宽度（米），为了节约计算量，可以粗略画格子
    uint32 uiMinEchoAmp;  // 最小回波幅度
    uint32 uiMaxEchoAmp;  // 最大回波幅度
    uint32 uiDisDet_M;  // 距离检测，M
    uint32 uiDisDet_N;  // 距离检测，N
    double dMinAziWid;        // 点迹最小方位展宽
    double dMaxAziWid;        // 点迹最大方位展宽
    double dMinDisWid;        // 点迹最小距离展宽
    double dMaxDisWid;        // 点迹最大距离展宽
    uint32 uiMinCellNum;  // 点迹最小距离单元数
    uint32 uiMaxCellNum;  // 点迹最大距离单元数
    uint16 usNosJugThr;   // 噪声点迹幅度判决门限

    bool bSlbeDet_Enable;
    double dGridAziWid;     // 格子方位宽度
    double dGridDisWid;     // 格子距离宽度
    int iMlobeAmpThr;       // 大于该门限的认为可能存在副瓣点迹
    double dDisSlobeDisWid;     // 距离副瓣距离延伸范围（副瓣点迹一般位于上下终点）
    double dDisSlobeDisRange;   // 距离副瓣距离波动范围，米，以上下延伸的终点为中心
    double dDisSlobeAziRange;   // 距离副瓣方位波动范围，度，以上下延伸的终点为中心
    int iMinMainDisSideRatio;   // 距离主副比下限
    int iMaxMainDisSideRatio;   // 距离主副比上限

    double dAziSlobeAziWid;         // 方位副瓣方位延伸范围（延伸范围内皆有可能）
    double dAziSlobeDisRange;       // 方位副瓣距离波动方位
    double dFirstAziSlobePos;       // 第1方位副瓣位置
    double dSecondAziSlobePos;      // 第2方位副瓣位置
    int iMinMainAziSideRatio_0;     // 方位主副比下限(0~2deg)
    int iMaxMainAziSideRatio_0;     // 方位主副比上限(0~2deg)
    int iMinMainAziSideRatio_1;     // 方位主副比下限(2~6deg)
    int iMaxMainAziSideRatio_1;     // 方位主副比上限(2~6deg)
    int iMinMainAziSideRatio_2;     // 方位主副比下限(>6deg)
    int iMaxMainAziSideRatio_2;     // 方位主副比上限(>6deg)
} sPlot2DParam;

class Plot
{
public:
    Plot();
    ~Plot();

    void MainFun(char *dataBuf, unsigned int dataSize, int *speedChannels, float* azi_arr);

    void XX92NRx8bit(char *xx9buf);
    void PlotConv(NRx8BitPulse* res_a, int *speed, size_t speedLength, float* azi_arr);

    void setMTDRange(int low, int high);
    void setSocket(int socket, sockaddr_in addr);

private:
    void ScanOverNorthJudge(NRx8BitPulse*);

    bool checkPolyArea(uint16 azi, uint32 dis);

    void PlotNetSend(NRx8BitPulse*);
    void DisDetCov(NRx8BitPulse *curPulse, NRx8BitPulse *curBaGAmp, NRx8BitPulse *curDetThr, int *speed);
    void AziDetCov(NRx8BitPulse*);
    void PlotsDetect(NRx8BitPulse*);

    sTempPlotBuff GridCombine(sTempPlotBuff, sTempPlotBuff);

private:
    sPlot2DParam plot2DParam;

    int32 m_pimAziDim;
    int32 pimRangeSamples;

    sTempPlotBuff* PrePlotBuff;   // 历史点迹格子
    sTempPlotBuff* CurPlotBuff;   // 当前点迹格子

    int32* PreDisDetInfo;  // 存储历史点迹格子填充标志
    int32* CurDisDetInfo;  // 存储当前点迹格子填充标志
    int32* FindConGridIdx;  // 存储连通格子序号

    NRxIf::NRxPlot* PlotInfoBuff;

    uint32 SecStartRefTime0;
    uint32 SecStartRefTime1;
    uint32 SecStopRefTime0;
    uint32 SecStopRefTime1;
    uint32 SecStartAbsTime0;
    uint32 SecStartAbsTime1;
    uint32 SecStopAbsTime0;
    uint32 SecStopAbsTime1;

    uint16 usSectorPlotNum;
    uint16 usPreSectorId;
    uint16 usCurSectorId;

    double dPrePulseAzi;
    double dCurPulseAzi;
    bool bCWScanNorthSign;     // 顺时针过正北标志
    bool bACWScanNorthSign;    // 逆时针过正北标志

    uint16* usCurPulDetSign;   // !!!!!!!!!!!!!!!!!!!!!
    double* dCurPulseData;
    NRx8BitPulse* nrx8bitPulse;
    char* m_nrx8bitBuf;
    char* m_outbuf;

    int mtdLowBound;    //MTD处理的距离下限, 单位：m
    int mtdHighBound;   //MTD处理的距离上限, 单位：m
    unsigned int sample_rate; // 采样率，20MHz一个距离单元是7.5m, 5MHz是30m
    double oneScanTime; //天线扫描周期（s）

    bool useClutterMap; //杂波图开关
    std::vector<char*> clutterMap; //杂波图缓存
    struct timeval tv;
    int localSocket;
    sockaddr_in remotePlotAddr;


private:
    void SidelobePlotDet(NRx8BitPulse*);

    // 副瓣点迹判决所用
    std::vector< std::vector< NRxIf::NRxPlot > > CurSecPlotBuff;        // 当前扇区点迹缓存
    std::vector< std::vector< NRxIf::NRxPlot > > PreSecPlotBuff;        // 上一扇区点迹缓存
    std::vector< uint32 > CurSecGridPlotNum;       // 当前扇区点迹缓存每个格子中的点迹个数
    std::vector< uint32 > PreSecGridPlotNum;       // 上一扇区点迹缓存每个格子中的点迹个数

    NRxIf::NRxSectorInfo presechead;

    std::ofstream outfile;
    void writePlotTxt(NRxPlot *pPlot);

    int plot_id;


};

#endif // PLOT_H
