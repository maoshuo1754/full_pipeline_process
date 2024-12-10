// stdafx.cpp : 只包括标准包含文件的源文件
// PcieTst.pch 将作为预编译头
// stdafx.obj 将包含预编译类型信息

#include "stdafx.h"
#include "math.h"
#include "myIpp.h"
#include "afxpriv.h"
#include "RspProcess.h"

volatile  BOOL g_bCaiji;
volatile UINT g_bCaijiEnable;
//20241121
volatile UINT g_bOneBeamCaijiEnable;

BYTE *g_pOneBeamCaijiBuf = NULL;
HANDLE g_hCaijiOneBeamBufReadyEvent = INVALID_HANDLE_VALUE;
CJPULSEGROUPDESC stCJRadarSigProcePlsGrpData;

//RadarSigProcPulseGroupData stRadarSigProcePlsGrpData[MAX_BEAM];
S4BPULSEGROUPDESC g_S4BPULSEGROUPDESC[8];

Ipp32f g_fNCI_coe[MAX_PULSE_NUMBER_IN_GROUP];
Ipp32f g_fMTDNCI_coe[MAX_PULSE_NUMBER_IN_GROUP];
Ipp32f g_fTempZhongduanData[MAX_BEAM][MAX_DISTANCE_ELEMENT_NUMBER];				//每个波束每个脉冲不超过1000点

Ipp32f g_fTempHuiboData_Beam0[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam1[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam2[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam3[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam4[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam5[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam6[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam7[MAX_DISTANCE_ELEMENT_NUMBER];

Ipp32f g_fTempHuiboData_Beam8[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam9[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam10[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam11[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam12[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam13[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam14[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam15[MAX_DISTANCE_ELEMENT_NUMBER];

Ipp32f g_fTempHuiboData_Beam16[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam17[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam18[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam19[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam20[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam21[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam22[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam23[MAX_DISTANCE_ELEMENT_NUMBER];

Ipp32f g_fTempHuiboData_Beam24[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam25[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam26[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam27[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam28[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam29[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam30[MAX_DISTANCE_ELEMENT_NUMBER];
Ipp32f g_fTempHuiboData_Beam31[MAX_DISTANCE_ELEMENT_NUMBER];
////////////////////////////////////////////////////////////
//Ipp32f fIQData_Maiya_DRM_Real[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER];		//
//Ipp32f fIQData_Maiya_DRM_Imag[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER];		//
//Ipp32f fIQData_Maiya_RDM_Real[MAX_DISTANCE_ELEMENT_NUMBER][MAX_PULSE_NUMBER_IN_GROUP];		//
//Ipp32f fIQData_Maiya_RDM_Imag[MAX_DISTANCE_ELEMENT_NUMBER][MAX_PULSE_NUMBER_IN_GROUP];		//

////多普勒算法
//Ipp32f m_fDoppParamTable[WN_FFT];
//Ipp32fc sWComplex[WN_FFT];
//Ipp32f dataABS_tmp[MAX_RNG_CELL][FIR_Num];
//Ipp32f dataABS[MAX_RNG_CELL][FIR_Num];
//Ipp32f dataABSNEW[MAX_RNG_CELL][FIR_Num];
//Ipp32f dataABS_Trans[MAX_RNG_CELL][FIR_Num];
//int MaxIn_dex[MAX_RNG_CELL];
//int MaxIn_dex_1[MAX_RNG_CELL];
//int MaxIn_dex_2[MAX_RNG_CELL];
//Ipp32fc sWkComplex[FIR_Num];
//Ipp32fc	FiComplex[FIR_Num];
//Ipp32f EstDelt0[MAX_RNG_CELL];
//Ipp32f EstDelt0_tmp[MAX_RNG_CELL];
//Ipp32f EstDelt1[MAX_RNG_CELL];
//Ipp32f EstDelt2[MAX_RNG_CELL];
//Ipp32f EstDelt1_tmp[MAX_RNG_CELL];
//Ipp32f EstDelt2_tmp[MAX_RNG_CELL];
//Ipp32f EstDelt0_pi[MAX_RNG_CELL];
//Ipp32f param_index[MAX_RNG_CELL];
//Ipp32f param_coe;
////    Ipp32f SigAmpl0[MAX_RNG_CELL];
////    Ipp32f SigFreq0[MAX_RNG_CELL];
//Ipp32f SigPhas0[MAX_RNG_CELL];
//Ipp32f SigPhas0_cos[MAX_RNG_CELL];
//Ipp32f SigPhas0_sin[MAX_RNG_CELL];
//Ipp32fc V1Complex[MAX_RNG_CELL];
//Ipp32f V1Complex_re[MAX_RNG_CELL];
//Ipp32f V1Complex_im[MAX_RNG_CELL];
////    Ipp32f SigAmpl1[MAX_RNG_CELL];
////    Ipp32f SigFreq1[MAX_RNG_CELL];
//
//Ipp32f result[2][MAX_RNG_CELL];

//void DopplerProcess(Ipp32fc *FFTwn, Ipp32f wnsum, int MTD_Channel, UINT DataLen, Ipp32f *SigAmpl0, Ipp32f *SigFreq0, Ipp16s *Vaild_Flag0, Ipp32f *SigAmpl1, Ipp32f *SigFreq1, Ipp16s *Vaild_Flag1, Ipp32f *SigAmpl2, Ipp32f *SigFreq2, Ipp16s *Vaild_Flag2);
//void DopplerDataSel(Ipp32f *Ipp32fSigAmpl0, Ipp32f *Ipp32fSigFreq0, Ipp16s *Vaild_Flag0, Ipp32f *Ipp32fSigAmpl1, Ipp32f *Ipp32fSigFreq1, Ipp16s *Vaild_Flag1, Ipp32f *Ipp32fSigAmpl2, Ipp32f *Ipp32fSigFreq2, Ipp16s *Vaild_Flag2, int DataLen, Ipp32f *Ipp32fSigAmp, Ipp32f *Ipp32fSigFreq);
//void CFAR_Exci(Ipp32f *Ipp32f_Data, Ipp32f *Ipp32f_Linear_Data, Ipp32f *Ipp32f_DataCfar, UINT DataLen, UINT Naverage_Exci, UINT Nprotect, UINT NumTemp_Exci, UINT Naverage, UINT cfar_Sym, UINT cfar_xishu_Exci);

/////////////////////////////////////////////////////////////////

PULSEGROUPDESC g_stuPulseGrpDesc;

PULSEGROUPDESC ActiveRadarDSPThread_stuPulseGrpDesc;


DMAREADINFO targetsim_dmareadinfo;

int g_bk_desc_total_number_final = 0;

int  g_nFreqPoint=0;
double bochang0;

Ipp32f Caiji_Buf[4096 * 128] = { 0 };
Ipp32fc IQCaiji_Buf[4096 * 128];
Ipp32fc IQCaiji_Buf_1[4096 * 128];


Ipp32f MTI_Coe[1024] = {
	-682, -56, -56, -56, -55, -53, -50, -45, -40,
	-33, -25, -16, -6, 5, 18, 31, 46, 61,
	77, 93, 111, 128, 146, 163, 181, 197, 214,
	230, 244, 258, 270, 280, 288, 294, 298, 299,
	297, 292, 284, 271, 255, 235, 209, 179, 144,
	102, 55, 1, -61, -132, -212, -303, -408, -529,
	-670, -838, -1043, -1299, -1635, -2102, -2811, -4051, -6881,
	-20836, 20836, 6881, 4051, 2811, 2102, 1635, 1299, 1043,
	838, 670, 529, 408, 303, 212, 132, 61, -1,
	-55, -102, -144, -179, -209, -235, -255, -271, -284,
	-292, -297, -299, -298, -294, -288, -280, -270, -258,
	-244, -230, -214, -197, -181, -163, -146, -128, -111,
	-93, -77, -61, -46, -31, -18, -5, 6, 16,
	25, 33, 40, 45, 50, 53, 55, 56, 56,
	56, 682
};



UINT DMACaiji(LPVOID pParam)
{
	THREADPARAMS *ptp=(THREADPARAMS *)pParam;
	HWND hWnd=ptp->hWnd;
	HANDLE hPcieHandle=ptp->hPcieHandle;
	UINT nBokongEnable=ptp->nBokongEnable;
	HANDLE hEvent=ptp->hEvent;
	HANDLE hCaijiBufReadyEvent=ptp->hCaijiBufReadyEvent;
	BYTE *pCaijiBuf=ptp->pBuf;
	delete ptp;

	/////////////////////////////////////////
	DWORD nReturn = 0, nWritten=0;
	BOOL b;

	MSGPARAMS msgParam[1024];
	int nmsgParamCount=0;


	ULONG nDMADir=1;	//Caiji

	IO_READ_WRITE_DATA theIOData;
	IO_READ_WRITE_DATA pIOData;
	//pIOData=new IO_READ_WRITE_DATA;
	ZeroMemory(&pIOData,sizeof(IO_READ_WRITE_DATA));

	UINT nCaijiPacketcount=0;

	//BYTE *pBuf=NULL;

	////pBuf=new BYTE[BLOCKCOUNT*RAM_SIZE];
	//pBuf=new BYTE[RAM_SIZE];		//fot test
	//if(pBuf==NULL)
	//{
	//	AfxMessageBox("Cannot allocate memory");
	//	return;
	//}
	BYTE pBuf[RAM_SIZE];

	ZeroMemory(pBuf,BLOCKCOUNT*RAM_SIZE);
	UINT ii=0,kk=0;;
	ULONG64 nOffset=0;

	LARGE_INTEGER nTimeBegin,nTimeStop,nFreq;
	if(!QueryPerformanceFrequency(&nFreq))
	{
		AfxMessageBox(_T("Failed to set timer,now exit!"));
		return 1;
	}
	//TRACE("	:%d\n",nFreq.QuadPart);
///////////////////////
	//设置分频系数
	theIOData.nMode=2;	//字节写
	theIOData.nAddr=3;
	theIOData.nData=0x00;	//自检模式
	b=DeviceIoControl(hPcieHandle,IOCTL_MEM0_WRITE,&theIOData,sizeof(IO_READ_WRITE_DATA),&pIOData,sizeof(IO_READ_WRITE_DATA),&nReturn,NULL);

	//设置分频系数
	theIOData.nMode=2;	//字节写
	theIOData.nAddr=1;
	theIOData.nData=10;	//分频系数的分母

	b=DeviceIoControl(hPcieHandle,IOCTL_MEM0_WRITE,&theIOData,sizeof(IO_READ_WRITE_DATA),&pIOData,sizeof(IO_READ_WRITE_DATA),&nReturn,NULL);

	theIOData.nMode=2;	//字节写
	theIOData.nAddr=2;
	theIOData.nData=8;	//分频系数的分子
	b=DeviceIoControl(hPcieHandle,IOCTL_MEM0_WRITE,&theIOData,sizeof(IO_READ_WRITE_DATA),&pIOData,sizeof(IO_READ_WRITE_DATA),&nReturn,NULL);
	/////////////////////////////////////////////////////////////////////////
	//theIOData.nMode = 0;	//32bits写
	//theIOData.nAddr = 11 * 4;
	//theIOData.nData = 0x00000100;	//	设置为外监测模式，模拟目标
	//b = DeviceIoControl(hPcieHandle, IOCTL_MEM0_WRITE, &theIOData, sizeof(IO_READ_WRITE_DATA), &pIOData, sizeof(IO_READ_WRITE_DATA), &nReturn, NULL);
	///////////////////////////////////////////////////////////////////////
	theIOData.nMode = 0;	//32bits写
	theIOData.nAddr = 11 * 4;
	theIOData.nData = 0x02000000;	//	设置为AuroraFront模式 20240722
	b = DeviceIoControl(hPcieHandle, IOCTL_MEM0_WRITE, &theIOData, sizeof(IO_READ_WRITE_DATA), &pIOData, sizeof(IO_READ_WRITE_DATA), &nReturn, NULL);

	if (!b)
	{
		AfxMessageBox(_T("bkThead:mem0 reg write Failed"));
	}
	/////////////////////////////////////////////////////////////////////
	//20200507 lvwx
	theIOData.nMode = 0;	//32bits写
	theIOData.nAddr = 9*4;
	//theIOData.nData = 0x0E;	//每100ms增加1个  控制开关发射  bit3 mtp sel =1 drm,bit2 tx en,bit 1 array data en,bit 0 sim data en
	//theIOData.nData = 0x0E;	//每100ms增加1个  控制开关发射  bit3 mtp sel =1 drm,bit2 tx en,bit 1 array data en,bit 0 sim data en

	//theIOData.nData = 0x00010001;	//每100ms增加1个  控制开关发射  bit3 mtp sel =1 drm,bit2 tx en,bit 1 array data en,bit 0 sim data en  20210907		for test
	//theIOData.nData = 0x00011000;	//每100ms增加1个  控制开关发射  bit3 mtp sel =1 drm,bit2 tx en,bit 1 array data en,bit 0 sim data en  20210907		for test
	//theIOData.nData = 0x00000001;	//每100ms增加1个  控制开关发射  bit3 mtp sel =1 drm,bit2 tx en,bit 1 array data en,bit 0 sim data en  20220705
	
//	theIOData.nData = 0x0038000A;// 0x0088000E;// 0x0008000B;// 0x0008000E;	//每100ms增加1个  控制开关发射  打开接收通道,截位6
	//theIOData.nData = 0x00880006;// 0x0088000E;// 0x0008000B;// 0x0008000E;	//每100ms增加1个  控制开关发射  打开接收通道,截位6
	//theIOData.nData = 0x8000000A;	//每100ms增加1个  控制开关发射  打开接收通道,截位6 接收模式
	//theIOData.nData = 0x0088000A;// 0x0089000E;	//每100ms增加1个  控制开关发射  打开接收通道,截位6
 	theIOData.nData = 0x0038000E;// 0x0008000E;	// 0x0008000E;	//每100ms增加1个  控制开关发射  打开接收通道,截位6  23:20 和差波束截尾 4:15:0;3:16:1;2:17:2;1:18:3;0:19:4  19:16 FFT截位  27：24 宽带截位
	//theIOData.nData = 0x0008000D;	//每100ms增加1个  控制开关发射  打开接收通道,截位6
	//theIOData.nData = 0x00080000;	//每100ms增加1个  控制开关发射  打开接收通道,截位6
	
	b = DeviceIoControl(hPcieHandle, IOCTL_MEM0_WRITE, &theIOData, sizeof(IO_READ_WRITE_DATA), &pIOData, sizeof(IO_READ_WRITE_DATA), &nReturn, NULL);
	if (!b)
	{
		AfxMessageBox(_T("bkThead:mem0 reg write Failed"));
	}
	///////////////////////////////////////////////////////////////////////////
	theIOData.nMode = 0;	//32bits写
	theIOData.nAddr = 10 * 4;

	theIOData.nData = 0x04;	//Not used 20231130
	b = DeviceIoControl(hPcieHandle, IOCTL_MEM0_WRITE, &theIOData, sizeof(IO_READ_WRITE_DATA), &pIOData, sizeof(IO_READ_WRITE_DATA), &nReturn, NULL);
	if (!b)
	{
		AfxMessageBox(_T("bkThead:mem0 reg write Failed"));
	}

	/////////////////////////////////////////
	hActiveRadarDSPPipeRead=INVALID_HANDLE_VALUE;
	hActiveRadarDSPPipeWrite = INVALID_HANDLE_VALUE;

	hRadarSigProcPipeRead = INVALID_HANDLE_VALUE;
	hRadarSigProcPipeWrite = INVALID_HANDLE_VALUE;

	hTerminalPipeRead = INVALID_HANDLE_VALUE;
	hTerminalPipeWrite = INVALID_HANDLE_VALUE;

	int nRet = InitializeActiveRadarDSPPipe();
	
	if (nRet)
	{
		AfxBeginThread(ActiveRadarDSPThread, NULL,0,16*1024*1024);

	}
	/////////////////////////////////////////
	//启动DMA
	if(nBokongEnable)
	{
		nDMADir|=0x80000000;
	}
	QueryPerformanceCounter(&nTimeBegin);
	b=DeviceIoControl(hPcieHandle,IOCTL_DMA_CTRL,&nDMADir,sizeof(ULONG),NULL,0,&nReturn,NULL);

	//for(ii=0;ii<BLOCKCOUNT;ii++)
	//while(ii<BLOCKCOUNT)
	while(g_bCaiji)
	{
		WaitForSingleObject(hEvent, INFINITE);
		//b=ReadFile(m_hPcieHandle,pBuf+nOffset,RAM_SIZE,&nReturn,NULL);
		b=ReadFile(hPcieHandle,pBuf,RAM_SIZE,&nReturn,NULL);		//for test
		if(nReturn>0)
		{
			//TRACE("Read size:%d\n", nReturn);
			//////////////////////////////////////////////
			if (hActiveRadarDSPPipeWrite)
			{
				WriteFile(hActiveRadarDSPPipeWrite, pBuf, nReturn, &nWritten, NULL);		//实际是数据包找头
				if (nWritten != nReturn)
				{
					TRACE("Error in Write:%d\n", nWritten);
				}
			}
			//////////////////////////////////////////////
			ii++;
			if(ii%DISPLAYCOUNT==0)
			{
				//TRACE("%d:%d\n",ii,nReturn);

				QueryPerformanceCounter(&nTimeStop);

				msgParam[nmsgParamCount].nPacketCount=ii;
				msgParam[nmsgParamCount].nTimeDiff=nTimeStop.QuadPart-nTimeBegin.QuadPart;
				msgParam[nmsgParamCount].nFreq=nFreq.QuadPart;

				//TRACE("%lld,%lld,%lld\n",msgParam[nmsgParamCount].nPacketCount,msgParam[nmsgParamCount].nTimeDiff,msgParam[nmsgParamCount].nFreq);

				PostMessage(hWnd,ID_MESSAGE_CAIJI,WPARAM(msgParam+nmsgParamCount),0);
				nmsgParamCount++;
				nmsgParamCount%=1024;
			}

			if(g_bCaijiEnable)
			{
				CopyMemory(pCaijiBuf+nOffset,pBuf,nReturn);
				nOffset+=nReturn;
				//nCaijiPacketcount++;
			}
			//if(nOffset>=256*1024*1024)
			if (nOffset >= CJ_RAW_SIZE)
			{
				g_bCaijiEnable=0;
				//nCaijiPacketcount=0;
				nOffset=0;
				SetEvent(hCaijiBufReadyEvent);
			}

		}
		//nOffset	+=nReturn;
		//TRACE("%d:%d\n",ii,nReturn);


	}
//	QueryPerformanceCounter(&nTimeStop);
//	TRACE("%f\n",(nTimeStop.QuadPart-nTimeBegin.QuadPart)*1.0e6/nFreq.QuadPart);



	//CFile f;
	//f.Open("d:\\tst1.dat",CFile::modeCreate|CFile::modeWrite);
	//f.Write(pBuf,nOffset);
	//f.Close();

	//delete []pBuf;

	return 1;


}


UINT BoKongThread(LPVOID pParam)
{
	THREADPARAMS *ptp=(THREADPARAMS *)pParam;
	HWND hWnd=ptp->hWnd;
	HANDLE hPcieHandle=ptp->hPcieHandle;
	UINT nBokongEnable=ptp->nBokongEnable;
	HANDLE hCaijiBufReadyEvent=ptp->hCaijiBufReadyEvent;
	HANDLE hEvent = ptp->hEvent;
	BYTE *pCaijiBuf=ptp->pBuf;
	delete ptp;

	HANDLE	m_hEvent;
	ULONG	nOutput;	// Count written to bufOutpu

	DWORD nReturn=0;
	BOOL b;

	DWORD nPacketNumber=0;

	IO_READ_WRITE_DATA theIOData;
	IO_READ_WRITE_DATA pIOData;
	ZeroMemory(&pIOData,sizeof(IO_READ_WRITE_DATA));

	//////////////////////////////////////////
	DMAREADINFO dmareadinfo;
	BKINFO bkinfo;

	dmareadinfo.nHead0=DMAREADINFOHEAD0;
	dmareadinfo.nHead1=DMAREADINFOHEAD1;
	dmareadinfo.nHead2=DMAREADINFOHEAD2;
	dmareadinfo.nHead3=DMAREADINFOHEAD3;
	dmareadinfo.nHead4=DMAREADINFOHEAD4;
	dmareadinfo.nHead5=DMAREADINFOHEAD5;

	dmareadinfo.nInfoType=0x00000001;
	dmareadinfo.nInfoLength=64/16;		//以128位（16字节）为单位
	////////////////////////////////////////
	UINT nCurrentBKDescCount=0;		//下一个100ms，执行的波控描述字数目
	//UINT nTotalTime=100000*125;		//100ms  按照8ns为单位计算
	//UINT nTotalTime = (UINT)(100000 * SAMPLE_FREQ_MHZ);		//100ms  按照10ns为单位计算
	UINT nTotalTime = (UINT)(250000 * SAMPLE_FREQ_MHZ);		//100ms  按照10ns为单位计算,20231202 改为250ms
	UINT nLeftTime=nTotalTime;

	UINT nCurrentPulseCount;
	UINT nCurrentPulseRepeatInterval;		//注意，这里是按照8ns为单位进行计算的
	UINT nBuXiangPulseCount;
	UINT nBuXiangPulseRepeatInterval=0;
	int nWorkMode;

	UINT nCurrentDescTaskOccupyTime,nTriedOccupyTime;

	/////////////////////////////////////////
	//初始化波控扫描表
	InitializeBKScanTable();
	int nBKScanTableReadPoint=0;
	/////////////////////////////////////////
	LARGE_INTEGER nTimeBegin,nTimeStop,nFreq,nLastTime;
	float fDiffTime;
	
	if(!QueryPerformanceFrequency(&nFreq))
	{
		AfxMessageBox(_T("Failed to set timer,now exit!"));
		return 1;
	}
	QueryPerformanceCounter(&nTimeBegin);

	m_hEvent = CreateEvent(NULL, FALSE, FALSE, NULL);//创建自动重置事件
	if( m_hEvent==NULL)
	{
		TRACE(_T("ERROR: CreateEvent returns %0x."), GetLastError());
		goto exit;
	}

	//将创建的自动重置事件句柄传递给驱动程序
	if (!DeviceIoControl(hPcieHandle,
		IOCTL_REGISTER_BKREQ_EVENT,
		&m_hEvent,
		sizeof(m_hEvent),
		NULL,
		0,
		&nOutput,
		NULL)
		)
	{
		TRACE(_T("ERROR: DeviceIoControl returns %0x."), GetLastError());
		goto exit;
	}
	///////////////////////////////////////////////////
	AfxBeginThread(TargetSimThread, hPcieHandle, 0, 16 * 1024);
	////////////////////////////////////////////////////
	UINT nLength, nLength1;
	nLastTime.QuadPart = 0;
	nTimeStop.QuadPart = 0;
	while(g_bCaiji)
	{
		//while (WaitForSingleObject(m_hEvent, 0)!=WAIT_OBJECT_0);//等待事件发生


		WaitForSingleObject(m_hEvent, INFINITE);
		ResetEvent(hEvent);			//让DMA采集线程暂停

		nLastTime = nTimeStop;
		QueryPerformanceCounter(&nTimeStop);
		

		/////////////////////////////
		//1.先读取bk申请的包号
		theIOData.nAddr=32;
		b=DeviceIoControl(hPcieHandle,IOCTL_MEM0_READ,&theIOData,sizeof(IO_READ_WRITE_DATA),&pIOData,sizeof(IO_READ_WRITE_DATA),&nReturn,NULL);
		if(!b)
		{
			AfxMessageBox(_T("bkThead:mem0 reg Read Failed"));
		}
		else
		{
			nPacketNumber=pIOData.nData;
		}
		////////////////////////////////////
		//2.写入包号，表示应答
		theIOData.nMode=0;	//32bits写
		theIOData.nAddr=32;
		theIOData.nData=nPacketNumber;	//每100ms增加1个
		b=DeviceIoControl(hPcieHandle,IOCTL_MEM0_WRITE,&theIOData,sizeof(IO_READ_WRITE_DATA),&pIOData,sizeof(IO_READ_WRITE_DATA),&nReturn,NULL);
		if(!b)
		{
			AfxMessageBox(_T("bkThead:mem0 reg write Failed"));
		}
		/////////////////////////////////////
		//3.波控信息包的写入
		//得到波控扫描表中当前位置波控描述字，得到其所占用的时间
		nCurrentBKDescCount=0;		//下一个100ms，执行的波控描述字数目
		nLeftTime=nTotalTime;
		
		//
		while(nLeftTime>0)
		{
			nCurrentPulseCount=bk_table[nBKScanTableReadPoint].nPulseGroup;
			nCurrentPulseRepeatInterval=bk_table[nBKScanTableReadPoint].nPRI;

			nWorkMode = bk_table[nBKScanTableReadPoint].nWorkMode;

			nCurrentDescTaskOccupyTime = nCurrentPulseCount*nCurrentPulseRepeatInterval;
			//if (nWorkMode == 0)
			//{
			//	nBuXiangPulseCount = nCurrentPulseCount;
			//	nBuXiangPulseRepeatInterval = nCurrentPulseRepeatInterval;
			//	nCurrentPulseCount = bk_table[nBKScanTableReadPoint+1].nPulseGroup;					//注意这里没有判断越界
			//	nCurrentPulseRepeatInterval = bk_table[nBKScanTableReadPoint+1].nPRI;

			//	nTriedOccupyTime = nCurrentPulseCount*nCurrentPulseRepeatInterval + nBuXiangPulseCount*nBuXiangPulseRepeatInterval;
			//	//TRACE("idle:%d,%d,%d,%d\n", nBuXiangPulseCount, nBuXiangPulseRepeatInterval, nCurrentPulseCount, nCurrentPulseRepeatInterval);
			//}
			//else
			{
				nTriedOccupyTime = nCurrentPulseCount*nCurrentPulseRepeatInterval;
				//TRACE("TimeOcp:%d\n", nCurrentDescTaskOccupyTime);
			}

			

			//if (nCurrentDescTaskOccupyTime > nLeftTime)
			if (nTriedOccupyTime > nLeftTime)
			{
				//TRACE("leftTime:%d\n", nLeftTime);
				////////////////////////////////////////////
				//这里填充空闲包
				bkinfo.nInfoData[0] = nPacketNumber;

				bkinfo.nInfoData[1] = 0xf0000000;		//20200610
				bkinfo.nInfoData[2] = 0;
				bkinfo.nInfoData[3] = 0;
				bkinfo.nInfoData[4] = 0;
				bkinfo.nInfoData[5] = 0;
				bkinfo.nInfoData[6] = 0;
				bkinfo.nInfoData[7] = 0;
				bkinfo.nInfoData[8] = 0;
				bkinfo.nInfoData[9] = 0;
				bkinfo.nInfoData[10] = 0;
				bkinfo.nInfoData[11] = 0;
				bkinfo.nInfoData[12] = 0;
				bkinfo.nInfoData[13] = 0;
				bkinfo.nInfoData[14] = 0;
				bkinfo.nInfoData[15] = 0;
				CopyMemory(&dmareadinfo.pInfo[nCurrentBKDescCount * 64], bkinfo.nInfoData, 64);
				nCurrentBKDescCount++;
				////////////////////////////////////////////
				//TRACE("nCurrentBKDescCount:%d\n", nCurrentBKDescCount);
				break;
			}
				

			bkinfo.nInfoData[0]=nPacketNumber;

			bkinfo.nInfoData[1]=0;
			bkinfo.nInfoData[1]|=bk_table[nBKScanTableReadPoint].nWorkMode<<28;		//4bits
			bkinfo.nInfoData[1]|=bk_table[nBKScanTableReadPoint].nPulseGroup<<16;	//12bits
			//TRACE("pulsegrout:%08x\n", bkinfo.nInfoData[1]);
			bkinfo.nInfoData[1]|=bk_table[nBKScanTableReadPoint].nSigType<<12;		//4bits
			bkinfo.nInfoData[1]|=bk_table[nBKScanTableReadPoint].nTxAtten<<9;		//3bits	
			bkinfo.nInfoData[1]|=bk_table[nBKScanTableReadPoint].nReserved0<<0;		//9bits

			bkinfo.nInfoData[2]=0;
			bkinfo.nInfoData[2]|=bk_table[nBKScanTableReadPoint].nFreqPoint<<20;	//12bits
			bkinfo.nInfoData[2]|=bk_table[nBKScanTableReadPoint].nPulseWidth<<0;	//20bits	

			bkinfo.nInfoData[3]=bk_table[nBKScanTableReadPoint].nPRI;

			bkinfo.nInfoData[4]=bk_table[nBKScanTableReadPoint].nLFMSTARTWORD;
			bkinfo.nInfoData[5]=bk_table[nBKScanTableReadPoint].nLFMINCWORD;

			bkinfo.nInfoData[6]=0;
			bkinfo.nInfoData[6] |= bk_table[nBKScanTableReadPoint].nTxEleCode << 16;		//仰角在高位 20200416
			bkinfo.nInfoData[6] |= bk_table[nBKScanTableReadPoint].nTxAziCode;

			bkinfo.nInfoData[7]=0;
			//bkinfo.nInfoData[7] |=bk_table[nBKScanTableReadPoint].nTxChSel<<16;
			bkinfo.nInfoData[7] |= bk_table[nBKScanTableReadPoint].nTxChSel ;
			//bkinfo.nInfoData[7] |=bk_table[nBKScanTableReadPoint].nRxEleCode;		//lvwx 0507

			bkinfo.nInfoData[8]=0;
			bkinfo.nInfoData[8] |=bk_table[nBKScanTableReadPoint].nBeam0Code<<16;
			bkinfo.nInfoData[8] |=bk_table[nBKScanTableReadPoint].nBeam1Code;

			bkinfo.nInfoData[9]=0;
			bkinfo.nInfoData[9] |=bk_table[nBKScanTableReadPoint].nBeam2Code<<16;
			bkinfo.nInfoData[9] |= bk_table[nBKScanTableReadPoint].nRxEleCode;		//lvwx 20240202 bk_table[nBKScanTableReadPoint].nBeam3Code;

			bkinfo.nInfoData[10]=0;
			bkinfo.nInfoData[10] |=bk_table[nBKScanTableReadPoint].nBeam4Code<<16;
			bkinfo.nInfoData[10] |=bk_table[nBKScanTableReadPoint].nBeam5Code;

			bkinfo.nInfoData[11]=0;
			bkinfo.nInfoData[11] |=bk_table[nBKScanTableReadPoint].nBeam6Code<<16;
			bkinfo.nInfoData[11] |=bk_table[nBKScanTableReadPoint].nBeam7Code;

			bkinfo.nInfoData[12]=0;
			bkinfo.nInfoData[12] |=bk_table[nBKScanTableReadPoint].nBeam8Code<<16;
			bkinfo.nInfoData[12] |=bk_table[nBKScanTableReadPoint].nBeam9Code;

			bkinfo.nInfoData[13]=0;
			bkinfo.nInfoData[13] |=bk_table[nBKScanTableReadPoint].nBeam10Code<<16;
			bkinfo.nInfoData[13] |=bk_table[nBKScanTableReadPoint].nBeam11Code;

			bkinfo.nInfoData[14]=0;
			bkinfo.nInfoData[14] |=bk_table[nBKScanTableReadPoint].nPitchCode<<16;
			bkinfo.nInfoData[14] |=bk_table[nBKScanTableReadPoint].nRollCode;

			bkinfo.nInfoData[15]=0;
			bkinfo.nInfoData[15] |=bk_table[nBKScanTableReadPoint].nCourseCode<<16;
			bkinfo.nInfoData[15] |=bk_table[nBKScanTableReadPoint].nReserved1;

			CopyMemory(&dmareadinfo.pInfo[nCurrentBKDescCount*64],bkinfo.nInfoData,64);
			//TRACE("Offset:%d\n",nCurrentBKDescCount*64);
			//for(int mm=0;mm<16;mm++)
			//{
			//	TRACE("%08x,",bkinfo.nInfoData[mm]);
			//}
			//TRACE("\n");

			nCurrentBKDescCount++;

			nLeftTime=nLeftTime-nCurrentDescTaskOccupyTime;
			//TRACE("tt:%d,%d\n", nCurrentBKDescCount, nLeftTime);
			nBKScanTableReadPoint++;
			//if(nBKScanTableReadPoint>=BK_DESC_TOTAL_NUMBER)
			if (nBKScanTableReadPoint >= g_bk_desc_total_number_final)		//20200609
			
				nBKScanTableReadPoint=0;
		}
//		TRACE("nCurrentBKDesc:%d\n", nCurrentBKDescCount);
		////////////////////////////////////////////////
		CopyMemory(&dmareadinfo.pInfo[nCurrentBKDescCount * 64], &targetsim_dmareadinfo, 512);
		nLength = dmareadinfo.nInfoLength * 4 + 8 + 128;
		////////////////////////////////////////////////

		//TRACE(_T("bk Info added count:%d\n"),nCurrentBKDescCount);

		dmareadinfo.nInfoLength=nCurrentBKDescCount*(64/16);

		//CFile f;
		//f.Open("e:\\tt.dat",CFile::modeCreate|CFile::modeWrite);
		//f.Write(&dmareadinfo,sizeof(dmareadinfo));
		//f.Close();
		//nLength = dmareadinfo.nInfoLength * 4 + 8;

		b=WriteFile(hPcieHandle,&dmareadinfo,sizeof(dmareadinfo),&nReturn,NULL);		
		//b = WriteFile(hPcieHandle, &dmareadinfo, nLength*4, &nReturn, NULL);
		////////////////////////////////////
		//nLength = dmareadinfo.nInfoLength * 4 + 8;
		//nLength1 = ((nLength & 0xff) << 24);
		//nLength1 |= ((nLength& 0xFF00) << 8);
		//nLength1 |= ((nLength& 0xFF0000) >> 8);
		//nLength1 |= ((nLength & 0xFF000000) >> 24);
		//
		//theIOData.nMode = 0;	//32bits
		//theIOData.nAddr = 4;
		//theIOData.nData = 0;	//
		//b = DeviceIoControl(hPcieHandle, IOCTL_MEM0_WRITE, &theIOData, sizeof(IO_READ_WRITE_DATA), &pIOData, sizeof(IO_READ_WRITE_DATA), &nReturn, NULL);
		//if (!b)
		//{
		//	AfxMessageBox(_T("bkThead:mem0 reg write Failed"));
		//}

		//4.当写入完成后，写入ready标志，通知FPGA开始DMA读
		theIOData.nMode=0;	//32bits
		theIOData.nAddr=16;
		theIOData.nData=0x20000000;	//
		b=DeviceIoControl(hPcieHandle,IOCTL_MEM0_WRITE,&theIOData,sizeof(IO_READ_WRITE_DATA),&pIOData,sizeof(IO_READ_WRITE_DATA),&nReturn,NULL);
		if(!b)
		{
			AfxMessageBox(_T("bkThead:mem0 reg write Failed"));
		}
		SetEvent(hEvent);

		//fDiffTime = (nTimeStop.QuadPart - nLastTime.QuadPart)*1000000.0f / nFreq.QuadPart;
		//if ( fDiffTime<800.0f)
		//{
		//	TRACE(_T("bkEvent:Time:%.9f,diffTime:%.3f,PacketNumber:%d\n"), (nTimeStop.QuadPart - nTimeBegin.QuadPart)*1.0f / nFreq.QuadPart, fDiffTime, nPacketNumber);
		//}
		
		////////////////////////////////////////////
		//Sleep(5);

		//nLength = 128;// dmareadinfo.nInfoLength * 4 + 8;
		//nLength1 = ((nLength & 0xff) << 24);
		//nLength1 |= ((nLength & 0xFF00) << 8);
		//nLength1 |= ((nLength & 0xFF0000) >> 8);
		//nLength1 |= ((nLength & 0xFF000000) >> 24);

		//// write target sim info
		//b = WriteFile(hPcieHandle, &targetsim_dmareadinfo, nLength*4, &nReturn, NULL);
		//
		//theIOData.nMode = 0;	//32bits
		//theIOData.nAddr = 4;
		//theIOData.nData = nLength1;	//
		//b = DeviceIoControl(hPcieHandle, IOCTL_MEM0_WRITE, &theIOData, sizeof(IO_READ_WRITE_DATA), &pIOData, sizeof(IO_READ_WRITE_DATA), &nReturn, NULL);
		//if (!b)
		//{
		//	AfxMessageBox(_T("targetSim:mem0 reg write Failed"));
		//}
		//////////////////////////////////////
		////4.当写入完成后，写入ready标志，通知FPGA开始DMA读
		//theIOData.nMode = 0;	//32bits
		//theIOData.nAddr = 16;
		//theIOData.nData = 0x20000000;	//
		//b = DeviceIoControl(hPcieHandle, IOCTL_MEM0_WRITE, &theIOData, sizeof(IO_READ_WRITE_DATA), &pIOData, sizeof(IO_READ_WRITE_DATA), &nReturn, NULL);
		//if (!b)
		//{
		//	AfxMessageBox(_T("targetSim:mem0 reg write Failed"));
		//}
		//////////////////////////////////////////////

	}

exit:

	return 1;
}

UINT ActiveRadarDSPThread(LPVOID pParam)
{
	//DWORD szReadBuf[16384];
	//DWORD dwHuiboData[16384];
	DWORD szReadBuf[32768];
	DWORD dwHuiboData[32768];

	DWORD nReadNum;
	if (!hActiveRadarDSPPipeRead) return -1;

	UINT nState=0;
	UINT ii,kk;

	UINT nMTPParam[HUIBOHEADERLEN];

	UINT nRemainDWORDs;
	UINT nHuiboLengthInMTP;

	UINT nHaveCopiedHeaderDWords=0;
	UINT nHaveCopiedHuiboDWords = 0;

	UINT nLastMTPCount=0, nCurrentMTPCount = 0;

	UINT nPulseCountInPulseGroup;

	BOOL bOneMTPFinishFlag = FALSE;
	BOOL bFindPulseGroupStartFlag = FALSE;

	//PULSEGROUPDESC stuPulseGrpDesc;

	DWORD nWritten;

	////////////////////////////////////////////////////////
	AfxBeginThread(RadarSigProcThread, NULL,0,32*1024*1024);
	////////////////////////////////////////////////////////

	ii = 0;
	while (1)
	{
		//读取管道中的数据
		ReadFile(hActiveRadarDSPPipeRead, szReadBuf, 65536, &nReadNum, NULL);
		//找MTP的头
		kk = 0;
		while (kk<16384)
		{
			switch (nState)
			{
				case 0:
					if (szReadBuf[kk] == HUIBOINFOHEAD0) nState = 1;
					else nState = 0;
					kk = kk + 1;
					break;
				case 1:
					if (szReadBuf[kk] == HUIBOINFOHEAD1) nState = 2;
					else if (szReadBuf[kk] == HUIBOINFOHEAD0) nState = 1;
					else nState = 0;

					kk = kk + 1;
					break;
				case 2:
					if (szReadBuf[kk] == HUIBOINFOHEAD2) nState = 3;
					else if (szReadBuf[kk] == HUIBOINFOHEAD0) nState = 1;
					else nState = 0;


					kk = kk + 1;
					break;
				case 3:
					if (szReadBuf[kk] == HUIBOINFOHEAD3) nState = 4;
					else if (szReadBuf[kk] == HUIBOINFOHEAD0) nState = 1;
					else nState = 0;



					kk = kk + 1;
					break;
				case 4:
					nRemainDWORDs = 16384 - kk;
					if (nRemainDWORDs >= HUIBOHEADERLEN)		//
					{
						CopyMemory(nMTPParam, &szReadBuf[kk], HUIBOHEADERLEN * 4);
						kk = kk + HUIBOHEADERLEN;
						nRemainDWORDs = 16384 - kk;
						//nHuiboLengthInMTP = nMTPParam[10];
						nHuiboLengthInMTP = nMTPParam[10] * 8;		//个波束nMTPParam[10]个距离单元(这个长度是按照位宽256位计算的，20231202），所以总长度需要*8
						//assert(nHuiboLengthInMTP < 32768);
						if (nHuiboLengthInMTP >= 32768)
						{
							TRACE("nHuiboLen:%lu\n", nHuiboLengthInMTP);
							nHuiboLengthInMTP = 32000;
							//assert(nHuiboLengthInMTP < 32768);
						}
						if (nLastMTPCount + 1 != nMTPParam[0])
							TRACE(_T("Lost MTP,L:%d,C:%d,1:%d,2:%d,nLen:%d\n"), nLastMTPCount, nMTPParam[0], nMTPParam[1], nMTPParam[2], nHuiboLengthInMTP);
						nLastMTPCount = nMTPParam[0];
						//TRACE(_T("MTP Count:%d\n"), nMTPParam[0]);
						if (nRemainDWORDs >= nHuiboLengthInMTP)		//
						{
							CopyMemory(dwHuiboData, &szReadBuf[kk], nHuiboLengthInMTP * 4);
							kk = kk + nHuiboLengthInMTP;

							bOneMTPFinishFlag = TRUE;
							nState = 0;
						}
						else
						{
							CopyMemory(dwHuiboData, &szReadBuf[kk], nRemainDWORDs * 4);
							nHaveCopiedHuiboDWords = nRemainDWORDs;
							kk = kk + nRemainDWORDs;
							nState = 6;
						}
					}
					else
					{
						CopyMemory(nMTPParam, &szReadBuf[kk], nRemainDWORDs * 4);
						nHaveCopiedHeaderDWords = nRemainDWORDs;
						kk = kk + nRemainDWORDs;
						nState = 5;
					}

					break;
				case 5:
					CopyMemory(&nMTPParam[nHaveCopiedHeaderDWords], &szReadBuf[kk], (HUIBOHEADERLEN - nHaveCopiedHeaderDWords) * 4);
					kk = kk + HUIBOHEADERLEN - nHaveCopiedHeaderDWords;

					nRemainDWORDs = 16384 - kk;
					//nHuiboLengthInMTP = nMTPParam[10];
					nHuiboLengthInMTP = nMTPParam[10] * 8;		//个波束nMTPParam[10]个距离单元(这个长度是按照位宽256位计算的，20231202），所以总长度需要*8
					assert(nHuiboLengthInMTP < 32768);
					if (nLastMTPCount + 1 != nMTPParam[0])
						TRACE(_T("Lost MTP1,L:%d,C:%d\n"), nLastMTPCount, nMTPParam[0]);
					nLastMTPCount = nMTPParam[0];
					//TRACE(_T("MTP Count:%d\n"), nMTPParam[0]);
					if (nRemainDWORDs >= nHuiboLengthInMTP)		//
					{
						CopyMemory(dwHuiboData, &szReadBuf[kk], nHuiboLengthInMTP * 4);
						kk = kk + nHuiboLengthInMTP;

						bOneMTPFinishFlag = TRUE;

						nState = 0;
					}
					else
					{
						CopyMemory(dwHuiboData, &szReadBuf[kk], nRemainDWORDs * 4);
						nHaveCopiedHuiboDWords = nRemainDWORDs;
						kk = kk + nRemainDWORDs;
						nState = 6;
					}
					break;
				case 6:
					nRemainDWORDs = 16384 - kk;
					if (nRemainDWORDs >= nHuiboLengthInMTP - nHaveCopiedHuiboDWords)		//
					{
						CopyMemory(&dwHuiboData[nHaveCopiedHuiboDWords], &szReadBuf[kk], (nHuiboLengthInMTP - nHaveCopiedHuiboDWords) * 4);
						kk = kk + nHuiboLengthInMTP - nHaveCopiedHuiboDWords;

						bOneMTPFinishFlag = TRUE;

						nState = 0;
					}
					else
					{
						CopyMemory(&dwHuiboData[nHaveCopiedHuiboDWords], &szReadBuf[kk], nRemainDWORDs * 4);
						nHaveCopiedHuiboDWords = nHaveCopiedHuiboDWords + nRemainDWORDs;
						kk = kk + nRemainDWORDs;
						nState = 6;
					}
					break;
				default:
					break;
			}
			////////////////////////////////////////////////////////
			if (bOneMTPFinishFlag)		//一个主触发的数据处理完成
			{
				
				if (nMTPParam[1] & 0x02)	//脉组起始
				{
					bFindPulseGroupStartFlag = TRUE;
					nPulseCountInPulseGroup = 0;
				}

				if (bFindPulseGroupStartFlag)
				{
					//if ((nPulseCountInPulseGroup & 0xffffff400) == 0)
					if (nPulseCountInPulseGroup <= MAX_PULSE_NUMBER_IN_GROUP)
					{
						CopyMemory(&ActiveRadarDSPThread_stuPulseGrpDesc.hbData.nIQData[nPulseCountInPulseGroup], dwHuiboData, nHuiboLengthInMTP * 4);
					}
					
					//TRACE("nHuiboLen:%d\n", nHuiboLengthInMTP);// nPulseCountInPulseGroup);
					nPulseCountInPulseGroup++;

					assert(nPulseCountInPulseGroup <= 8192);

				}


				
				if ((nMTPParam[1] & 0x01) && bFindPulseGroupStartFlag)	//脉组End
				{
					CopyMemory(&ActiveRadarDSPThread_stuPulseGrpDesc.PulseGroupHead.nHeader[0], nMTPParam, HUIBOHEADERLEN * 4);
					//ActiveRadarDSPThread_stuPulseGrpDesc.PulseGroupHead.stuPGHeader.nPulseNumber = nPulseCountInPulseGroup + 1;
					ActiveRadarDSPThread_stuPulseGrpDesc.PulseGroupHead.stuPGHeader.nPulseNumber = nPulseCountInPulseGroup ;

					//if (nPulseCountInPulseGroup != MAX_PULSE_NUMBER_IN_GROUP)
					//{
					//	TRACE("Error pulse number:%d\n",nPulseCountInPulseGroup);
					//}
					
					if (hRadarSigProcPipeWrite)		//写到pipe中
					{
						WriteFile(hRadarSigProcPipeWrite, &ActiveRadarDSPThread_stuPulseGrpDesc, sizeof(PULSEGROUPDESC), &nWritten, NULL);
					}

					//TRACE("MTP Count:%d,Pulse Count:%d\n", nMTPParam[0], nPulseCountInPulseGroup);

					bFindPulseGroupStartFlag = FALSE;
					nPulseCountInPulseGroup = 0;
				}

				bOneMTPFinishFlag = FALSE;

			}
			////////////////////////////////////////////////////////
		}
		
			ii++;
			if (ii % 16384 == 0)
			{
				TRACE(_T("DSP Readed:%d\n"), ii);
			}
	}
}


//初始化波控扫描表
void InitializeBKScanTable(void)
{
	//1
	UINT nWorkMode=1;		//4bits		active radar,1 NS mode,2 single Narrow beam
	UINT nPulseGroup = MAX_PULSE_NUMBER_IN_GROUP;// 1024;	//12bits	
	UINT nSigType=0;		//4bits
	UINT nTxAtten=1;		//3bits
	UINT nReserved0=0x000;		//9bits
	//2
	UINT nFreqPoint = g_nFreqPoint;// 3;		//12bits			20230912
	float fPulseWidth = TAO_US;
	UINT nPulseWidth = UINT(SAMPLE_FREQ_MHZ*fPulseWidth - 1);		//20bits
	//3
	UINT nPRI;
	if (MAX_PULSE_NUMBER_IN_GROUP <= 1024)
	{
		nPRI = (UINT)(SAMPLE_FREQ_MHZ * 240 - 1);
	}
	else if (MAX_PULSE_NUMBER_IN_GROUP == 2048)
	{
		nPRI = (UINT)(SAMPLE_FREQ_MHZ * 120 - 1);
	}
	else
	{
		nPRI = (UINT)(SAMPLE_FREQ_MHZ * 60 - 1);
	}
	assert(nPulseWidth * 10 < nPRI);
	
	//4
	float fSampleFreq = SAMPLE_FREQ_MHZ;//20200507 lvwx / 8.0f;
	float fBW = BW_MHZ;

	float fLFMStartWord=(fSampleFreq-fBW/2.0f)/fSampleFreq*pow(2.0f,32);
	UINT nLFMSTARTWORD=UINT(fLFMStartWord);
	//5
	float fLFMIncWord=fBW/(fSampleFreq*fPulseWidth)/fSampleFreq*pow(2.0f,32);
	UINT nLFMINCWORD=UINT(fLFMIncWord);
	//6
	UINT nTxAziCode=0;
	UINT nTxEleCode=0;
	//7
	//UINT nTxChSel = 0x0203ff; //0x00ffffff;			//20200507lvwx 010101  选择发射通道
	UINT nTxChSel = 0x0101FF; //0x00ffffff;			//20200507lvwx 010101  选择发射通道1通道
	//UINT nTxChSel = 0x0103FF; //0x00ffffff;			//20200507lvwx 010101  选择发射通道1子阵
	//UINT nTxChSel = 0x0103FF; //0x00ffffff;			//20200507lvwx 010101  选择发射通道1子阵
	//UINT nTxChSel = 0x010FFF; //0x00ffffff;			//20200507lvwx 010101  选择发射通道1子阵
	//UINT nTxChSel = 0x01FFFF; //0x00ffffff;			//20200507lvwx 010101  选择发射通道1子阵
	//UINT nTxChSel = 0x03FFFF; //0x00ffffff;			//20200507lvwx 010101  选择发射通道2子阵
	//UINT nTxChSel = 0x0FFFFF; //0x00ffffff;			//20200507lvwx 010101  选择发射通道2子阵
	//UINT nTxChSel = 0x0203ff; //0x00ffffff;			//20200507lvwx 010101  选择发射通道
	UINT nRxEleCode=0;
	//8
	UINT nBeam0Code=0;
	UINT nBeam1Code=0;
	//9
	UINT nBeam2Code=0;
	UINT nBeam3Code=0;
	//10
	UINT nBeam4Code=0;
	UINT nBeam5Code=0;
	//11
	UINT nBeam6Code=0;
	UINT nBeam7Code=0;
	//12
	UINT nBeam8Code=0;
	UINT nBeam9Code=0;
	//13
	UINT nBeam10Code=0;
	UINT nBeam11Code=0;
	//14
	UINT nPitchCode=0;
	UINT nRollCode=0;
	//15
	UINT nCourseCode=0;
	UINT nReserved1=0;

	int ii;
	//float fTxAzi = -60.0f;
	//float fTxAzi = -25.87f;
	float fTxAzi = 0.0f;
	//float fTxAziStep = 2.0f;
	//float fTxAziStep = 2.0f;// 2.0f;
	float fTxAziStep = 0.0f;// 2.0f;
	WORD wTxAzi=0,wRxAzi=0;
	float phaseshift_azi = 0.0f;
	//float bochang = 0.03f;
	//float bochang = 0.0375f;
	float bochang = BOCHANG;
	//float d = 0.015f;
	float d = JIANJU;
	float fTmp;
	short wTmp;

	float fEleScan = 20;
	//float fEleScan = 0;
	WORD wTxEle = 0;
	bochang0 = c_speed / ((nFreqPoint * 10 + 9600)*1e6);


	for(ii=0;ii<BK_DESC_TOTAL_NUMBER;ii++)
	{
		bk_table[ii].nPacketCount=0;

		bk_table[ii].nWorkMode=nWorkMode;
		bk_table[ii].nPulseGroup=nPulseGroup;
		bk_table[ii].nSigType=nSigType;
		bk_table[ii].nTxAtten=nTxAtten;
		bk_table[ii].nReserved0=nReserved0;

		bk_table[ii].nFreqPoint=nFreqPoint;
		bk_table[ii].nPulseWidth=nPulseWidth;

		bk_table[ii].nPRI=nPRI;
		//4
		bk_table[ii].nLFMSTARTWORD=nLFMSTARTWORD;
		//5
		bk_table[ii].nLFMINCWORD=nLFMINCWORD;
		//6
		//if(fTxAzi<0.0f)
		//	fTxAzi+=360.0f;
		//fTmp = 2 * 3.1415926535f*d / bochang0*sin(fTxAzi / 180.0f*3.1415926535f);
		phaseshift_azi = 2 * 3.1415926535f*d / bochang0*sin(fTxAzi / 180.0f*3.1415926535f);

		wTmp =  phaseshift_azi / (2 * 3.1415926535f)*65536.0f;
		if (wTmp < 0)
			wTmp += 65536.0f;

		//wTxAzi=unsigned short(fTxAzi/360.0f*65536.0f);
		wTxAzi = WORD(wTmp);// unsigned short(phaseshift_azi / (2 * 3.1415926535f)*65536.0f);


		phaseshift_azi = 2 * 3.1415926535f*d / bochang0*sin(fTxAzi / 180.0f*3.1415926535f);

		wTmp = phaseshift_azi / (2 * 3.1415926535f)*65536.0f;
		if (wTmp < 0)
			wTmp += 65536.0f;

		//wTxAzi=unsigned short(fTxAzi/360.0f*65536.0f);
		wRxAzi = WORD(wTmp);// unsigned short(phaseshift_azi / (2 * 3.1415926535f)*65536.0f);
		

		TRACE("azi:%.3f,wTxAzi:%d\n", phaseshift_azi, wTxAzi);
		bk_table[ii].nTxAziCode = wTxAzi;// wTxAzi;  lvwx0416

		phaseshift_azi = 2 * 3.1415926535f*d / bochang0*sin(fEleScan / 180.0f*3.1415926535f);

		wTmp = phaseshift_azi / (2 * 3.1415926535f)*65536.0f;
		if (wTmp < 0)
			wTmp += 65536.0f;

		//wTxAzi=unsigned short(fTxAzi/360.0f*65536.0f);
		wTxEle = WORD(wTmp);// unsigned short(phaseshift_azi / (2 * 3.1415926535f)*65536.0f);
		
		bk_table[ii].nTxEleCode = wTxEle;
		//7
		bk_table[ii].nTxChSel=nTxChSel;
		phaseshift_azi = 2 * 3.1415926535f*d / bochang0*sin(fEleScan / 180.0f*3.1415926535f);

		wTmp = phaseshift_azi / (2 * 3.1415926535f)*65536.0f;
		if (wTmp < 0)
			wTmp += 65536.0f;

		//wTxAzi=unsigned short(fTxAzi/360.0f*65536.0f);
		wTxEle = WORD(wTmp);// unsigned short(phaseshift_azi / (2 * 3.1415926535f)*65536.0f);
		bk_table[ii].nRxEleCode = wTxEle;// 
		//8
		bk_table[ii].nBeam0Code = wRxAzi;//nBeam0Code;
		bk_table[ii].nBeam1Code = wRxAzi;//nBeam1Code;
		//9
		bk_table[ii].nBeam2Code=nBeam2Code;
		bk_table[ii].nBeam3Code=nBeam3Code;
		//10
		bk_table[ii].nBeam4Code=nBeam4Code;
		bk_table[ii].nBeam5Code=nBeam5Code;
		//11
		bk_table[ii].nBeam6Code=nBeam6Code;
		bk_table[ii].nBeam7Code=nBeam7Code;
		//12
		bk_table[ii].nBeam8Code=nBeam8Code;
		bk_table[ii].nBeam9Code=nBeam9Code;
		//13
		bk_table[ii].nBeam10Code=nBeam10Code;
		bk_table[ii].nBeam11Code=nBeam11Code;
		//14
		bk_table[ii].nPitchCode=nPitchCode;
		bk_table[ii].nRollCode=nRollCode;
		//15
		bk_table[ii].nCourseCode=nCourseCode;
		bk_table[ii].nReserved1=nReserved1;

		fTxAzi += fTxAziStep;		//假设波宽为4°
		//fTxAzi += 0;		//20230920  96.0f / BK_DESC_TOTAL_NUMBER;
	}
	g_bk_desc_total_number_final = BK_DESC_TOTAL_NUMBER;

	
	//load bk table file
	CStdioFile f;
	 //f.Open(_T("e:\myScanTable20200823_2.txt"), CFile::modeRead);
	//f.Open(_T("e:\myScanTable20200823_2.txt"), CFile::modeRead);
//	f.Open(_T("e:\\myScanTable_10Deg_2.txt"), CFile::modeRead);
	 //f.Open(_T("e:\myScanTable20200724_1.txt"), CFile::modeRead);

	//f.Open(_T("e:\myScanTable_20us_200us_10MHz.txt"), CFile::modeRead);   
	
//	if (f == CFile::hFileNull)
	{
		return;
	}

	CString strTmp;
	int nReadLine = 0;
	UINT32 dTmp[31];
	while (f.ReadString(strTmp))
	{
		

		USES_CONVERSION;
		//LPCSTR c;
		//c = (LPCSTR)T2A(strTmp);
		
		sscanf((LPCSTR)T2A(strTmp), "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
			&dTmp[0], &dTmp[1], &dTmp[2], &dTmp[3], &dTmp[4], &dTmp[5], &dTmp[6], &dTmp[7],
			&dTmp[8], &dTmp[9], &dTmp[10], &dTmp[11], &dTmp[12], &dTmp[13], &dTmp[14], &dTmp[15],
			&dTmp[16], &dTmp[17], &dTmp[18], &dTmp[19], &dTmp[20], &dTmp[21], &dTmp[22], &dTmp[23],
			&dTmp[24], &dTmp[25], &dTmp[26], &dTmp[27], &dTmp[28], &dTmp[29], &dTmp[30]
			);
		///////////////////////////////////////////////////////
		bk_table[nReadLine].nPacketCount = 0;

		bk_table[nReadLine].nWorkMode = dTmp[1];
		bk_table[nReadLine].nPulseGroup = dTmp[2];
		bk_table[nReadLine].nSigType = dTmp[3];
		bk_table[nReadLine].nTxAtten = dTmp[4];
		bk_table[nReadLine].nReserved0 = ((dTmp[5] & 0x01) << 8) | (dTmp[6]&0xff);
		TRACE("ps_disable:%08x\n", bk_table[nReadLine].nReserved0);

		bk_table[nReadLine].nFreqPoint = dTmp[7];
		bk_table[nReadLine].nPulseWidth = dTmp[8];

		bk_table[nReadLine].nPRI = dTmp[9];
		//4
		bk_table[nReadLine].nLFMSTARTWORD = dTmp[10];
		//5
		bk_table[nReadLine].nLFMINCWORD = dTmp[11];
		//6
		bk_table[nReadLine].nTxAziCode = dTmp[13];// wTxAzi;  lvwx0416
		bk_table[nReadLine].nTxEleCode = dTmp[12];
		//7
		bk_table[nReadLine].nTxChSel = dTmp[14];
		bk_table[nReadLine].nRxEleCode = dTmp[12];
		//8
		bk_table[nReadLine].nBeam0Code = dTmp[15];//nBeam0Code;
		bk_table[nReadLine].nBeam1Code = dTmp[15];//nBeam1Code;
		//9
		bk_table[nReadLine].nBeam2Code = dTmp[17];
		bk_table[nReadLine].nBeam3Code = dTmp[18];
		//10
		bk_table[nReadLine].nBeam4Code = dTmp[19];
		bk_table[nReadLine].nBeam5Code = dTmp[20];
		//11
		bk_table[nReadLine].nBeam6Code = dTmp[21];
		bk_table[nReadLine].nBeam7Code = dTmp[22];
		//12
		bk_table[nReadLine].nBeam8Code = dTmp[23];
		bk_table[nReadLine].nBeam9Code = dTmp[24];
		//13
		bk_table[nReadLine].nBeam10Code = dTmp[25];
		bk_table[nReadLine].nBeam11Code = dTmp[26];
		//14
		bk_table[nReadLine].nPitchCode = dTmp[27];
		bk_table[nReadLine].nRollCode = dTmp[28];
		//15
		bk_table[nReadLine].nCourseCode = dTmp[29];
		bk_table[nReadLine].nReserved1 = dTmp[30];
		//////////////////////////////////////////////////////
		g_bk_desc_total_number_final = nReadLine + 1;
		nReadLine++;
		if (nReadLine >= BK_DESC_TOTAL_NUMBER)
		{
			TRACE(_T("Bk Desc Word too large!"));
			break;
		}



	}

	f.Close();
	
	

}

int InitializeActiveRadarDSPPipe(void)	//初始化信号处理的pipe
{

	BOOL bRet;
	
	SECURITY_ATTRIBUTES sa,sa1,sa2;
	sa.bInheritHandle = TRUE; //必须为TRUE，父进程的读写句柄可以被子进程继承
	sa.lpSecurityDescriptor = NULL;
	sa.nLength = sizeof(SECURITY_ATTRIBUTES);

	sa1.bInheritHandle = TRUE; //必须为TRUE，父进程的读写句柄可以被子进程继承
	sa1.lpSecurityDescriptor = NULL;
	sa1.nLength = sizeof(SECURITY_ATTRIBUTES);

	sa2.bInheritHandle = TRUE; //必须为TRUE，父进程的读写句柄可以被子进程继承
	sa2.lpSecurityDescriptor = NULL;
	sa2.nLength = sizeof(SECURITY_ATTRIBUTES);	
	//创建匿名管道
	bRet = CreatePipe(&hActiveRadarDSPPipeRead, &hActiveRadarDSPPipeWrite, &sa, 128*1024*1024);
	if (bRet)
	{
		TRACE(_T("成功创建DDS匿名管道！\n"));
	}
	else
	{
		TRACE(_T("创建DDS匿名管道失败！错误码:[%d]\n"), GetLastError());
		return 0;
	}
	///////////////////////////////////////////////
	bRet = CreatePipe(&hRadarSigProcPipeRead, &hRadarSigProcPipeWrite, &sa1, 512 * sizeof(PULSEGROUPDESC));
	if (bRet)
	{
		TRACE(_T("成功创建SigProc匿名管道！\n"));
	}
	else
	{
		TRACE(_T("创建SigProc匿名管道失败！错误码:[%d]\n"), GetLastError());
		return 0;
	}
	///////////////////////////////////////////////
	bRet = CreatePipe(&hTerminalPipeRead, &hTerminalPipeWrite, &sa1, 256 * sizeof(ZhongduanDisplayData));
	if (bRet)
	{
		TRACE(_T("成功创建Terminal匿名管道！\n"));
	}
	else
	{
		TRACE(_T("创建Terminal匿名管道失败！错误码:[%d]\n"), GetLastError());
		return 0;
	}
	///////////////////////////////////////////////
	for (int ii = 0; ii < 8; ii++)
	{
		bRet = CreatePipe(&g_hSigProcPipeReadArray[ii], &g_hSigProcPipeWriteArray[ii], &sa, 64 * 1024 * 1024);
		if (bRet)
		{
			TRACE(_T("成功创建SigProcArray匿名管道！\n"));
		}
		else
		{
			TRACE(_T("创建SigProcArray匿名管道失败！错误码:[%d]\n"), GetLastError());
			return 0;
		}

		bRet = CreatePipe(&g_hSBTerminalPipeReadArray[ii], &g_hSBTerminalPipeWriteArray[ii], &sa, 16 * sizeof(S4BZhongduanDisplayData));
		if (bRet)
		{
			TRACE(_T("成功创建S4BZhongduanDisplayData匿名管道！\n"));
		}
		else
		{
			TRACE(_T("创建S4BZhongduanDisplayData匿名管道失败！错误码:[%d]\n"), GetLastError());
			return 0;
		}

		g_hSigProcEvent[ii] = CreateEvent(NULL, TRUE, TRUE, NULL);//创建手动重置事件，初始状态为signaled
		if (g_hSigProcEvent[ii] == NULL)
		{
			//AfxMessageBox(LPTSTR("ERROR: CreateEvent returns %0x.", GetLastError()));
			AfxMessageBox(_T("ERROR: CreateSigProcEvent failed"));

		}
	}
	///////////////////////////////////////////////


	return 1;
	
	
}


UINT RadarSigProcThread(LPVOID pParam)
{
	DWORD nReaded=0;
	DWORD nLastMTPCount=0;
	unsigned int myLength;

	DWORD nWritten = 0;
	////////////////////////////////////////////////////////
	//20191123
	AfxBeginThread(TerminalTxThread, NULL, 0, 1 * 1024 * 1024);
	////////////////////////////////////////////////////////
	Ipp32f *p_fHuibo_4Beams[MAX_BEAM];

	p_fHuibo_4Beams[0] = g_fTempHuiboData_Beam0;
	p_fHuibo_4Beams[1] = g_fTempHuiboData_Beam1;
	p_fHuibo_4Beams[2] = g_fTempHuiboData_Beam2;
	p_fHuibo_4Beams[3] = g_fTempHuiboData_Beam3;
	p_fHuibo_4Beams[4] = g_fTempHuiboData_Beam4;
	p_fHuibo_4Beams[5] = g_fTempHuiboData_Beam5;
	p_fHuibo_4Beams[6] = g_fTempHuiboData_Beam6;
	p_fHuibo_4Beams[7] = g_fTempHuiboData_Beam7;

	p_fHuibo_4Beams[8] = g_fTempHuiboData_Beam8;
	p_fHuibo_4Beams[9] = g_fTempHuiboData_Beam9;
	p_fHuibo_4Beams[10] = g_fTempHuiboData_Beam10;
	p_fHuibo_4Beams[11] = g_fTempHuiboData_Beam11;
	p_fHuibo_4Beams[12] = g_fTempHuiboData_Beam12;
	p_fHuibo_4Beams[13] = g_fTempHuiboData_Beam13;
	p_fHuibo_4Beams[14] = g_fTempHuiboData_Beam14;
	p_fHuibo_4Beams[15] = g_fTempHuiboData_Beam15;

	p_fHuibo_4Beams[16] = g_fTempHuiboData_Beam16;
	p_fHuibo_4Beams[17] = g_fTempHuiboData_Beam17;
	p_fHuibo_4Beams[18] = g_fTempHuiboData_Beam18;
	p_fHuibo_4Beams[19] = g_fTempHuiboData_Beam19;
	p_fHuibo_4Beams[20] = g_fTempHuiboData_Beam20;
	p_fHuibo_4Beams[21] = g_fTempHuiboData_Beam21;
	p_fHuibo_4Beams[22] = g_fTempHuiboData_Beam22;
	p_fHuibo_4Beams[23] = g_fTempHuiboData_Beam23;

	p_fHuibo_4Beams[24] = g_fTempHuiboData_Beam24;
	p_fHuibo_4Beams[25] = g_fTempHuiboData_Beam25;
	p_fHuibo_4Beams[26] = g_fTempHuiboData_Beam26;
	p_fHuibo_4Beams[27] = g_fTempHuiboData_Beam27;
	p_fHuibo_4Beams[28] = g_fTempHuiboData_Beam28;
	p_fHuibo_4Beams[29] = g_fTempHuiboData_Beam29;
	p_fHuibo_4Beams[30] = g_fTempHuiboData_Beam30;
	p_fHuibo_4Beams[31] = g_fTempHuiboData_Beam31;
	unsigned char chBeamCount = 0;
	//////////////////////////////////////////////////////
	S4BPROCTHREADPARAMS *ptp[8];
	for (int ii = 0; ii < BEAM_GROUP; ii++)
	{
		ptp[ii] = new S4BPROCTHREADPARAMS;
		ptp[ii]->hRadarSBSigProcPipeRead = g_hSigProcPipeReadArray[ii];
		ptp[ii]->hEvent = g_hSigProcEvent[ii];

		ptp[ii]->hCaijiBufReadyEvent = INVALID_HANDLE_VALUE;
		ptp[ii]->p_fHuibo_4Beams[0] = p_fHuibo_4Beams[ii * 4 + 0];
		ptp[ii]->p_fHuibo_4Beams[1] = p_fHuibo_4Beams[ii * 4 + 1];
		ptp[ii]->p_fHuibo_4Beams[2] = p_fHuibo_4Beams[ii * 4 + 2];
		ptp[ii]->p_fHuibo_4Beams[3] = p_fHuibo_4Beams[ii * 4 + 3];
		ptp[ii]->nSigProcParam = 0x02;
		ptp[ii]->nBeamID = ii;

		AfxBeginThread(SingleBeamProcThread, ptp[ii], 0, 512 * 1024 * 1024);
	}
	/////////////////////////////////////////////////////
	while(1)
	{
			//读取管道中的数据
			ReadFile(hRadarSigProcPipeRead, &g_stuPulseGrpDesc, sizeof(PULSEGROUPDESC), &nReaded, NULL);
			if (nLastMTPCount + g_stuPulseGrpDesc.PulseGroupHead.stuPGHeader.nPulseNumber != g_stuPulseGrpDesc.PulseGroupHead.stuPGHeader.nMTPCount)
			{
				TRACE("nRead:%d,pulseNum:%d,Lost MTP:%d,%d\n", nReaded, g_stuPulseGrpDesc.PulseGroupHead.stuPGHeader.nPulseNumber, nLastMTPCount, g_stuPulseGrpDesc.PulseGroupHead.stuPGHeader.nMTPCount);
			}
			nLastMTPCount = g_stuPulseGrpDesc.PulseGroupHead.stuPGHeader.nMTPCount;
			/////////////////////////////////////////////////////////

			//stRadarSigProcePlsGrpData.nDistanceUnitNumber >>= 2;

			if (g_stuPulseGrpDesc.PulseGroupHead.stuPGHeader.nPulseNumber >MAX_PULSE_NUMBER_IN_GROUP)//大于128重新执行这个循环（加保护）
			{
				TRACE("Error PulseNumber\n");
				continue;
			}

			//continue;
			for (int ii = 0; ii < g_stuPulseGrpDesc.PulseGroupHead.stuPGHeader.nPulseNumber; ii++)		//对脉组内每个脉冲进行32波束分离，然后脉压,最大一共1024个脉冲
			{
				//legacy90ippsDeinterleave_32f((Ipp32f*)(stuPulseGrpDesc.hbData.nIQData[ii]), MAX_BEAM, stuPulseGrpDesc.PulseGroupHead.stuPGHeader.nPeriodLenth * 8 / MAX_BEAM, p_fHuibo_4Beams);
				legacy90ippsDeinterleave_32f((Ipp32f*)(g_stuPulseGrpDesc.hbData.nIQData[ii]), MAX_BEAM, MAX_DISTANCE_ELEMENT_NUMBER, p_fHuibo_4Beams);
				for (int jj = 0; jj <= 7; jj++)
				{
					CopyMemory((Ipp16s*)(&g_S4BPULSEGROUPDESC[jj].sbHBData[0].sData[ii]), p_fHuibo_4Beams[jj * 4 + 0], MAX_DISTANCE_ELEMENT_NUMBER * 4);
					CopyMemory((Ipp16s*)(&g_S4BPULSEGROUPDESC[jj].sbHBData[1].sData[ii]), p_fHuibo_4Beams[jj * 4 + 1], MAX_DISTANCE_ELEMENT_NUMBER * 4);
					CopyMemory((Ipp16s*)(&g_S4BPULSEGROUPDESC[jj].sbHBData[2].sData[ii]), p_fHuibo_4Beams[jj * 4 + 2], MAX_DISTANCE_ELEMENT_NUMBER * 4);
					CopyMemory((Ipp16s*)(&g_S4BPULSEGROUPDESC[jj].sbHBData[3].sData[ii]), p_fHuibo_4Beams[jj * 4 + 3], MAX_DISTANCE_ELEMENT_NUMBER * 4);
				}
				//if (g_S4BPULSEGROUPDESC[0].sbHBData[0].sData[0] == 0 && g_S4BPULSEGROUPDESC[0].sbHBData[0].sData[1] == 0)
				//{
				//	int aa = 0;
				//}
			}

			for (int jj = 0; jj <= BEAM_GROUP-1; jj++)
			//int jj = 0;
			{
				CopyMemory(&g_S4BPULSEGROUPDESC[jj].PulseGroupHead.stuPGHeader, &g_stuPulseGrpDesc.PulseGroupHead.stuPGHeader, sizeof(PULSEGROUPHEADER));
				g_S4BPULSEGROUPDESC[jj].PulseGroupHead.stuPGHeader.nPeriodLenth = MAX_DISTANCE_ELEMENT_NUMBER;

				WriteFile(g_hSigProcPipeWriteArray[jj], &g_S4BPULSEGROUPDESC[jj], sizeof(S4BPULSEGROUPDESC), &nWritten, NULL);
			}
	}


}

///////////////////////////////////////////
//20241127
//单4个波束的信号处理
static UINT SingleBeamProcThread(LPVOID pParam)
{
	S4BPROCTHREADPARAMS *ptp = (S4BPROCTHREADPARAMS *)pParam;
	HANDLE hRadarSBSigProcPipeRead = ptp->hRadarSBSigProcPipeRead;
	HANDLE hEvent = ptp->hEvent;
	HANDLE hCaijiBufReadyEvent = ptp->hCaijiBufReadyEvent;
	UINT nSigProcParm = ptp->nSigProcParam;
	Ipp32f *p_fHuibo_4Beams[4];
	p_fHuibo_4Beams[0] = ptp->p_fHuibo_4Beams[0];
	p_fHuibo_4Beams[1] = ptp->p_fHuibo_4Beams[1];
	p_fHuibo_4Beams[2] = ptp->p_fHuibo_4Beams[2];
	p_fHuibo_4Beams[3] = ptp->p_fHuibo_4Beams[3];
	UINT nBeamID = ptp->nBeamID;
	delete ptp;

	Ipp32f fIQData_Maiya_DRM_Real[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER];		//
	Ipp32f fIQData_Maiya_DRM_Imag[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER];		//
	Ipp32f fIQData_Maiya_RDM_Real[MAX_DISTANCE_ELEMENT_NUMBER][MAX_PULSE_NUMBER_IN_GROUP];		//
	Ipp32f fIQData_Maiya_RDM_Imag[MAX_DISTANCE_ELEMENT_NUMBER][MAX_PULSE_NUMBER_IN_GROUP];		//

	
	DWORD nReaded = 0;
	DWORD nLastMTPCount = 0;
	unsigned int myLength;

	UINT nMTD_Mode = nSigProcParm & 0x03;

	S4BPULSEGROUPDESC stuS4BPulseGrpDesc;
	RadarSigProcPulseGroupData stSBRadarSigProcePlsGrpData[4];
	S4BZhongduanDisplayData stZDDispData;
	Ipp32f g_fTempZhongduanData[4][MAX_DISTANCE_ELEMENT_NUMBER];				//每个波束每个脉冲不超过1000点

	stZDDispData.nHeader[0] = HUIBOINFOHEAD0;
	stZDDispData.nHeader[1] = HUIBOINFOHEAD1;
	stZDDispData.nHeader[2] = HUIBOINFOHEAD2;
	stZDDispData.nHeader[3] = HUIBOINFOHEAD3;

	ZeroMemory(g_fTempZhongduanData, 4 * MAX_DISTANCE_ELEMENT_NUMBER * 4);
	
	Ipp32f *p_fMyaiyaResult;
	Ipp32fc *p_fcMaiyaFFTMultResult;
	Ipp32f *p_fHuiboData;
	Ipp32fc *p_fcFFTSigIQ;
	CMyIpp theIpp;

	myLength = MAX_DISTANCE_ELEMENT_NUMBER;		//20231202

	p_fMyaiyaResult = ippsMalloc_32f(myLength);
	p_fcMaiyaFFTMultResult = ippsMalloc_32fc(myLength);
	p_fcFFTSigIQ = ippsMalloc_32fc(myLength);
	p_fHuiboData = ippsMalloc_32f(myLength * 2);	//4个波束的数据

	////////////////////////////////////////////
	unsigned int myOrder = PC_FFT_ORDER;		//20231202
	IppsFFTSpec_C_32fc *p_mySpec = 0;
	Ipp8u *pMemSpec = 0;
	Ipp8u *pMemInit = 0;
	Ipp8u *pMemBuffer = 0;
	int sizeSpec = 0;
	int sizeInit = 0;
	int sizeBuffer = 0;
	int flag = IPP_FFT_DIV_INV_BY_N;

	/// get sizes for required buffers
	ippsFFTGetSize_C_32fc(myOrder, flag, ippAlgHintNone, &sizeSpec, &sizeInit, &sizeBuffer);
	//TRACE("%d\n", sizeBuffer);
	/// allocate memory for required buffers
	pMemSpec = (Ipp8u*)ippsMalloc_8u(sizeSpec);

	if (sizeInit > 0)
	{
		pMemInit = (Ipp8u*)ippsMalloc_8u(sizeInit);
	}
	if (sizeBuffer > 0)
	{
		pMemBuffer = (Ipp8u*)ippsMalloc_8u(sizeBuffer);
	}

	/// initialize FFT specification structure
	ippsFFTInit_C_32fc(&p_mySpec, myOrder, flag, ippAlgHintNone, pMemSpec, pMemInit);
	/////////////////////////////////////////////
	//doppler
	unsigned int nDopplerFFTOrder = DOPPERFFTORDER;		//log2(1024)
	IppsFFTSpec_C_32fc *p_DopplerFFTSpec = 0;
	Ipp8u *pDopplerFFTMemSpec = 0;
	Ipp8u *pDopplerFFTMemInit = 0;
	Ipp8u *pDopplerFFTMemBuffer = 0;
	int sizeDopplerFFTInit = 0;
	int sizeDopperSpec = 0;
	int sizeDopperBuffer = 0;
	/////////////////////////////////////////
	//Doppler FFT Init
	ippsFFTGetSize_C_32fc(nDopplerFFTOrder, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, &sizeDopperSpec, &sizeDopplerFFTInit, &sizeDopperBuffer);
	pDopplerFFTMemSpec = (Ipp8u*)ippsMalloc_8u(sizeDopperSpec);

	if (sizeDopplerFFTInit > 0)
	{
		pDopplerFFTMemInit = (Ipp8u*)ippsMalloc_8u(sizeDopplerFFTInit);
	}
	if (sizeDopperBuffer > 0)
	{
		pDopplerFFTMemBuffer = (Ipp8u*)ippsMalloc_8u(sizeDopperBuffer);
	}
	/// initialize FFT specification structure
	ippsFFTInit_C_32fc(&p_DopplerFFTSpec, nDopplerFFTOrder, flag, ippAlgHintNone, pDopplerFFTMemSpec, pDopplerFFTMemInit);
	/////////////////////////////////////////////////
	////////////////////////////////////////////
	ippsSet_32f(1.0f, g_fNCI_coe, MAX_PULSE_NUMBER_IN_GROUP);
	/////////////////////////////////////////

	CBLAS_LAYOUT Layout = CblasRowMajor;
	int M = 1;
	unsigned int N = myLength;// stRadarSigProcePlsGrpData.nDistanceUnitNumber;
	int K;
	/////////////////////////////////////////////
	//MTD FFT

	Ipp32fc fcMTI_Coe[MAX_PULSE_NUMBER_IN_GROUP];
	for (int ii = 0; ii < MAX_PULSE_NUMBER_IN_GROUP; ii++)
	{
		fcMTI_Coe[ii].re = MTI_Coe[ii];
		fcMTI_Coe[ii].im = 0.0f;
	}


	Ipp32fc myAlpha, myBeta;
	myAlpha.re = 3.051757812500000e-05f * 4; myAlpha.im = 0.0f;
	myBeta.re = 0.0f; myBeta.im = 0.0f;
	MKL_Complex8 myComplexAlpha;
	myComplexAlpha.real = 1.0f;
	myComplexAlpha.imag = 0.0f;
	/////////////////////////////////////////////////

	ippsSet_32f(1.0f, g_fMTDNCI_coe, MAX_PULSE_NUMBER_IN_GROUP);

	//g_fMTDNCI_coe[0] = 0.0f;
	//g_fMTDNCI_coe[1] = 0.0f;

	//g_fMTDNCI_coe[127] = 0.0f;
	//g_fMTDNCI_coe[126] = 0.0f;
	
	Ipp32fc fcTmp1[MAX_DISTANCE_ELEMENT_NUMBER], fcTmp2[MAX_DISTANCE_ELEMENT_NUMBER];
	////////////////////////////////////////////
	DWORD nWritten = 0;

	ULONG64 nCaijiOffset = 0;
	////////////////////////////////////////////////////////

	unsigned char chBeam = 0;
	//////////////////////////////////////////////////////
	UINT nCaijiCount = 0;
	UINT nIQCaijiCount = 0;
	UINT nFlag = 0;
	/////////////////////////////////////////////////////
	//多普勒算法
	CRspProcess rspProc[4];
	Ipp32f result[8][MAX_RNG_CELL];
	/////////////////////////////////////////////////////
	//切除cfar单元个数
	UINT Naverage_Exci = 21;// 11;// 21;
	UINT Naverage = 16;// 6;// 16;
	UINT Nprotect = 3;// 1;// 3;

	UINT NumTemp_Exci = Nprotect + Naverage_Exci;
	//控制多普勒切除cfar
	UINT cfar_Sym1 = 1;
	UINT xishu_CA1 = 900 / 100.f;
	/////////////////////////////////////////////
	IppiSize srcInvRoi = { MAX_DISTANCE_ELEMENT_NUMBER, MAX_PULSE_NUMBER_IN_GROUP };
	/////////////////////////////////////////////

	while (1)
	{
		//TRACE("BeamID:%d\n", nBeamID);
		//读取管道中的数据
		ReadFile(hRadarSBSigProcPipeRead, &stuS4BPulseGrpDesc, sizeof(S4BPULSEGROUPDESC), &nReaded, NULL);

		//if (stuS4BPulseGrpDesc.sbHBData[0].nIQData[0][16] == 0 && stuS4BPulseGrpDesc.sbHBData[1].nIQData[0][16] == 0 && stuS4BPulseGrpDesc.sbHBData[2].nIQData[0][16] == 0)
		//{
		//	int aa = 0;
		//	TRACE("BEAMID:%d,1\n",nBeamID);
		//}
		//else
		//{
		//	int bb = 0;
		//	TRACE("BEAMID:%d,2\n", nBeamID);
		//}

		//ippsZero_32f(p_fHuiboData, myLength * 2);
		for (chBeam = 0; chBeam < 4; chBeam++)
		{

			for (int ii = 0; ii < stuS4BPulseGrpDesc.PulseGroupHead.stuPGHeader.nPulseNumber; ii++)		//对脉组内每个脉冲进行脉压,最大一共1024个脉冲
			{
				stSBRadarSigProcePlsGrpData[chBeam].nPulseNumber = stuS4BPulseGrpDesc.PulseGroupHead.stuPGHeader.nPulseNumber;
				stSBRadarSigProcePlsGrpData[chBeam].nDistanceUnitNumber = stuS4BPulseGrpDesc.PulseGroupHead.stuPGHeader.nPeriodLenth;// +(4 - stuPulseGrpDesc.PulseGroupHead.stuPGHeader.nPeriodLenth % 4);	//

				//ippsConvert_16s32f((Ipp16s*)(p_fHuibo_4Beams[chBeam]), p_fHuiboData, stSBRadarSigProcePlsGrpData[chBeam].nDistanceUnitNumber * 2);
				ippsConvert_16s32f((Ipp16s*)(stuS4BPulseGrpDesc.sbHBData[chBeam].sData[ii]), p_fHuiboData, stSBRadarSigProcePlsGrpData[chBeam].nDistanceUnitNumber * 2);


				//2.脉冲压缩
				ippsFFTFwd_CToC_32fc((Ipp32fc*)p_fHuiboData, p_fcFFTSigIQ, p_mySpec, pMemBuffer);

				ippsMul_32fc(p_fcFFTSigIQ, theIpp.m_pMyCoeB, p_fcMaiyaFFTMultResult, myLength);
				IppStatus status = ippsFFTInv_CToC_32fc(p_fcMaiyaFFTMultResult, (Ipp32fc*)(&stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Maiya[ii]), p_mySpec, pMemBuffer);

				ippsMagnitude_32fc((Ipp32fc*)(&stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Maiya[ii]), (Ipp32f*)(&stSBRadarSigProcePlsGrpData[chBeam].fDataMo[ii]), myLength);
			}
		}
		//////////////////////////////////////////////////////
		//caiji
		//if (g_bOneBeamCaijiEnable)
		//{
			//CopyMemory(stCJRadarSigProcePlsGrpData.PulseGroupHead.nHeader, stuPulseGrpDesc.PulseGroupHead.nHeader, 12 * 4);
			//CopyMemory(stCJRadarSigProcePlsGrpData.fcIQData_Maiya, (Ipp32fc*)(&stRadarSigProcePlsGrpData[0].fcIQData_Maiya), MAX_PULSE_NUMBER_IN_GROUP*MAX_DISTANCE_ELEMENT_NUMBER * 2 * 4);		//Complex Float Data  
			//CopyMemory(stCJRadarSigProcePlsGrpData.fcIQData_Maiya_diff, (Ipp32fc*)(&stRadarSigProcePlsGrpData[1].fcIQData_Maiya), MAX_PULSE_NUMBER_IN_GROUP*MAX_DISTANCE_ELEMENT_NUMBER * 2 * 4);		//Complex Float Data  

			//CopyMemory(g_pOneBeamCaijiBuf + nCaijiOffset, &stCJRadarSigProcePlsGrpData, sizeof(CJPULSEGROUPDESC));
			//nCaijiOffset += sizeof(CJPULSEGROUPDESC);
			////nCaijiPacketcount++;

			////if(nOffset>=256*1024*1024)
			//if (nCaijiOffset + sizeof(CJPULSEGROUPDESC) >= CJ_ONEBEAM_SIZE)
			//{
			//g_bOneBeamCaijiEnable = 0;
			//nCaijiOffset = 0;
			//	SetEvent(g_hCaijiOneBeamBufReadyEvent);
			//}
		//}

		/////////////////////////////////////////////////////

		if (nMTD_Mode == 0)
		{
			for (chBeam = 0; chBeam <= 3; chBeam++)			//32个波束的数据
			{
				//3.积累
				N = myLength;// stRadarSigProcePlsGrpData.nDistanceUnitNumber;
				K = stSBRadarSigProcePlsGrpData[chBeam].nPulseNumber;

				cblas_sgemm(Layout, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, g_fNCI_coe, K, (Ipp32f*)(&stSBRadarSigProcePlsGrpData[chBeam].fDataMo[0][0]), N, 0.0, stSBRadarSigProcePlsGrpData[chBeam].fDataResult, N);
				ippsDivC_32f_I(K, stSBRadarSigProcePlsGrpData[chBeam].fDataResult, stSBRadarSigProcePlsGrpData[chBeam].nDistanceUnitNumber);

				//if (chBeam == 0 && nBeamID == 4)
				//{
				//	int aa = 1;
				//}
			}
		}
		else if (nMTD_Mode == 1)
		{
			for (chBeam = 0; chBeam <= 3; chBeam++)			//32个波束的数据
			{
				//三脉冲对消
				for (int ii = 0; ii < stuS4BPulseGrpDesc.PulseGroupHead.stuPGHeader.nPulseNumber - 2; ii++)		//对脉组内每个脉冲进行4波束分离，然后脉压
				{

					ippsSub_32fc((Ipp32fc*)stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Maiya[ii], (Ipp32fc*)(&stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Maiya[ii + 1]), fcTmp1, stSBRadarSigProcePlsGrpData[chBeam].nDistanceUnitNumber);
					ippsSub_32fc((Ipp32fc*)&stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Maiya[ii + 2], (Ipp32fc*)(&stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Maiya[ii + 1]), fcTmp2, stSBRadarSigProcePlsGrpData[chBeam].nDistanceUnitNumber);
					ippsAdd_32fc(fcTmp1, fcTmp2, (Ipp32fc*)&stSBRadarSigProcePlsGrpData[chBeam].fcIQData_DUIXIAO[ii], stSBRadarSigProcePlsGrpData[chBeam].nDistanceUnitNumber);
					ippsMagnitude_32fc((Ipp32fc*)(&stSBRadarSigProcePlsGrpData[chBeam].fcIQData_DUIXIAO[ii]), (Ipp32f*)(&stSBRadarSigProcePlsGrpData[chBeam].fMoData_DUIXIAO[ii]), myLength);


				}
				//3.积累
				N = myLength;// stRadarSigProcePlsGrpData.nDistanceUnitNumber;
				K = stSBRadarSigProcePlsGrpData[chBeam].nPulseNumber - 2;

				cblas_sgemm(Layout, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, g_fNCI_coe, K, (Ipp32f*)(&stSBRadarSigProcePlsGrpData[chBeam].fMoData_DUIXIAO[0][0]), N, 0.0, stSBRadarSigProcePlsGrpData[chBeam].fDataResult, N);
				ippsDivC_32f_I(K, stSBRadarSigProcePlsGrpData[chBeam].fDataResult, stSBRadarSigProcePlsGrpData[chBeam].nDistanceUnitNumber);
			}
		}
		else
		{
			for (chBeam = 0; chBeam <= 3; chBeam++)
			{
				//TRACE("beamid:%d,chBeam:%d\n", nBeamID,chBeam);
				for (int ii = 0; ii < stuS4BPulseGrpDesc.PulseGroupHead.stuPGHeader.nPulseNumber; ii++)		//对脉组内每个脉冲进行32波束分离，然后脉压,最大一共1024个脉冲
				{
					//分别取实部和虚部
					ippsReal_32fc((Ipp32fc*)(&stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Maiya[ii]), (Ipp32f*)(&fIQData_Maiya_DRM_Real[ii]), MAX_DISTANCE_ELEMENT_NUMBER);
					ippsImag_32fc((Ipp32fc*)(&stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Maiya[ii]), (Ipp32f*)(&fIQData_Maiya_DRM_Imag[ii]), MAX_DISTANCE_ELEMENT_NUMBER);
				}
				//1 矩阵转置
				ippiTranspose_32f_C1R((Ipp32f*)(&fIQData_Maiya_DRM_Real[0][0]), MAX_DISTANCE_ELEMENT_NUMBER * 4,
					(Ipp32f*)(&fIQData_Maiya_RDM_Real[0][0]), MAX_PULSE_NUMBER_IN_GROUP * 4, srcInvRoi);
				ippiTranspose_32f_C1R((Ipp32f*)(&fIQData_Maiya_DRM_Imag), MAX_DISTANCE_ELEMENT_NUMBER * 4,
					(Ipp32f*)(&fIQData_Maiya_RDM_Imag), MAX_PULSE_NUMBER_IN_GROUP * 4, srcInvRoi);

				for (int ii = 0; ii < MAX_DISTANCE_ELEMENT_NUMBER; ii++)
				{
					//2 合成复数
					ippsRealToCplx_32f((Ipp32f*)(&fIQData_Maiya_RDM_Real[ii]), (Ipp32f*)(&fIQData_Maiya_RDM_Imag[ii]),
						(Ipp32fc*)(&stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Maiya_RDM[ii]), MAX_PULSE_NUMBER_IN_GROUP);
					//3 fft
					ippsFFTFwd_CToC_32fc((Ipp32fc*)(&stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Maiya_RDM[ii]), (Ipp32fc*)(&stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Doppler_RDM[ii]), p_DopplerFFTSpec, pDopplerFFTMemBuffer);
				}
				//if ((nBeamID == 4) && (chBeam==0))
				//{
				//	CStdioFile f;
				//	CString strTmp;
				//	f.Open(_T("E:\\vccj.txt"), CFile::modeCreate | CFile::modeWrite);
				//	for (int nLL = 0; nLL < MAX_DISTANCE_ELEMENT_NUMBER; nLL++)
				//	{
				//		strTmp = _T("\n");
				//		f.WriteString(strTmp);
				//		for (int nVV = 0; nVV < MAX_PULSE_NUMBER_IN_GROUP; nVV++)
				//		{
				//			strTmp.Format(_T("%.3f %.3f "), stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Doppler_RDM[nLL][nVV].re, stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Doppler_RDM[nLL][nVV].im);
				//			f.WriteString(strTmp);
				//		}
				//	}
				//	strTmp = _T("\n");
				//	f.WriteString(strTmp);
				//	f.Close();
				//	int nXX = 0;
				//}
				///////////////////////////////////////////////////////
				//if (stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Doppler_RDM[16][0].re == 0 && stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Doppler_RDM[16][0].im == 0.0)
				//{
				//	int aa = 0;
				//	TRACE("beamID:%d,beam:%d,Error\n",nBeamID,chBeam);
				//}
				rspProc[chBeam].DopplerProcess((&stSBRadarSigProcePlsGrpData[chBeam].fcIQData_Doppler_RDM[0][0]), 32, MAX_PULSE_NUMBER_IN_GROUP, MAX_DISTANCE_ELEMENT_NUMBER,
					&(stSBRadarSigProcePlsGrpData[chBeam].Ipp32fSigAmp0_MS[0]), stSBRadarSigProcePlsGrpData[chBeam].Ipp32fSigFreq0_MS, stSBRadarSigProcePlsGrpData[chBeam].Vaild_Flag0,
					stSBRadarSigProcePlsGrpData[chBeam].Ipp32fSigAmp1_MS, stSBRadarSigProcePlsGrpData[chBeam].Ipp32fSigFreq1_MS, stSBRadarSigProcePlsGrpData[chBeam].Vaild_Flag1,
					stSBRadarSigProcePlsGrpData[chBeam].Ipp32fSigAmp2_MS, stSBRadarSigProcePlsGrpData[chBeam].Ipp32fSigFreq2_MS, stSBRadarSigProcePlsGrpData[chBeam].Vaild_Flag2);

				Ipp32f * fData_MTDCfarMax_MS = result[chBeam*2+0];//发送多普勒幅度值
				Ipp32f * Ipp32f_MTDFreq_MS = result[chBeam * 2+1];   //发送多普勒频率值
				////5.两次估算结果选择  发送幅度、多普勒频率
				////估算后通道不做cfar//////////////////////////////
				rspProc[chBeam].DopplerDataSel(stSBRadarSigProcePlsGrpData[chBeam].Ipp32fSigAmp0_MS, stSBRadarSigProcePlsGrpData[chBeam].Ipp32fSigFreq0_MS, stSBRadarSigProcePlsGrpData[chBeam].Vaild_Flag0,
					stSBRadarSigProcePlsGrpData[chBeam].Ipp32fSigAmp1_MS, stSBRadarSigProcePlsGrpData[chBeam].Ipp32fSigFreq1_MS, stSBRadarSigProcePlsGrpData[chBeam].Vaild_Flag1,
					stSBRadarSigProcePlsGrpData[chBeam].Ipp32fSigAmp2_MS, stSBRadarSigProcePlsGrpData[chBeam].Ipp32fSigFreq2_MS, stSBRadarSigProcePlsGrpData[chBeam].Vaild_Flag2,
					MAX_DISTANCE_ELEMENT_NUMBER, stSBRadarSigProcePlsGrpData[chBeam].Ipp32fSigAmpGet_MS, Ipp32f_MTDFreq_MS);
				ippsLog10_32f_A11(stSBRadarSigProcePlsGrpData[chBeam].Ipp32fSigAmpGet_MS, stSBRadarSigProcePlsGrpData[chBeam].fData0_DoplerLOG_MS, MAX_DISTANCE_ELEMENT_NUMBER);
				ippsMulC_32f_I(20, stSBRadarSigProcePlsGrpData[chBeam].fData0_DoplerLOG_MS, MAX_DISTANCE_ELEMENT_NUMBER);
				//    algoAlgorithm.CA_CFAR((&fData0_DoplerLOG_MS[0]),(&Ipp32fSigAmpGet_MS[0]),(fData_MTDCfarMax_MS),nDistanceUnitNumber,targetSigProFunction.Naverage_CA,targetSigProFunction.Nprotect_CA,targetSigProFunction.NumTemp_CA,targetSigProFunction.cfar_Sym1,targetSigProFunction.xishu_CA1);
				rspProc[chBeam].CFAR_Exci((&stSBRadarSigProcePlsGrpData[chBeam].fData0_DoplerLOG_MS[0]), (&stSBRadarSigProcePlsGrpData[chBeam].Ipp32fSigAmpGet_MS[0]), (fData_MTDCfarMax_MS), MAX_DISTANCE_ELEMENT_NUMBER,
					Naverage_Exci, Nprotect, NumTemp_Exci, Naverage, cfar_Sym1, xishu_CA1);

				CopyMemory((Ipp32f *)stSBRadarSigProcePlsGrpData[chBeam].fDataResult, (Ipp32f *)fData_MTDCfarMax_MS, MAX_DISTANCE_ELEMENT_NUMBER * 4);
				//////////////////////////////////////////////////////
			}
	
		}
		////////////////////////////////////////////////////////
		//3.送给终端
		CopyMemory(&stZDDispData.nHeader[4], stuS4BPulseGrpDesc.PulseGroupHead.nHeader, 12 * 4);
		stZDDispData.nHeader[14] = MAX_DISTANCE_ELEMENT_NUMBER;// stRadarSigProcePlsGrpData[0].nDistanceUnitNumber*MAX_BEAM;
		for (int jj = 0; jj <= 3; jj++)
		{
			ippsThreshold_32f(stSBRadarSigProcePlsGrpData[jj].fDataResult, (Ipp32f *)&g_fTempZhongduanData[jj], stSBRadarSigProcePlsGrpData[jj].nDistanceUnitNumber, (1 << 24) - 1, ippCmpGreater);
			ippsConvert_32f8u_Sfs((Ipp32f *)&g_fTempZhongduanData[jj], (Ipp8u *)&stZDDispData.nData[jj], stSBRadarSigProcePlsGrpData[jj].nDistanceUnitNumber, ippRndNear, 6);

		}

		if (g_hSBTerminalPipeWriteArray[nBeamID])		//写到pipe中
		{
			WriteFile(g_hSBTerminalPipeWriteArray[nBeamID], &stZDDispData, sizeof(stZDDispData), &nWritten, NULL);
		}

		SetEvent(hEvent);
		//TRACE("beamid:%d,chBeam:%d\n", nBeamID, chBeam);
	}


	ippsFree(p_fMyaiyaResult);
	ippsFree(p_fcMaiyaFFTMultResult);
	ippsFree(p_fcFFTSigIQ);
	ippsFree(p_fHuiboData);
	ippsFree(pMemInit);
	ippsFree(pMemBuffer);
	ippsFree(pMemSpec);

	ippsFree(pDopplerFFTMemInit);
	ippsFree(pDopplerFFTMemBuffer);
	ippsFree(pDopplerFFTMemSpec);

}
///////////////////////////////////////////


void InitializeIPP(void)
{
	//unsigned int myOrder = 16, myLength, n;
	//int myBufferSize, myBufferSize1;
	//Ipp8u *myBuffer, *myBuffer1;
	//Ipp32fc *SigA, *SigB, *CoeA, *CoeB, *SigC;
	//IppsFFTSpec_C_32fc *mySpec, *mySpec1;
	//Ipp32f *MaiyaResult;
	//myLength = 1 << 12;		//4096 points
	//ippsFFTInitAlloc_C_32fc(&mySpec, myOrder, IPP_FFT_NODIV_BY_ANY, ippAlgHintFast);
	//ippsFFTInitAlloc_C_32fc(&mySpec1, myOrder, IPP_FFT_NODIV_BY_ANY, ippAlgHintFast);

	//myBufferSize = 0;
	//ippsFFTGetBufSize_C_32fc(mySpec, &myBufferSize);
	//ippsFFTGetBufSize_C_32fc(mySpec1, &myBufferSize);
	//myBuffer = ippsMalloc_8u(myBufferSize);
	//myBuffer1 = ippsMalloc_8u(myBufferSize);
	//SigA = ippsMalloc_32fc(myLength);
	//SigB = ippsMalloc_32fc(myLength);
	//CoeA = ippsMalloc_32fc(myLength);
	//CoeB = ippsMalloc_32fc(myLength);
	//SigC = ippsMalloc_32fc(myLength);
	//MaiyaResult = ippsMalloc_32f(myLength);

	//double t = 0.0;
	////for (unsigned int i = 0; i < myLength; ++i) 
	////{
	////	if (i<FS*TAO)
	////	{
	////		SigA[i].re = cos(-2 * PI*BW / 2.0f*t + 2 * PI*1.0f / 2 * BW / TAO*t*t);
	////		SigA[i].im = sin(-2 * PI*BW / 2.0f*t + 2 * PI*1.0f / 2 * BW / TAO*t*t);
	////	}
	////	else
	////	{
	////		SigA[i].re = 0;
	////		SigA[i].im = 0;
	////	}

	////	t = t + 1.0f / FS;
	////}

	//t = 0.0;
	//for (unsigned int i = 0; i < myLength; ++i) 
	//{
	//	if (i<FS*TAO)
	//	{
	//		CoeA[i].re = (0.54 - 0.46f*cos(2.0*PI*i / (FS*TAO)))*cos(-2.0f*PI*BW / 2.0f*t + 2.0f*PI * 1 / 2 * BW / TAO*t*t);
	//		CoeA[i].im = (0.54 - 0.46f*cos(2.0*PI*i / (FS*TAO)))*(-sin(-2.0f*PI*BW / 2.0f*t + 2.0f*PI * 1 / 2 * BW / TAO*t*t));
	//		//				TRACE("%f\n",(i*2.0*PI/FS*TAO));
	//	}
	//	else
	//	{
	//		CoeA[i].re = 0;
	//		CoeA[i].im = 0;
	//	}
	//	t = t + 1.0f / FS;
	//}

	////计算系数的FFT，并存入CoeB
	//ippsFFTFwd_CToC_32fc(CoeA, CoeB, mySpec, myBuffer);

	////for (int nCount = 0; nCount<4; nCount++)
	////{
	////	ippsFFTFwd_CToC_32fc(SigA, SigB, mySpec, myBuffer);
	////	ippsMul_32fc(SigB, CoeB, SigC, myLength);
	////	ippsFFTInv_CToC_32fc_I(SigC, mySpec1, myBuffer1);
	////	ippsMagnitude_32fc(SigC, MaiyaResult, myLength);

	////}


	//ippsFree(SigA);
	//ippsFree(SigB);
	//ippsFree(CoeA);
	//ippsFree(CoeB);
	//ippsFree(SigC);
	//ippsFree(MaiyaResult);
	//ippsFree(myBuffer);
	//ippsFFTFree_C_32fc(mySpec);
	//ippsFree(myBuffer1);
	//ippsFFTFree_C_32fc(mySpec1);
}

#ifdef NEW_TERMINAL
UINT TerminalTxThread(LPVOID pParam)
{
	DWORD nReaded = 0;
	ZhongduanDisplayData stZDDispData;
	ZhongduanDisplayData1Beam stZDDispData1Beam;
	VideoToNRXGUI videoMsg;

	VideoToGUI videoMsg_Old;

	SOCKET sendSocket;

	//CSocket sockSend;

	//sockSend.Create(0x2000, SOCK_DGRAM, NULL);
	//sockSend.Bind(0x2000, _T("127.0.0.2"));

	sendSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (sendSocket == INVALID_SOCKET)
	{
		TRACE(_T("不能创建发送的SOCKET!!!"));
	}

	sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_port = htons(0x2001);
#ifdef LOCAL_IP127
	addr.sin_addr.s_addr = inet_addr("127.0.0.2");
#else
	addr.sin_addr.s_addr = inet_addr("192.168.6.254");
#endif
	if (bind(sendSocket, (LPSOCKADDR)&addr, sizeof(addr)) == SOCKET_ERROR)
	{

		closesocket(sendSocket);
		sendSocket = INVALID_SOCKET;
		TRACE("bind error\n");
		return 1;
	}

	sockaddr_in RecvAddr;
	RecvAddr.sin_family = AF_INET;
	RecvAddr.sin_port = htons(0x2000);
#ifdef LOCAL_IP127
	RecvAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
#else
	RecvAddr.sin_addr.s_addr = inet_addr("239.168.6.189");
#endif

	sockaddr_in RecvAddr_Old;
	RecvAddr_Old.sin_family = AF_INET;
	RecvAddr_Old.sin_port = htons(0x2000);
#ifdef LOCAL_IP127
	RecvAddr_Old.sin_addr.s_addr = inet_addr("127.0.0.1");
#else
	RecvAddr_Old.sin_addr.s_addr = inet_addr("192.168.6.254");
#endif

	//unsigned short azi_table[] =
	//{ 47462, 48337, 49265, 50244, 51271, 52342, 53455, 54605, 55789,
	//57004, 58245, 59509, 60791, 62089, 63397, 64712, 494, 1810,
	//3120, 4420, 5707, 6976, 8223, 9445, 10637, 11796, 12918,
	//14000, 15038, 16030, 16971, 17859 };
	//unsigned short azi_table[] =
	//{ 50972, 50972, 50972, 50972, 52744, 54825, 56367, 57671,
	//58837, 59912, 60924, 61892, 62828, 63742, 64642, 0,
	//893, 1793, 2707, 3643, 4611, 5623, 6698, 7864,
	//9168, 10710, 12791, 14563, 14563, 14563, 14563, 14563 };
	//unsigned short azi_table[] ={
	//	50972, 50972, 50972, 50972, 50972, 50972, 50972, 53541,
	//	55887, 57671, 59204, 60592, 61892, 63134, 64343, 0,
	//	1192, 2401, 3643, 4943, 6331, 7864, 9648, 11994,
	//	14563, 14563, 14563, 14563, 14563, 14563, 14563, 14563};
	//unsigned short azi_table[] = {
	//	41965, 41965, 41965, 41965, 41965, 41965, 41965, 43690,
	//	46421, 49152, 51882, 54613, 57344, 60074, 62805, 0,
	//	2730, 5461, 8192, 10922, 13653, 16384, 19114, 21845,
	//	23570, 23570, 23570, 23570, 23570, 23570, 23570, 23570 };
	unsigned short azi_table[] = {
		41965, 41965, 41965, 41965, 43008, 45056, 47104, 49152,
		51200, 53248, 55296, 57344, 59392, 61440, 63488, 0,
		2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
		18432, 20480, 22528, 23570, 23570, 23570, 23570, 23570 };


	unsigned short nAzi;
	float fAzi;
	DWORD dwTemp,dwTemp1;

	float bochang = BOCHANG;
	float d = JIANJU;

	int nAzmCode;
		
	float rAzm;
	int kk = 0;


	while (1)
	{
		WaitForMultipleObjects(BEAM_GROUP, g_hSigProcEvent, TRUE, INFINITE);
		//kk++;
		//if (kk != 7)
		//{
		//	for (int ii = 0; ii <= BEAM_GROUP-1; ii++)
		//	{
		//		//读取管道中的数据
		//		ReadFile(g_hSBTerminalPipeReadArray[ii], &stZDDispData, sizeof(S4BZhongduanDisplayData), &nReaded, NULL);
		//		ResetEvent(g_hSigProcEvent[ii]);
		//	}
		//	continue;
		//}
		//kk = 0;
		for (int ii = 0; ii <= BEAM_GROUP-1; ii++)
		{
			//读取管道中的数据
			ReadFile(g_hSBTerminalPipeReadArray[ii], &stZDDispData, sizeof(S4BZhongduanDisplayData), &nReaded, NULL);
			ResetEvent(g_hSigProcEvent[ii]);

			for (int jj = 0; jj <= 3; jj++)
			{
				videoMsg.CommonHeader.wCOUNTER = htons(stZDDispData.nHeader[4] & 0xffff);
				dwTemp = (stZDDispData.nHeader[6] & 0x1fffffff);
				dwTemp1 = dwTemp / 10000;

				//TRACE("FPGA TIME:%x\n", dwTemp);
				//TRACE("dwTxSecondTime:%x\n", dwTemp1);
				//TRACE("dwTxMicroSecondTime:%x\n", (dwTemp - dwTemp1 * 10000) * 100);
				//修改绝对时间为GPS时间//20241125 zhou
				dwTemp = (stZDDispData.nHeader[7] & 0x3fffffff) / 10;//0.1ms->1ms
				//TRACE("FPGA TIME ms:%d\n", dwTemp);
				int h = dwTemp / 1000 / 60 / 60;
				int min = (dwTemp - (h * 60 * 60 * 1000)) / 1000 / 60;
				int sec = (dwTemp - h * 3600000 - min * 60000) / 1000+ii*0.5+jj;
				//TRACE("FPGA TIME h:%d;min:%d;sec:%d\n", h, min, sec);


				videoMsg.CommonHeader.dwTxSecondTime = htonl(dwTemp);		//FPGA Time//修改位FPGA GPS时间
				//videoMsg.CommonHeader.dwTxMicroSecondTime = htonl((dwTemp - dwTemp1*10000)*100);		//FPGA Time
				videoMsg.CommonHeader.dwTxMicroSecondTime = htonl(dwTemp);		//FPGA Time



				videoMsg.RadarVideoHeader.dwTxAbsSecondTime = htonl(dwTemp);
				videoMsg.RadarVideoHeader.dwTxAbsMicroSecondTime = htonl(0);

				videoMsg.RadarVideoHeader.dwTxRelMilliSecondTime_H = htonl(0);
				videoMsg.RadarVideoHeader.dwTxRelMilliSecondTime_L = htonl(dwTemp);

				//videoMsg.RadarVideoHeader.wAbsCourse = 0 * 65536 / 360; //绝对航向
				//videoMsg.RadarVideoHeader.wRelCourse = 116 * 65536 / 360; //相对航向

				//nAzmCode = (stZDDispData.nHeader[8] & 0xffff);
				nAzmCode = (azi_table[31-(ii*4+jj)] & 0xffff);

				if (nAzmCode > 32768)
					nAzmCode -= 65536;

				//rAzm = asin((nAzmCode * BOCHANG) / (65536 * JIANJU))/3.1415926*180.0f;  //20200829
				//rAzm = 60 + asin((nAzmCode * bochang0) / (65536 * JIANJU)) / 3.1415926*180.0f;//加入雷达安装偏角
				rAzm = 60 + asin((nAzmCode * bochang0) / (65536 * JIANJU)) / 3.1415926*180.0f;//加入雷达安装偏角
				//rAzm = 0 + asin((nAzmCode * bochang0) / (65536 * JIANJU)) / 3.1415926*180.0f;//加入雷达安装偏角
				//rAzm = 122 + asin((nAzmCode * BOCHANG) / (65536 * JIANJU)) / 3.1415926*180.0f;//加入雷达安装偏角
				//---------------------------------
				if (rAzm < 0)
					rAzm += 360.f;

				TRACE("rAzm:%d,%.3f\n", nAzmCode, rAzm);


				dwTemp = UINT16(rAzm / 360.0 * 65536.0f);
				videoMsg.RadarVideoHeader.wAziCode = htons(dwTemp);

				int xiuZheng =  TAO_US * 150 / 38.4;// +TAO_US * 15 / 4.8;//零距
				//CopyMemory(videoMsg.bytVideoData, stZDDispData.nData+688, sizeof(stZDDispData.nData)-688);//20us
				CopyMemory(videoMsg.bytVideoData, stZDDispData.nData[jj] + xiuZheng, stZDDispData.nHeader[14] - xiuZheng);//5us

				//for (int ii = 0; ii < 10; ii++)
				//{
				//	rAzm += ii*0.2;

				dwTemp = UINT16(rAzm / 360.0 * 65536.0f);
				videoMsg.RadarVideoHeader.wAziCode = htons(dwTemp);
				sendto(sendSocket, (char *)&videoMsg, sizeof(videoMsg), 0, (SOCKADDR *)&RecvAddr, sizeof(RecvAddr));
				//}
				/////////////////////////////////////

				//			CopyMemory(videoMsg_Old.data.nHeader, stZDDispData.nHeader, sizeof(stZDDispData.nHeader));
				//			CopyMemory(videoMsg_Old.data.nData, stZDDispData.nData[jj], stZDDispData.nHeader[14]);



				//videoMsg.RadarVideoHeader.wAziCode = htons(dwTemp);
				//sockSend.SendTo(&stZDDispData, 64 + stZDDispData.nHeader[14], 0x2001, _T("127.0.0.3"), 0);
				//sendto(sendSocket, (char *)&stZDDispData, 64 + stZDDispData.nHeader[14], 0, (SOCKADDR *)&RecvAddr, sizeof(RecvAddr));
				//			sendto(sendSocket, (char *)&videoMsg_Old, sizeof(videoMsg_Old), 0, (SOCKADDR *)&RecvAddr_Old, sizeof(RecvAddr_Old));
				//sendSocket.SendTo(&stZDDispData, 64 + stZDDispData.nHeader[14], 0x2000, _T("127.0.0.1"), 0);
				
				//for (int ll = 0; ll < 5000; ii++)
				//{
				//}
				//if (jj & 0x01)
				//{
				//	Sleep(1);
				//}
			}
			//Sleep(1);
		}
	}
}
#else
UINT TerminalTxThread(LPVOID pParam)
{
	DWORD nReaded = 0;
	ZhongduanDisplayData stZDDispData;
	ZhongduanDisplayData1Beam stZDDispData1Beam;
	VideoToGUI videoMsg;

	SOCKET sendSocket;

	//CSocket sockSend;

	//sockSend.Create(0x2000, SOCK_DGRAM, NULL);
	//sockSend.Bind(0x2000, _T("127.0.0.2"));

	sendSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	if (sendSocket == INVALID_SOCKET)
	{
		TRACE(_T("不能创建发送的SOCKET!!!"));
	}

	sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_port = htons(0x2001);
#ifdef LOCAL_IP127
	addr.sin_addr.s_addr = inet_addr("127.0.0.2");
#else
	addr.sin_addr.s_addr = inet_addr("192.168.6.254");
#endif
	if (bind(sendSocket, (LPSOCKADDR)&addr, sizeof(addr)) == SOCKET_ERROR)
	{

		closesocket(sendSocket);
		sendSocket = INVALID_SOCKET;
		TRACE("bind error\n");
		return 1;
	}

	sockaddr_in RecvAddr;
	RecvAddr.sin_family = AF_INET;
	RecvAddr.sin_port = htons(0x2000);
#ifdef LOCAL_IP127
	RecvAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
#else
	RecvAddr.sin_addr.s_addr = inet_addr("192.168.6.255");
#endif
	unsigned short nAzi;
	float fAzi;

	while (1)
	{
		//读取管道中的数据
		ReadFile(hTerminalPipeRead, &stZDDispData, sizeof(ZhongduanDisplayData), &nReaded, NULL);

		stZDDispData.nData[0] = 0x01;
		stZDDispData.nData[1] = 0x02;
		stZDDispData.nData[2] = 0x03;
		stZDDispData.nData[3] = 0x04;

		//		nAzi = stZDDispData.nData[8] & 0xffff;
		//		fAzi = nAzi >> 4;

		CopyMemory(videoMsg.data.nHeader, stZDDispData.nHeader, sizeof(stZDDispData.nHeader));
		CopyMemory(videoMsg.data.nData, stZDDispData.nData, sizeof(stZDDispData.nData));



		//sockSend.SendTo(&stZDDispData, 64 + stZDDispData.nHeader[14], 0x2001, _T("127.0.0.3"), 0);
		//sendto(sendSocket, (char *)&stZDDispData, 64 + stZDDispData.nHeader[14], 0, (SOCKADDR *)&RecvAddr, sizeof(RecvAddr));
		sendto(sendSocket, (char *)&videoMsg, sizeof(videoMsg), 0, (SOCKADDR *)&RecvAddr, sizeof(RecvAddr));
		//sendSocket.SendTo(&stZDDispData, 64 + stZDDispData.nHeader[14], 0x2000, _T("127.0.0.1"), 0);
	}
}
#endif


UINT TargetSimThread(LPVOID pParam)
{
	TARGETSIMDESC TargetSimDesc0, TargetSimDesc1, TargetSimDesc2, TargetSimDesc3;

	ULONG	nOutput;	// Count written to bufOutpu

	DWORD nReturn = 0;
	BOOL b;

	HANDLE hPcieHandle = (HANDLE)pParam;

	IO_READ_WRITE_DATA theIOData;
	IO_READ_WRITE_DATA pIOData;
	ZeroMemory(&pIOData, sizeof(IO_READ_WRITE_DATA));

	UINT kkkk=0;

	float x, y, r, sita,x0,y0,sita0;
	BOOL bInc = TRUE; 

	//////////////////////////////////////////
	//DMAREADINFO targetsim_dmareadinfo;

	targetsim_dmareadinfo.nHead0 = DMAREADINFOHEAD0;
	targetsim_dmareadinfo.nHead1 = DMAREADINFOHEAD1;
	targetsim_dmareadinfo.nHead2 = DMAREADINFOHEAD2;
	targetsim_dmareadinfo.nHead3 = DMAREADINFOHEAD3;
	targetsim_dmareadinfo.nHead4 = DMAREADINFOHEAD4;
	targetsim_dmareadinfo.nHead5 = DMAREADINFOHEAD5;

	targetsim_dmareadinfo.nInfoType = 0x00000002;			//类型：目标参数
	targetsim_dmareadinfo.nInfoLength = 4;		//以128位（16字节）为单位

	///////////////////////////////////////////
	TargetSimDesc0.wTargetNumber = 0xaa01;
	TargetSimDesc0.wReserved2 = 0x00;// 0x0C6F;
	TargetSimDesc0.wReserved1 = 1;
	TargetSimDesc0.wEle = 0x0000;	//20200416 1365;		//0deg
	TargetSimDesc0.wAzi = 0x2000;						//-60°~+60°  只用高12位,0-4096 对应-60~+60°
	TargetSimDesc0.wDist = 2500;
	TargetSimDesc0.wAmp = 0x1000;// 0x0100;

	TargetSimDesc1.wTargetNumber = 0xaa02;
	TargetSimDesc1.wReserved2 = 0x0C6F;
	TargetSimDesc1.wReserved1 = 1;
	TargetSimDesc1.wEle = 0x6000;	//20200416 1365;
	TargetSimDesc1.wAzi = 0x6000;
	TargetSimDesc1.wDist = 2500;
	TargetSimDesc1.wAmp = 0x100;

	TargetSimDesc2.wTargetNumber = 0xaa03;
	TargetSimDesc2.wReserved2 = 0;
	TargetSimDesc2.wReserved1 = 0;
	TargetSimDesc2.wEle = 1365;
	TargetSimDesc2.wAzi = 0x9000;
	TargetSimDesc2.wDist = 3500;
	TargetSimDesc2.wAmp = 0x1000;

	TargetSimDesc3.wTargetNumber = 0xaa04;
	TargetSimDesc3.wReserved2 = 0;
	TargetSimDesc3.wReserved1 = 0;
	TargetSimDesc3.wEle = 1365;
	TargetSimDesc3.wAzi = 0x7000;
	TargetSimDesc3.wDist = 4500;
	TargetSimDesc3.wAmp = 0x1000;

	UINT nLength, nLength1;

#define RR 2000
#define OFF 3500

	x = OFF - RR;

	sita0 = 0;

	while (g_bCaiji)
	{
		Sleep(200);
		kkkk = kkkk + 1;

		sita0 = sita0 + 5;
		sita0 = fmod(sita0, 360);

		x0 = RR*cos(sita0 / 180.0f * 3.1415926f);
		y0 = RR*sin(sita0 / 180.0f * 3.1415926f);

		x = x0 + OFF;
		y = y0;

		//y = RR * RR - (x - OFF)*(x - OFF);
		r = x*x + y*y;
		//y = sqrt(y);
		r = sqrt(r);
		sita = atan(y / x);

		sita = sita / 3.1415926f*180.0f;

		//TRACE("sita:%.3f\n", sita);
		sita = sita / 60 * 32768 + 0x8000;
				
		
		if (x >= OFF + RR)
		{
			bInc = FALSE;
		}
		else if (x <= OFF - RR)
		{
			bInc = TRUE;
		}

		//TargetSimDesc1.wDist = WORD(r);			lvwx 0416
//		TargetSimDesc0.wAzi = WORD(sita);
		

		//if (kkkk % 16 == 0)
//		if (TargetSimDesc0.wDist > 5000)
//		{
//			TargetSimDesc0.wDist = 1000;
//		}
//		else
//		{
//			TargetSimDesc0.wDist = TargetSimDesc0.wDist +10 ;
//		}
		CopyMemory(&targetsim_dmareadinfo.pInfo[0 * sizeof(TARGETSIMDESC)], &TargetSimDesc0, sizeof(TARGETSIMDESC));
		CopyMemory(&targetsim_dmareadinfo.pInfo[1 * sizeof(TARGETSIMDESC)], &TargetSimDesc1, sizeof(TARGETSIMDESC));
		CopyMemory(&targetsim_dmareadinfo.pInfo[2 * sizeof(TARGETSIMDESC)], &TargetSimDesc2, sizeof(TARGETSIMDESC));
		CopyMemory(&targetsim_dmareadinfo.pInfo[3 * sizeof(TARGETSIMDESC)], &TargetSimDesc3, sizeof(TARGETSIMDESC));

		targetsim_dmareadinfo.nInfoLength = 4;		//以128位（16字节）为单位

//		b = WriteFile(hPcieHandle, &targetsim_dmareadinfo, sizeof(targetsim_dmareadinfo), &nReturn, NULL);

		nLength = 128;// dmareadinfo.nInfoLength * 4 + 8;
		nLength1 = ((nLength & 0xff) << 24);
		nLength1 |= ((nLength & 0xFF00) << 8);
		nLength1 |= ((nLength & 0xFF0000) >> 8);
		nLength1 |= ((nLength & 0xFF000000) >> 24);

		//theIOData.nMode = 0;	//32bits
		//theIOData.nAddr = 4;
		//theIOData.nData = nLength1;	//
		//b = DeviceIoControl(hPcieHandle, IOCTL_MEM0_WRITE, &theIOData, sizeof(IO_READ_WRITE_DATA), &pIOData, sizeof(IO_READ_WRITE_DATA), &nReturn, NULL);
		//if (!b)
		//{
		//	AfxMessageBox(_T("targetSim:mem0 reg write Failed"));
		//}
		//////////////////////////////////////
		////4.当写入完成后，写入ready标志，通知FPGA开始DMA读
		//theIOData.nMode = 0;	//32bits
		//theIOData.nAddr = 16;
		//theIOData.nData = 0x20000000;	//
		//b = DeviceIoControl(hPcieHandle, IOCTL_MEM0_WRITE, &theIOData, sizeof(IO_READ_WRITE_DATA), &pIOData, sizeof(IO_READ_WRITE_DATA), &nReturn, NULL);
		//if (!b)
		//{
		//	AfxMessageBox(_T("targetSim:mem0 reg write Failed"));
		//}
	}

	return 1;
}
//////////////////////////////////////////////////////////////////////

