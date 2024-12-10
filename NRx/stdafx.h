
#pragma once

#ifndef _SECURE_ATL
#define _SECURE_ATL 1
#endif

#ifndef VC_EXTRALEAN
#define VC_EXTRALEAN		// �� Windows ͷ���ų�����ʹ�õ�����
#endif

// ���������ʹ��������ָ����ƽ̨֮ǰ��ƽ̨�����޸�����Ķ��塣
// �йز�ͬƽ̨����Ӧֵ��������Ϣ����ο� MSDN��
#ifndef WINVER				// ����ʹ���ض��� Windows XP ����߰汾�Ĺ��ܡ�
#define WINVER 0x0501		// ����ֵ����Ϊ��Ӧ��ֵ���������� Windows �������汾��
#endif

#ifndef _WIN32_WINNT		// ����ʹ���ض��� Windows XP ����߰汾�Ĺ��ܡ�
#define _WIN32_WINNT 0x0501	// ����ֵ����Ϊ��Ӧ��ֵ���������� Windows �������汾��
#endif						

#ifndef _WIN32_WINDOWS		// ����ʹ���ض��� Windows 98 ����߰汾�Ĺ��ܡ�
#define _WIN32_WINDOWS 0x0410 // ��������Ϊ�ʺ� Windows Me ����߰汾����Ӧֵ��
#endif

#ifndef _WIN32_IE			// ����ʹ���ض��� IE 6.0 ����߰汾�Ĺ��ܡ�
#define _WIN32_IE 0x0600	// ����ֵ����Ϊ��Ӧ��ֵ���������� IE �������汾��ֵ��
#endif

#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS	// ĳЩ CString ���캯��������ʽ��

// �ر� MFC ��ĳЩ�����������ɷ��ĺ��Եľ�����Ϣ������
#define _AFX_ALL_WARNINGS

#include <afxwin.h>         // MFC ��������ͱ�׼���
#include <afxext.h>         // MFC ��չ


#include <afxdisp.h>        // MFC �Զ�����

#include <setupapi.h>
#include <initguid.h>
#include <winioctl.h>

#include "..\intrface.h"

#include "ipp.h"
#include "ippac90legacy.h"
//#include "ippac90legacy_redef.h"

#include "ipps.h"

#include "mkl.h"


#ifndef _AFX_NO_OLE_SUPPORT
#include <afxdtctl.h>		// MFC �� Internet Explorer 4 �����ؼ���֧��
#endif
#ifndef _AFX_NO_AFXCMN_SUPPORT
#include <afxcmn.h>			// MFC �� Windows �����ؼ���֧��
#endif // _AFX_NO_AFXCMN_SUPPORT

#include <afxsock.h>            // MFC �׽�����չ
#include <afxcontrolbars.h>

//#define LOCAL_IP127
#define NEW_TERMINAL

typedef struct IO_READ_WRITE_DATA_tag
{
	ULONG nMode;		//0:DWORD,1:WORD,2:BYTE
	UINT nAddr;
	UINT nData;
}IO_READ_WRITE_DATA;

#define  BLOCKCOUNT (1024*16*512UL)
//#define RAM_SIZE 1024*128
#define RAM_SIZE (1024*128UL)

//#define CJ_SIZE (512*1024*1024UL)
#define CJ_SIZE (3800*1024*1024ULL)
//#define CJ_SIZE (512*1024*1024UL)
//#define CJ_SIZE (128*1024*1024UL)
#define CJ_RAW_SIZE (24*CJ_SIZE)

//20241121
#define ONEFILESIZE (512*1024*1024ULL)
#define CJ_ONEBEAM_SIZE (1*ONEFILESIZE)

#ifdef _UNICODE
#if defined _M_IX86
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='x86' publicKeyToken='6595b64144ccf1df' language='*'\"")
#elif defined _M_IA64
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='ia64' publicKeyToken='6595b64144ccf1df' language='*'\"")
#elif defined _M_X64
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='amd64' publicKeyToken='6595b64144ccf1df' language='*'\"")
#else
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#endif
#endif

#define DISPLAYCOUNT (1024)
#define SAMPLE_FREQ_MHZ (125.0f)

UINT DMACaiji(LPVOID pParam);
UINT BoKongThread(LPVOID nParam);
UINT ActiveRadarDSPThread(LPVOID pParam);
UINT RadarSigProcThread(LPVOID pParam);
UINT TerminalTxThread(LPVOID pParam);
UINT TargetSimThread(LPVOID pParam);
static UINT SingleBeamProcThread(LPVOID pParam);

//20241121
extern BYTE *g_pOneBeamCaijiBuf;
extern HANDLE g_hCaijiOneBeamBufReadyEvent;

typedef struct tagTHREADPARAMS
{
	HWND hWnd;
	HANDLE hPcieHandle;
	UINT nBokongEnable;
	HANDLE hEvent;
	HANDLE hCaijiBufReadyEvent;
	BYTE *pBuf;
}THREADPARAMS;

typedef struct tagS4BPROCTHREADPARAMS
{
	HANDLE hRadarSBSigProcPipeRead;
	HANDLE hEvent;
	HANDLE hCaijiBufReadyEvent;
	UINT nSigProcParam;
	Ipp32f *p_fHuibo_4Beams[4];
	UINT nBeamID;
}S4BPROCTHREADPARAMS;

typedef struct tagMSGPARAMS
{
	LONGLONG  nPacketCount;
	LONGLONG  nTimeDiff;
	LONGLONG  nFreq;
	
}MSGPARAMS;

typedef struct tagDMAREADINFO		//from PC to board
{
	UINT nHead0;
	UINT nHead1;
	UINT nHead2;
	UINT nHead3;
	UINT nHead4;
	UINT nHead5;
	UINT nInfoType;
	UINT nInfoLength;
	UCHAR pInfo[65536];
}DMAREADINFO;

typedef struct tagBKINFO
{
	UINT nInfoData[16];
}BKINFO;

#define DMAREADINFOHEAD0 0x072495bc
#define DMAREADINFOHEAD1 0x00090009
#define DMAREADINFOHEAD2 0xa5a51234
#define DMAREADINFOHEAD3 0x55aa9966
#define DMAREADINFOHEAD4 0x00000000
#define DMAREADINFOHEAD5 0x00000000

typedef struct tagBKDESC		//����������
{
	//0
	UINT nPacketCount;	//ÿ100ms��1
	//1
	UINT nWorkMode;		//4bits
	UINT nPulseGroup;	//12bits	
	UINT nSigType;		//4bits
	UINT nTxAtten;		//3bits
	UINT nReserved0;	//9bits
	//2
	UINT nFreqPoint;	//12bits
	UINT nPulseWidth;	//20bits
	//3
	UINT nPRI;			
	//4
	UINT nLFMSTARTWORD;
	//5
	UINT nLFMINCWORD;
	//6
	UINT nTxAziCode;
	UINT nTxEleCode;
	//7
	UINT nTxChSel;
	UINT nRxEleCode;
	//8
	UINT nBeam0Code;
	UINT nBeam1Code;
	//9
	UINT nBeam2Code;
	UINT nBeam3Code;
	//10
	UINT nBeam4Code;
	UINT nBeam5Code;
	//11
	UINT nBeam6Code;
	UINT nBeam7Code;
	//12
	UINT nBeam8Code;
	UINT nBeam9Code;
	//13
	UINT nBeam10Code;
	UINT nBeam11Code;
	//14
	UINT nPitchCode;
	UINT nRollCode;
	//15
	UINT nCourseCode;
	UINT nReserved1;

}BKDESC;

void InitializeBKScanTable(void);		//��ʼ������ɨ���
int InitializeActiveRadarDSPPipe(void);	//��ʼ���źŴ����pipe

//#define BK_DESC_TOTAL_NUMBER 10 //768			//��Ĳ����������������跽λ����4�㣬24��Ϊ0�����ǲ�λ����λ���ǡ�48��
//#define BK_DESC_TOTAL_NUMBER 1 //768			//��Ĳ����������������跽λ����4�㣬24��Ϊ0�����ǲ�λ����λ���ǡ�48��
#define BK_DESC_TOTAL_NUMBER 60 //768			//��Ĳ����������������跽λ����4�㣬24��Ϊ0�����ǲ�λ����λ���ǡ�48��
//#define BK_DESC_TOTAL_NUMBER 60*2*40 //768			//��Ĳ����������������跽λ����4�㣬24��Ϊ0�����ǲ�λ����λ���ǡ�48��
BKDESC bk_table[BK_DESC_TOTAL_NUMBER];
extern int g_bk_desc_total_number_final;


typedef struct tagTARGETSIMDESC
{
	WORD wReserved0;
	WORD wTargetNumber;
	WORD wReserved2;		//Low 16bits
	WORD wReserved1;		//High 16bits
	WORD wAzi;
	WORD wEle;
	WORD wAmp;
	WORD wDist;
}TARGETSIMDESC;

extern volatile  BOOL g_bCaiji;
extern volatile UINT g_bCaijiEnable;
extern volatile UINT g_bOneBeamCaijiEnable;

HANDLE hActiveRadarDSPPipeRead;			//pipe1
HANDLE hActiveRadarDSPPipeWrite;
HANDLE g_hSigProcPipeReadArray[8];
HANDLE g_hSigProcPipeWriteArray[8];
HANDLE g_hSBTerminalPipeWriteArray[8];
HANDLE g_hSBTerminalPipeReadArray[8];
HANDLE g_hSigProcEvent[8];

#define HUIBOINFOHEAD0 0x072495bc
#define HUIBOINFOHEAD1 0x00090009
#define HUIBOINFOHEAD2 0xa5a51234
#define HUIBOINFOHEAD3 0x55aa9966

typedef struct tagPULSEGROUPHEADER
{
	UINT nMTPCount;
	UINT nWorkMode;
	UINT nTimeHigh;
	UINT nTimeLow;
	USHORT nTxEle;
	USHORT nTxAzi;
	USHORT nRxBeam1Azi;
	USHORT nRxBeam2Azi;
	USHORT nRxBeam3Azi;
	USHORT nRxBeam4Azi;
	UINT nTstWORD1;
	UINT nTstWORD2;
	UINT nPulseWidth;
	UINT nPeriodLenth;
	UINT nPulseNumber;
}PULSEGROUPHEADER;

#define HUIBOHEADERLEN 12
typedef struct tagINT16IQ
{
	short dI;
	short dQ;
}INT16IQ;

//typedef union tagHUIBODATA
//{
//	UINT nIQData[128][8192];		//4������
//	short sData[128][16384];		//4������
//	INT16IQ dataIQ[128][8192];	//4������
//}HUIBODATA;

//typedef union tagHUIBODATA
//{
//	UINT nIQData[128][16384];		//4������
//	short sData[128][32768];		//4������
//	INT16IQ dataIQ[128][16384];	//4������
//}HUIBODATA;
#define PULSEGROUPDEF 2048

#define MAX_BEAM 32		//32
#if PULSEGROUPDEF==2048
#define MAX_PULSE_NUMBER_IN_GROUP 2048 //1024		//2048		//1024		//1024
#define MAX_DISTANCE_ELEMENT_NUMBER 512	//1024		//512	//1024	//1024 //(16384/4)
#define PC_FFT_ORDER 9  //10		//9			//log2(MAX_DISTANCE_ELEMENT_NUMBER)
#define DOPPERFFTORDER 11 //10		//11		//log2(MAX_PULSE_NUMBER_IN_GROUP)
#elif PULSEGROUPDEF==4096
#define MAX_PULSE_NUMBER_IN_GROUP 4096 //1024		//2048		//1024		//1024
#define MAX_DISTANCE_ELEMENT_NUMBER 256	//1024		//512	//1024	//1024 //(16384/4)
#define PC_FFT_ORDER 8  //10		//9			//log2(MAX_DISTANCE_ELEMENT_NUMBER)
#define DOPPERFFTORDER 12 //10		//11		//log2(MAX_PULSE_NUMBER_IN_GROUP)	
#else
#define MAX_PULSE_NUMBER_IN_GROUP 1024		//2048		//1024		//1024
#define MAX_DISTANCE_ELEMENT_NUMBER 1024		//512	//1024	//1024 //(16384/4)
#define PC_FFT_ORDER 10		//9			//log2(MAX_DISTANCE_ELEMENT_NUMBER)
#define DOPPERFFTORDER 10		//11		//log2(MAX_PULSE_NUMBER_IN_GROUP)
#endif
#define BEAM_GROUP 8
#define BEAM_GROUP 8

#define MAX_RNG_CELL MAX_DISTANCE_ELEMENT_NUMBER
#define FIR_Num MAX_PULSE_NUMBER_IN_GROUP
#define WN_FFT MAX_DISTANCE_ELEMENT_NUMBER



typedef union tagHUIBODATA
{
	UINT nIQData[MAX_PULSE_NUMBER_IN_GROUP][MAX_BEAM*MAX_DISTANCE_ELEMENT_NUMBER];		//32������,ÿ������<1000��,���洢1024������
	short sData[MAX_PULSE_NUMBER_IN_GROUP][MAX_BEAM*MAX_DISTANCE_ELEMENT_NUMBER*2];		//32������
	INT16IQ dataIQ[MAX_PULSE_NUMBER_IN_GROUP][MAX_BEAM*MAX_DISTANCE_ELEMENT_NUMBER];	//32������
}HUIBODATA;

typedef struct tagPULSEGROUPDESC
{
	union {
		UINT nHeader[HUIBOHEADERLEN];
		PULSEGROUPHEADER stuPGHeader;
	}PulseGroupHead;
	
	HUIBODATA hbData;
	
}PULSEGROUPDESC;

typedef union tagSingleBeamHUIBODATA
{
	UINT nIQData[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER];		//1����,ÿ������<1000��,���洢1024������
	short sData[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER * 2];		//1����
	INT16IQ dataIQ[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER];	//1����
}SingleBeamHUIBODATA;

typedef struct tagS4BPULSEGROUPDESC
{
	union {
		UINT nHeader[HUIBOHEADERLEN];
		PULSEGROUPHEADER stuPGHeader;
	}PulseGroupHead;
	SingleBeamHUIBODATA sbHBData[4];
}S4BPULSEGROUPDESC;


typedef struct tagRadarSigProcPulseGroupData
{
	Ipp32fc fcIQData_Maiya[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER];
	//Ipp32fc fcIQData_fftMTD[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER];
	Ipp32fc fcIQData_DUIXIAO[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER];
	Ipp32f  fDataMo[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER];
	//Ipp32f  fMTDFFTDataMo[MAX_DISTANCE_ELEMENT_NUMBER][MAX_PULSE_NUMBER_IN_GROUP];
	Ipp32f  fMoData_DUIXIAO[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER];
	Ipp32fc  fcMTIDataResult[MAX_DISTANCE_ELEMENT_NUMBER];
	Ipp32f  fDataResult[MAX_DISTANCE_ELEMENT_NUMBER];

	Ipp32fc fcIQData_Maiya_RDM[MAX_DISTANCE_ELEMENT_NUMBER][MAX_PULSE_NUMBER_IN_GROUP];		//
	Ipp32fc fcIQData_Doppler_RDM[MAX_DISTANCE_ELEMENT_NUMBER][MAX_PULSE_NUMBER_IN_GROUP];		//

	//�������㷨
	Ipp32f Ipp32fSigAmp0_MS[MAX_RNG_CELL];
	Ipp32f Ipp32fSigFreq0_MS[MAX_RNG_CELL];
	Ipp16s Vaild_Flag0[MAX_RNG_CELL];
	Ipp32f Ipp32fSigAmp1_MS[MAX_RNG_CELL];
	Ipp32f Ipp32fSigFreq1_MS[MAX_RNG_CELL];
	Ipp16s Vaild_Flag1[MAX_RNG_CELL];
	Ipp32f Ipp32fSigAmp2_MS[MAX_RNG_CELL];
	Ipp32f Ipp32fSigFreq2_MS[MAX_RNG_CELL];
	Ipp16s Vaild_Flag2[MAX_RNG_CELL];
	Ipp32f Ipp32fSigAmpGet_MS[MAX_RNG_CELL];
	Ipp32f Ipp32fSigAmpCfar0_MS[MAX_RNG_CELL];
	Ipp32f Ipp32fSigAmpCfar1_MS[MAX_RNG_CELL];
	Ipp32f Ipp32fSigAmpCfar2_MS[MAX_RNG_CELL];
	UINT Ipp32fSigAmpCfarFLAg0_MS[MAX_RNG_CELL];
	UINT Ipp32fSigAmpCfarFLAg1_MS[MAX_RNG_CELL];
	UINT Ipp32fSigAmpCfarFLAg2_MS[MAX_RNG_CELL];


	Ipp32f fData0_DoplerLOG_MS[MAX_RNG_CELL];
	Ipp32f fData1_DoplerLOG_MS[MAX_RNG_CELL];
	Ipp32f fData2_DoplerLOG_MS[MAX_RNG_CELL];

	UINT nPulseNumber;
	UINT nDistanceUnitNumber;

}RadarSigProcPulseGroupData;

HANDLE hRadarSigProcPipeRead;			//pipe2
HANDLE hRadarSigProcPipeWrite;

void InitializeIPP(void);
//20241121
/////////////////////////////////////////////
typedef union tagCJHUIBODATA
{
	UINT nIQData[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER];		//1������,ÿ������<1000��,���洢1024������
	short sData[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER * 2];		//1������
	INT16IQ dataIQ[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER];	//1������
}CJHUIBODATA;

typedef struct tagCJPULSEGROUPDESC
{
	union {
		UINT nHeader[HUIBOHEADERLEN];
		PULSEGROUPHEADER stuPGHeader;
	}PulseGroupHead;

	Ipp32fc fcIQData_Maiya[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER];
	Ipp32fc fcIQData_Maiya_diff[MAX_PULSE_NUMBER_IN_GROUP][MAX_DISTANCE_ELEMENT_NUMBER];

}CJPULSEGROUPDESC;
/////////////////////////////////////////

#define FS (SAMPLE_FREQ_MHZ*1.0e6/32)
//#define BW (5e6)
#define BW (3.0e6)
//#define BW (15.0e6)
//#define BW (20e6)
#define BW_MHZ (BW/1e6f)










//#define TAO (0.6e-6)
//#define TAO (0.6e-6)
#define TAO (5.0e-6)
//#define TAO (20.0e-6)
//#define TAO (15.0e-6)
//#define TAO (20.0e-6)
#define TAO_US (TAO*1e6f)


















#define BOCHANG (0.0312f)
#define JIANJU  (0.0135f)

#define PI 3.1415926535f
#define c_speed 2.99792458e8
//////////////////////////////////////////////
#pragma pack(1)
typedef struct IpMsgHead
{
	unsigned int nFlag;   //0x7e7e��Ƶ����  0x7E7F�㼣���� 0x7e80�㼣
	//0x6e6e  �źŴ�����Ʊ���
	//0x5e5e��������
	unsigned int nMsgLen;  //��Ϣ��ĳ���(������ͷ)
	unsigned int nSendNo;
	unsigned int nRcvNo;
	unsigned int nRes;



} *LPIpMsgHead;
//////////////////////////////////////////////
#define ZHONGDUANHEADERLEN 16
typedef struct tagZhongduanDisplayData
{
	UINT nHeader[ZHONGDUANHEADERLEN];
	UINT8 nData[MAX_BEAM][MAX_DISTANCE_ELEMENT_NUMBER];
}ZhongduanDisplayData;

typedef struct tagS4BZhongduanDisplayData
{
	UINT nHeader[ZHONGDUANHEADERLEN];
	UINT8 nData[4][MAX_DISTANCE_ELEMENT_NUMBER];;
}S4BZhongduanDisplayData;

//���͸��Կص���Ƶ���ĸ�ʽ
struct VideoToGUI
{
	IpMsgHead head;
	ZhongduanDisplayData data;

	VideoToGUI()
	{
		head.nFlag = 0x7e7e;
		head.nMsgLen = sizeof(ZhongduanDisplayData);
	}
};

typedef struct tagZhongduanDisplayData1Beam
{
	UINT32 nHeader[ZHONGDUANHEADERLEN];
	UINT8 nData[MAX_DISTANCE_ELEMENT_NUMBER];
}ZhongduanDisplayData1Beam;

HANDLE hTerminalPipeRead;			//pipe3
HANDLE hTerminalPipeWrite;
/////////////////////////////////////////////
#define PCIE2MCU_PKT_HEAD0 (0x072495BC)
#define PCIE2MCU_PKT_HEAD1 (0xa5a51234)
////////////////////////////////////////////
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
														//b3 �����о���ʱ���ʽ. 0��ʱ��1��ʾ32λUTC��ʱ��2��ʾ�����µ�΢������1����ʾ�Զ���.����ʱ��1��ʾ0 - 86399999ms������ʱ��2��Ч��0.
														//b2 - 0��Ԥ������0.
	UINT8 bytRecChannel;								//��¼����ͨ����	��¼�ط�ʹ�õ�ͨ���ţ��������ֶ�ͨ������
	UINT16 wReserved0;									//Ԥ��	��0
	UINT16 wReserved1;									//Ԥ��	��0
	UINT16 wResesrved2;									//Ԥ��	��0
}NRX_COMMON_HEADER;

typedef struct tagNRX_COMMON_TAIL
{
	UINT dwCheckSum;
	UINT16 wTail1;
	UINT16 wTail2;
}NRX_COMMON_TAIL;

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


struct VideoToNRXGUI
{
	NRX_COMMON_HEADER CommonHeader;
	RX_RadarVideo_Head RadarVideoHeader;
	UINT8 bytVideoData[MAX_DISTANCE_ELEMENT_NUMBER];
	NRX_COMMON_TAIL CommonTail;

	VideoToNRXGUI()
	{
		ZeroMemory(&CommonHeader, sizeof(CommonHeader));
		CommonHeader.dwHEADER = htonl(0xF1A2B4C8);
		CommonHeader.wVERSION = htons(0);
		CommonHeader.wMsgTotalLen = htons(sizeof(CommonHeader) + sizeof(RadarVideoHeader)+sizeof(bytVideoData)+sizeof(CommonTail));
		CommonHeader.wMsgFlag = htons(0x0103);
		CommonHeader.wRadarID = htons(0x0012);
		CommonHeader.bytTxNodeNumber = 0x11;
		CommonHeader.bytRxNodeNumber = 0x22;
		CommonHeader.bytDataFlag = 0x08;
		CommonHeader.bytRecChannel = 0;

		ZeroMemory(&RadarVideoHeader, sizeof(RadarVideoHeader));
		RadarVideoHeader.dwSyncHeader = htonl(0xa5a61234);
		RadarVideoHeader.dwVideoLen = htonl(sizeof(bytVideoData));
		RadarVideoHeader.wHeadLen = htons(128);
		RadarVideoHeader.wEncodeFormat = htons(0);
		RadarVideoHeader.bytPulseMode = 0;
		RadarVideoHeader.bytSubPulseNumber = 1;
		RadarVideoHeader.bytSubPulseCount = 0;

		RadarVideoHeader.dwSigBWHz = htonl(3e6);
		RadarVideoHeader.dwSampleFreqHz = htonl(3906250);
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
////////////////////////////////////////////
extern int g_nFreqPoint;

#define ID_MESSAGE_CAIJI WM_USER+100


