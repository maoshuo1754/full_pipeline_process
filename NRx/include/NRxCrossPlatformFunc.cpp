#include "NRxCrossPlatformFunc.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <regex>

#ifdef WIN32
#include <windows.h> // find
#else
#include <sys/time.h>
#endif

#ifdef _MSVC_LANG
#define UINT16MAX 0xFF;
#else
#define UINT16MAX std::numeric_limits<unsigned short>::max();
#endif

///
/// \brief NRxMultiPlatform::getExePath, get exe path.
///
string NRxMultiPlatform::getExePath()
{
    char* buffer;
    string exePath;

#ifdef WIN32
    if ((buffer = _getcwd(NULL, 0)) == NULL) {
        //
        printf("NRxCrossPlatformFunc Warning: cann't get current working directory\n");
    } else {
        exePath = buffer;
        free(buffer);
    }
#else // linux
    if ((buffer = getcwd(NULL, 0)) == NULL) {
        //
        printf("NRxCrossPlatformFunc Warning: cann't get current working directory\n");
    } else {
        exePath = buffer;
        free(buffer);
    }
#endif
    printf("exe path is: %s\n", exePath.c_str());

    return exePath;
}

///
/// \brief NRxMultiPlatform::relPath2AbsPath, turn relative path to absolute path
///     according to exe path
/// \param relPath, "./" or "../"; if input absolute path like "Z:\\",
///     will return directly
/// \return
///     absolute path base on exe path
std::string NRxMultiPlatform::relPath2AbsPath(const std::string& relPath)
{
#ifdef WIN32
    char dir[1024];
    _fullpath(dir, relPath.c_str(), 1024);
#else
    char dir[1024 * 1024];
    auto res = realpath(relPath.c_str(), dir);
#endif
    std::stringstream ss;
    ss << dir;
    return ss.str();
}

bool NRxMultiPlatform::regexMatch(const std::string& reg, const std::string& src)
{
    std::regex re(reg);
    std::cmatch m;
    return std::regex_match(src.c_str(), m, re);
}

bool NRxMultiPlatform::isValidIP(const std::string& ip)
{
    const std::regex ipRegex("^(?:(?:\\d|[1-9]\\d|1\\d\\d|2[0-4]\\d|25[0-5])\\.){3}(?:\\d|[1-9]\\d|1\\d\\d|2[0-4]\\d|25[0-5])$");
    std::smatch ip_match;

    std::regex_match(ip, ip_match, ipRegex);

    return ip_match.size() == 1;
}

unsigned short NRxMultiPlatform::toBCD(unsigned short batchNo)
{
    if(!isBatchValid(batchNo))
    {
        return UINT16MAX;
    }

    const unsigned short maxRem = 9;
    const unsigned int baseIn = 10;
    const unsigned int baseOut = 16;
    unsigned int bcd32(0);
    const std::vector<unsigned int> prd{1, baseOut, baseOut*baseOut, baseOut*baseOut*baseOut};
    std::size_t offset(0);
    unsigned short rem(0);
    while(offset < prd.size())
    {
        rem = batchNo % baseIn;
        if(rem > maxRem )
        {
            return UINT16MAX;
        }
        bcd32 += rem * prd.at(offset);
        batchNo /= baseIn;
        offset++;
    }

    unsigned short bcd16(bcd32);
    if(bcd16 != bcd32)
    {
        std::stringstream msg;
        msg << "ToBCD bcd16 != bcd32, bcd16 =  " << bcd16 << " bcd32 = " << bcd32 << std::endl;
        std::cout << msg.str();
        return UINT16MAX;
    }
    return bcd16;
}

unsigned short NRxMultiPlatform::fromBCD(unsigned short BCDNo)
{
    if(!isBCDValid(BCDNo))
    {
        return UINT16MAX;
    }

    const unsigned short maxRem = 9;
    const unsigned int baseIn = 16;
    const unsigned int baseOut = 10;
    unsigned int bt32(0);
    const std::vector<unsigned int> prd{1, baseOut, baseOut*baseOut, baseOut*baseOut*baseOut};
    std::size_t offset(0);
    unsigned short rem(0);
    while(offset < prd.size())
    {
        rem = BCDNo % baseIn;
        if(rem > maxRem )
        {
            return UINT16MAX;
        }
        bt32 += rem * prd.at(offset);
        BCDNo /= baseIn;
        offset++;
    }

    unsigned short bt16(bt32);
    if(bt16 != bt32)
    {
        std::stringstream msg;
        msg << "BCDto bt16 != bt32, bt16 =  " << bt16 << " bt32 = " << bt32 << std::endl;
        std::cout << msg.str();
        return UINT16MAX;
    }
    return bt16;
}

bool NRxMultiPlatform::isBatchValid(unsigned short batch)
{
    return ((batch >= 1) && (batch <= 9999));
}

bool NRxMultiPlatform::isBCDValid(unsigned short bcd)
{
    return ((bcd >= 0x0001) && (bcd <= 0x9999));
}

void NRxMultiPlatform::truncateDouble(double& val, int precision)
{
    std::stringstream ss;
    ss << std::setiosflags(std::ios_base::fixed)
       << std::setprecision(precision)
       << val;
    ss >> val;
}

double NRxMultiPlatform::getCurLocalTimeOneDayMilliSec()
{
    double ret(-1);

    time_t curtm;
    ::time(&curtm); // get current time
    std::tm* localTime = std::localtime(&curtm); // STC: 1s
    ret = (localTime->tm_hour * 3600 + localTime->tm_min * 60 + localTime->tm_sec) * 1000; // sec of one day, STC: 1ms
#ifdef WIN32
    SYSTEMTIME sys;
    GetLocalTime(&sys); // 获取Windows平台时间
    ret += sys.wMilliseconds;
#else
    timeval now;
    gettimeofday(&now, NULL);
    ret += now.tv_usec / 1000.f;
#endif

    return ret;
}

double NRxMultiPlatform::getCurLocalTimeUTCMilliSec()
{
    double ret(-1);

    time_t curtm;
    ::time(&curtm); // get current time
    std::localtime(&curtm);
    ret = curtm * 1000.f; // UTC, STC: 1ms
#ifdef WIN32
    SYSTEMTIME sys;
    GetLocalTime(&sys); // 获取Windows平台时间
    ret += sys.wMilliseconds;
#else
    timeval now;
    gettimeofday(&now, NULL);
    ret += now.tv_usec / 1000.f;
#endif

    return ret;
}

double NRxMultiPlatform::getCurLocalTimeOneDayMicroSec()
{
    double ret(-1);

    time_t curtm;
    ::time(&curtm); // get current time
    std::tm* localTime = std::localtime(&curtm); // STC: 1s
    ret = (localTime->tm_hour * 3600 + localTime->tm_min * 60 + localTime->tm_sec) * 1e6; // sec of one day, STC: 1ms
#ifdef WIN32
    SYSTEMTIME sys;
    GetLocalTime(&sys); // 获取Windows平台时间
    ret += sys.wMilliseconds * 1000;
#else
    timeval now;
    gettimeofday(&now, NULL);
    ret += now.tv_usec;
#endif

    return ret;
}

double NRxMultiPlatform::getCurLocalTimeUTCMicroSec()
{
    double ret(-1);

    time_t curtm;
    ::time(&curtm); // get current time
    std::localtime(&curtm);
    ret = curtm * 1e6; // UTC, STC: 1us
#ifdef WIN32
    SYSTEMTIME sys;
    GetLocalTime(&sys); // 获取Windows平台时间
    ret += sys.wMilliseconds * 1000;
#else
    timeval now;
    gettimeofday(&now, NULL);
    ret += now.tv_usec;
#endif

    return ret;
}

double NRxMultiPlatform::getSelfCntMilliSec(double alarmMilliSecThr)
{
    double microSec = NRxMultiPlatform::getCurLocalTimeUTCMicroSec();
    static double UTCMicroSecPre(microSec); // ONLY CALL ONCE DURING ONE PROCESS
    static double relTime(0);
    double deltaTime = (microSec - UTCMicroSecPre) / 1e3;
    UTCMicroSecPre = microSec;
    if (deltaTime > alarmMilliSecThr) {
        printf("time jump, alarm millisec: %.2f, cur micro sec: %.4f, "
               "pre micro sec: %.4f, delta t is: %.4f", alarmMilliSecThr, microSec, UTCMicroSecPre, deltaTime);
        // std::cout << "time jump, alarm millisec: " << alarmMilliSecThr
        //           << ", cur micro sec: " << microSec
        //           << ", pre micro sec: " << UTCMicroSecPre
        //           << ", delta t is: " << deltaTime << std::endl;
    }
    relTime += deltaTime;

    return relTime;
}

double NRxMultiPlatform::selfCalcScanTMilliSec(double milliSec, double azi, int cntPulse,
                                               double minScanMilliT, double maxScanMilliT)
{
    static int pulseCnt(0);
    static double scanT(20000); // default 20s
    static double preMilliT(milliSec);
    static double preAzi(azi);
    pulseCnt++;
    if (pulseCnt == cntPulse) {
        double diffTime = milliSec - preMilliT;
        double diffAzi = azi - preAzi;
        if (diffAzi < 0) {
            diffAzi = 360 - preAzi + azi;
        }
        scanT = 360.f / (diffAzi / diffTime);
        //reinitial
        pulseCnt = 0;
        preAzi = azi;
        preMilliT = milliSec;
    }
    scanT = std::min(scanT, maxScanMilliT);
    scanT = std::max(scanT, minScanMilliT);

    return scanT;
}
