#ifndef NRXCROSSPLATFORM_H
#define NRXCROSSPLATFORM_H

#ifdef _WIN32
#include <direct.h> // find exe path(WIN32)
#include <io.h> // find exe path(WIN32)
#else
#include <unistd.h>
#endif

#include <string>

using std::string;

namespace NRxMultiPlatform {

///
/// \brief getExePath, get path where program executing
/// \return
///
extern string getExePath(void);

///
/// \brief relPath2AbsPath, change relative path to absolute path
/// \param relPath
/// \return
///
extern string relPath2AbsPath(const string& relPath);

///
/// \brief regexMatch, if string match regex
/// \param reg, regex to match
/// \param src, source string
/// \return
///
extern bool regexMatch(const string& reg, const string& src);

///
/// \brief isValidIP, check if IP is valid IP
/// \param ip
/// \return
///
extern bool isValidIP(const string& ip);

///
/// \brief toBCD, change unsigned short to BCD num
/// \param batchNo
/// \return
///
extern unsigned short toBCD(unsigned short batchNo);

///
/// \brief fromBCD, change BCD num to unsigned short
/// \param BCDNo
/// \return
///
extern unsigned short fromBCD(unsigned short BCDNo);

///
/// \brief isBatchValid, check if batch no valid
/// \param batch
/// \return
///
extern bool isBatchValid(unsigned short batch);

///
/// \brief isBCDValid, check if BCD num is valid
/// \param bcd
/// \return
///
extern bool isBCDValid(unsigned short bcd);

///
/// \brief truncateDouble, truncate double to precision
/// \param val
/// \param precision
///
extern void truncateDouble(double& val, int precision);

///
/// \brief getCurLocalTimeOneDayMs
/// \return current local time in range 0-86400000 ms, -1 if fail
///
extern double getCurLocalTimeOneDayMilliSec();

///
/// \brief getCurLocalTimeUTCMs
/// \return current local time in range 0-86400000 ms, -1 if fail
///
extern double getCurLocalTimeUTCMilliSec();

///
/// \brief getCurLocalTimeOneDayUs
/// \return current local time in range 0-86400000000 us, -1 if fail
///
extern double getCurLocalTimeOneDayMicroSec();

///
/// \brief getCurLocalTimeUTCUs
/// \return current local time in range 0-86400000000 us, -1 if fail
///
extern double getCurLocalTimeUTCMicroSec();

///
/// \brief getSelfCntMilliSec, self count milli second, with platform local
///     micro second when call this function ! CAN ONLY CALL ONCE DURING ONE PROGRAM
/// \param alarmMilliSecThr, milli second threshold to print alarm
/// \return
///
extern double getSelfCntMilliSec(double alarmMilliSecThr);

///
/// \brief selfCalcScanTSec
/// \param milliSec, current time ms, UTC
/// \param azi, current azimuth deg
/// \param cntPulse, pulse num to calc one time, pulseNoPerCyc / 32
/// \param minScanMilliT, minimum ms / cyc
/// \param maxScanMilliT, maximum ms / cyc
/// \return scanT in ms
///
extern double selfCalcScanTMilliSec(double milliSec, double azi, int cntPulse,
                                    double minScanMilliT, double maxScanMilliT);
}

#endif // NRXCROSSPLATFORM_H
