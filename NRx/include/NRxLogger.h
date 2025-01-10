/*
* NRxUtile.h
*
* Description:
* 		Defines utils in Windows and Linux platform.
*
* HINTS:
* 		20210512	0.0.1.	DDC.	Create the first version.
*/
#ifndef NRXLOGGER_H
#define NRXLOGGER_H

#include <string>

#ifdef BUILDLOGDLL
#ifdef WIN32
#define LOGDLLAPI __declspec(dllexport)
#else
#define LOGDLLAPI __attribute__((visibility("default")))
#endif
#else
#ifdef WIN32
#define LOGDLLAPI __declspec(dllimport)
#else
#define LOGDLLAPI __attribute__((visibility("default")))
#endif
#endif

namespace NRxLogger {

using std::string;

///
/// \brief setLogName, set name of logger file
/// \param fileName, name of logger file
///
extern LOGDLLAPI void setLogName(const string& fileName);

///
/// \brief logError, log error info, which criticle error will affect normal
///     data process.
/// \param msg, log msg, will output with new line character.
///
extern LOGDLLAPI void logError(const string& msg);

///
/// \brief logWarning, log warning info, which won't affect normal data process,
///     but needed for developer
/// \param msg, log msg, will output with new line character.
///
extern LOGDLLAPI void logWarning(const string& msg);

///
/// \brief logDebug, include several info you need to use when developing
/// \param msg, log msg, will output with new line character.
///
extern LOGDLLAPI void logDebug(const string& msg);

}

#endif// NRXLOGGER_H
