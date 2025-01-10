#include "NRxLogger.h"

#include <iostream>

void NRxLogger::logError(const std::string& msg)
{
    std::cout << msg << std::endl;
}

void NRxLogger::logWarning(const std::string& msg)
{
    std::cout << msg << std::endl;
}

void NRxLogger::logDebug(const std::string& msg)
{
#ifdef DEBUGPRINT
    std::cout << msg;
#else
    msg.data();
#endif
}

