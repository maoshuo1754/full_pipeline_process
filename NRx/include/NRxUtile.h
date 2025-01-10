/*
* NRxUtile.h
*
* Description:
* 		Defines utils in Windows and Linux platform.
*
* HINTS:
* 		20210512	0.0.1.	DDC.	Create the first version.
*/
#ifndef NRXUTILE_H
#define NRXUTILE_H

#include <list>
#include <vector>
#include <deque>
//#include <map>
#include <unordered_map> // use unordered_map instead of map
#include <set>
#include <string>
#include <algorithm>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <exception>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <fstream>
#include <ctime>
#include <random>
#include <regex>

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

using namespace std;

#ifdef WIN32
#include <Windows.h>
#include <direct.h> // find exe path(WIN32)
#include <io.h> // find exe path(WIN32)
#else
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <dirent.h>
#endif// WIN32

#ifdef QT_CORE_LIB
#include <QDebug>
#include <QTime>
#endif// QT_CORE_LIB

#endif// NRXUTILE_H
