#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/time.h>
#endif

/**
 * @brief The StaticTimer class, static function to calculate time consume,
 *      simple to use but cann't calculate multi work time at same time.
 */
class StaticTimer
{
public:
    StaticTimer();
    static void tick(void);
    static void tock(void);
    static double timeDuration(void);

private:
#ifdef WIN32
    static LARGE_INTEGER _startT;
    static LARGE_INTEGER _stopT;
    static double _dqFreq;
#else
    static std::chrono::time_point<std::chrono::high_resolution_clock> _startT;
    static std::chrono::time_point<std::chrono::high_resolution_clock> _stopT;
#endif
};

/**
 * @brief The Timer class, static Timer to calculate multiple time in multithread
 *      usage; tickOnce and timeOut to tick(), tock(), tock()... untill timeThr.
 */
class Timer
{
public:
    Timer();
    void tick(void);
    void tock(void);
    double timeDuration(void);
    void tickOnce(void);
    bool timeOut(double secThr);

private:
    bool _toTick;
#ifdef WIN32
    LARGE_INTEGER _startT;
    LARGE_INTEGER _stopT;
    double _dqFreq;
#else
    std::chrono::time_point<std::chrono::high_resolution_clock> _startT;
    std::chrono::time_point<std::chrono::high_resolution_clock> _stopT;
#endif
};

extern void MicrosecSleep(long timeUS);

#endif // CCALCCCOMSUMETIME_H
