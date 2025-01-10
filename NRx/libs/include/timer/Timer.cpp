#include "Timer.h"

#ifdef _WIN32
LARGE_INTEGER StaticTimer::_startT;
LARGE_INTEGER StaticTimer::_stopT;
double StaticTimer::_dqFreq = 0;
#else
std::chrono::time_point<std::chrono::high_resolution_clock> StaticTimer::_startT;
std::chrono::time_point<std::chrono::high_resolution_clock> StaticTimer::_stopT;
#endif
StaticTimer::StaticTimer()
{
}

void StaticTimer::tick()
{
#ifdef _WIN32
    if(_dqFreq < 1e-3)
    {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        _dqFreq = (double)freq.QuadPart;
    }
    QueryPerformanceCounter(&_startT);
#else
    _startT = std::chrono::high_resolution_clock::now();
#endif
}

// return seconds from call StartTick()
void StaticTimer::tock()
{
#ifdef _WIN32
    QueryPerformanceCounter(&_stopT);
#else
    _stopT = std::chrono::high_resolution_clock::now();
#endif
}

/**
 * @brief Timer::timeDuration
 * @return sec between tick() and tock()
 */
double StaticTimer::timeDuration()
{
#ifdef _WIN32
    // return s
    return ((double)(_stopT.QuadPart - _startT.QuadPart) / _dqFreq);
#else
    std::chrono::duration<double> diff = _stopT - _startT;
    return diff.count();
#endif
}

Timer::Timer()
    : _toTick(true)
{
#ifdef WIN32
    _dqFreq = 0;
#else
    ;
#endif
}

void Timer::tick()
{
#ifdef _WIN32
    if(_dqFreq < 1e-3)
    {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        _dqFreq = (double)freq.QuadPart;
    }
    QueryPerformanceCounter(&_startT);
#else
    _startT = std::chrono::high_resolution_clock::now();
#endif
}

void Timer::tock()
{
#ifdef _WIN32
    QueryPerformanceCounter(&_stopT);
#else
    _stopT = std::chrono::high_resolution_clock::now();
#endif
}

double Timer::timeDuration()
{
#ifdef _WIN32
    // return s
    return ((double)(_stopT.QuadPart - _startT.QuadPart) / _dqFreq);
#else
    std::chrono::duration<double> diff = _stopT - _startT;
    return diff.count();
#endif
}

void Timer::tickOnce()
{
    if (_toTick) {
        tick();
        _toTick = false;
    }
}

bool Timer::timeOut(double secThr)
{
    bool ret(false);

    tock();
    if (timeDuration() > secThr) {
        _toTick = true;
        ret = true;
    }

    return ret;
}

void MicrosecSleep(long timeUS)
{
#ifdef WIN32
    LARGE_INTEGER itmp;
    LONGLONG part1, part2;
    double minus, freq, tim, spec;
    QueryPerformanceFrequency(&itmp);
    freq = (double)itmp.QuadPart;
    QueryPerformanceCounter(&itmp);
    part1 = itmp.QuadPart;
    spec = 0.000001 * timeUS;

    do {
        QueryPerformanceCounter(&itmp);
        part2 = itmp.QuadPart;
        minus = (double)(part2 - part1);
        tim = minus / freq;
    } while(tim < spec);
#else
    usleep(timeUS); // not test precision
#endif
}
