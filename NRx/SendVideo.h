//
// Created by csic724 on 2024/12/5.
//

#ifndef READER_SENDVIDEO_H
#define READER_SENDVIDEO_H
#include "NRxUdpSender.h"
#include "MyStruct.h"
#include "fstream"
#include <string>
#include <iostream>
#include <vector_types.h>
#include <cmath>
#include <complex>

#define WAVE_NUM 32

class SendVideo {
public:
    SendVideo();
    ~SendVideo();
    void send(char *rawMessage, float2 *data);

private:
    unsigned int unMinPRTLen;
    unsigned int unTmpAzi;
    char* m_sendBufOri;
    double c_speed;

    static double asind(double x);
};


#endif //READER_SENDVIDEO_H
