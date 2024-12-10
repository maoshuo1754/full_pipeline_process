//
// Created by csic724 on 2024/12/5.
//

#ifndef READER_SENDVIDEO_H
#define READER_SENDVIDEO_H
#include "NRxUdpSender.h"
#include "MyStruct.h"
#include "fstream"
#include "../utils.h"
#include "../queue.h"
#include <string>
#include <iostream>
#include <vector_types.h>
#include <cmath>
#include <complex>
#include <sys/socket.h>
#include "netinet/in.h"
#include "arpa/inet.h"
#include "cstring"
#include "unistd.h"

#define WAVE_NUM 32

#define TAO (5.0e-6)
#define TAO_US (TAO*1e6f)

class SendVideo {
public:
    SendVideo();
    ~SendVideo();
    void send(char *rawMessage, float2 *data, int numSamples, int rangeNum);

private:
    unsigned int unMinPRTLen;
    unsigned int unTmpAzi;
    char* m_sendBufOri;
    double c_speed;
    int sendSocket;
    VideoToNRXGUI videoMsg;
    int timeArray[32];
    unsigned short azi_table[32];
    double d; // 间距
    sockaddr_in addr;
    sockaddr_in myaddr;

    static double asind(double x);
};


#endif //READER_SENDVIDEO_H
