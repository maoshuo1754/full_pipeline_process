//
// Created by csic724 on 2024/12/5.
//

#ifndef READER_SENDVIDEO_H
#define READER_SENDVIDEO_H
#include "MyStruct.h"
#include "fstream"
#include "utils.h"
#include <netinet/in.h>
#include "plot.h"

#define TAO (5.0e-6)
#define TAO_US (TAO*1e6f)

class SendVideo {
public:
    SendVideo();
    ~SendVideo();
    void send(struct RadarParams* radar_params_);

private:
    unsigned int unMinPRTLen;
    unsigned int unTmpAzi;
    char* m_sendBufOri;
    int sendSocket;
    VideoToNRXGUI videoMsg;
    int timeArray[WAVE_NUM];
    double aziArray[WAVE_NUM];
    sockaddr_in remoteAddr;
    sockaddr_in localAddr;
    Plot plot;
    static double asind(double x);
    ofstream outfile;
    std::mutex sendMutex;  // 用于保护 send 函数的互斥锁
};


#endif //READER_SENDVIDEO_H
