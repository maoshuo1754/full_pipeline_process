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
    int timeArray[32];
    sockaddr_in remoteAddr;
    sockaddr_in localAddr;
    Plot plot;
    static double asind(double x);
    ofstream outfile;
};


#endif //READER_SENDVIDEO_H
