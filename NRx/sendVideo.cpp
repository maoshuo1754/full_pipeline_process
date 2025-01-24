//
// Created by csic724 on 2024/12/5.
//

#include "SendVideo.h"
#include "../Config.h"
#include "../utils.h"

SendVideo::SendVideo(): plot(){ // , outfile("detectVideo.txt")
    m_sendBufOri = new char[1024 * 1024];
    unMinPRTLen = RANGE_NUM;
    unTmpAzi = 0;

    // 32个脉组的时间偏移量
    for (int ii = 0; ii < 8; ii++) {
        for (int jj = 0; jj < 4; jj++) {
            timeArray[ii * 4 + jj] = ii * 0.5 + jj;
        }
    }

    sendSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sendSocket < 0) {
        cerr << "Send Socket init failed!" << endl;
    }

    memset(&remoteAddr, 0, sizeof(remoteAddr));
    remoteAddr.sin_family = AF_INET;
    remoteAddr.sin_port = htons(remote_video_port);  // 0x2001是8192  0x2002岁8194
    remoteAddr.sin_addr.s_addr = inet_addr(remote_video_ip.c_str());

    memset(&localAddr, 0, sizeof(localAddr));
    localAddr.sin_family = AF_INET;
    localAddr.sin_port = htons(local_video_port);  // 0x2001是8192  0x2002岁8194
    localAddr.sin_addr.s_addr = inet_addr(local_video_ip.c_str());

    std::cout << "send_ip:      " << local_video_ip << ":" << local_video_port <<std::endl;
    std::cout << "multicast_ip: " << remote_video_ip << ":" << remote_video_port << std::endl;

    if (bind(sendSocket, (sockaddr*)&localAddr, sizeof(localAddr)) < 0)
    {
        close(sendSocket);
        std::cerr << "bind error\n";
    } else {
        std::cout << "Socket bind success" << std::endl;
    }

    plot.setSocket(sendSocket, remoteAddr);

}

SendVideo::~SendVideo() {
    close(sendSocket);
    delete[] m_sendBufOri;
}

double SendVideo::asind(double x) {
    std::complex<double> z(x, 0.0);
    std::complex<double> result = std::asin(z) * 180.0 / M_PI;
    return result.real();
}


// 输入参数：
// - rawMessage: 报文头的原始信息
// - detectedVideo: 检测后视频
// - chnSpeeds: 通道对应的速度值，0.01m/s
// - speedChannels: 检查视频每个点从哪个通道选出来的最大值，和detectedVideo大小一样

void SendVideo::send(unsigned char *rawMessage, float2 *detectedVideo, vector<int> &chnSpeeds, int *speedChannels,
                int numSamples, int rangeNum) {
//    msgInfo->AziVal = static_cast<unsigned short>(asind(j * lambda_0 / (WAVE_NUM * d)) * 65536.0 / 360.0);
    uint32 dwTemp;
    int nAzmCode;
    float rAzm;

    auto rawMsg = reinterpret_cast<uint32*>(rawMessage);
    uint32 freqPoint = ((rawMsg[11]) & 0x00000fff);
    double lambda_0 = c_speed / ((freqPoint * 10 + initCarryFreq) * 1e6);
    float data_amp;

    videoMsg.CommonHeader.wCOUNTER = rawMsg[4];  // 触发计数器
    dwTemp = rawMsg[6] / 10 + 8*60*60*1000; // FPGA时间 //0.1ms->1ms + 8h

    videoMsg.CommonHeader.dwTxSecondTime = dwTemp / 1000;
    videoMsg.CommonHeader.dwTxMicroSecondTime = dwTemp % 1000 * 1000;

    videoMsg.RadarVideoHeader.dwTxAbsSecondTime = dwTemp / 1000;
    videoMsg.RadarVideoHeader.dwTxAbsMicroSecondTime = dwTemp % 1000 * 1000;

    videoMsg.RadarVideoHeader.dwTxRelMilliSecondTime_H = dwTemp / 1000;
    videoMsg.RadarVideoHeader.dwTxRelMilliSecondTime_L = dwTemp % 1000 * 1000;

//    for (int ii = 0; ii < WAVE_NUM; ii++) {
    for (int ii = WAVE_NUM - 1; ii >=0; ii--) {

        int sec = dwTemp / 1000 % 60 + timeArray[ii];
//        cout << "time:" << h << ":" << min << ":" << sec << endl;

        nAzmCode = (azi_table[31 - ii] & 0xffff);

        if (nAzmCode > 32768)
            nAzmCode -= 65536;

        rAzm = 60 + asin((nAzmCode * lambda_0) / (65536 * d)) / 3.1415926 * 180.0f;

        if (rAzm < 0)
            rAzm += 360.f;

//        cout << "[rAzm:]" << ii << " " << rAzm << endl;
        dwTemp = UINT16(rAzm / 360.0f * 65536.0f);
        videoMsg.RadarVideoHeader.wAziCode = htons(dwTemp);

        auto* rowData = detectedVideo + ii * NFFT;
        auto* rowSpeed = speedChannels + ii * NFFT;
        for (int k = 0; k < unMinPRTLen; ++k) {
            // + system_delay
            //TODO: 这个偏移会不会动
            data_amp = rowData[k + numSamples - 1 + 52].x;
            data_amp = data_amp * 1.0;
            if(data_amp > 255)
                data_amp = 255;

            videoMsg.bytVideoData[k] = (unsigned char)data_amp;
            rowSpeed[k] = chnSpeeds[rowSpeed[k + numSamples - 1 + 52]];
        }

//        cout << "ii:" << ii << " [rAzm]:" << rAzm << endl;
        dwTemp = UINT16(rAzm / 360.0 * 65536.0f);
        videoMsg.RadarVideoHeader.wAziCode = htons(dwTemp);

        auto sendres = sendto(sendSocket, &videoMsg, sizeof(videoMsg), 0, (sockaddr *)&remoteAddr, sizeof(remoteAddr));
        if (sendres < 0) {
            std::cerr << "Detected video sendto() failed!" << std::endl;
            break;
        }
        plot.MainFun(reinterpret_cast<char *>(&videoMsg), sizeof(videoMsg), rowSpeed);
    }


}
