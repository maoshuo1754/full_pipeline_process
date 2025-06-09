//
// Created by csic724 on 2024/12/5.
//

#include "SendVideo.h"
#include "ThreadPool.h"
#include <complex>
#include "Config.h"
#include "fdacoefs.h"
#include "utils.h"

SendVideo::SendVideo() { // ,:outfile("azi.txt")

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
    localAddr.sin_port = htons(local_video_port);    // 0x2001是8192  0x2002岁8194
    localAddr.sin_addr.s_addr = inet_addr(local_video_ip.c_str());

    std::cout << "local Address:      " << local_video_ip << ":" << local_video_port <<std::endl;
    std::cout << "remote video Address: " << remote_video_ip << ":" << remote_video_port << std::endl;

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

void SendVideo::send(RadarParams* radar_params_) {
    std::lock_guard<std::mutex> lock(sendMutex);  // 锁定互斥锁，函数退出时自动解锁
    uint32 dwTemp;
    int nAzmCode;
    float rAzm;
    auto rawMsg = reinterpret_cast<uint32*>(radar_params_->rawMessage);

    videoMsg.CommonHeader.wCOUNTER = rawMsg[4];  // 触发计数器
    dwTemp = rawMsg[6] / 10 ; // FPGA时间 //0.1ms->1ms + 8h
    // cout << "dwTemp = " << dwTemp << endl;
    videoMsg.CommonHeader.dwTxSecondTime = dwTemp / 1000;
    videoMsg.CommonHeader.dwTxMicroSecondTime = dwTemp % 1000 * 1000;

    videoMsg.RadarVideoHeader.dwTxAbsSecondTime = dwTemp / 1000;
    videoMsg.RadarVideoHeader.dwTxAbsMicroSecondTime = dwTemp % 1000 * 1000;

    videoMsg.RadarVideoHeader.dwTxRelMilliSecondTime_H = dwTemp / 1000;
    videoMsg.RadarVideoHeader.dwTxRelMilliSecondTime_L = dwTemp % 1000 * 1000;

    auto fLFMStartWord = rawMsg[16];
    videoMsg.RadarVideoHeader.dwSigBWHz = htonl((Fs_system - fLFMStartWord / pow(2.0f, 32) * Fs_system) * 2.0);
    videoMsg.RadarVideoHeader.dwSampleFreqHz = htonl(Fs);

    // for (int ii = WAVE_NUM - 1; ii >= 0; ii--) {
    for (int wave_idx = end_wave - 1; wave_idx >= start_wave; wave_idx--) {
        int sec = dwTemp / 1000 % 60 + timeArray[wave_idx];

        rAzm = getAzi(wave_idx, radar_params_->lambda);
        // rAzm = (getAzi(wave_idx, radar_params_->lambda) + 120);
        // if (rAzm > 360) rAzm -= 360;

        // cout << ii << " " << rAzm << endl;
        dwTemp = UINT16(rAzm / 360.0f * 65536.0f);
        videoMsg.RadarVideoHeader.wAziCode = htons(dwTemp);

        auto* rowData = radar_params_->h_max_results_ + wave_idx * NFFT;
        auto* rowSpeed = radar_params_->h_speed_channels_ + wave_idx * NFFT;
        auto* rowAzi = radar_params_->h_azi_densify_results_ + wave_idx * NFFT;

        // int offset = range_correct + radar_params_->numSamples - 1 + floor((BL-1)/2);
        int offset = range_correct + radar_params_->numSamples - 1;

        for (int k = 0; k < unMinPRTLen - offset - 1; ++k) {
            // + system_delay
            auto data_amp = rowData[k + offset];
            data_amp = data_amp * 1.0;
            if(data_amp > 255)
                data_amp = 255;
            videoMsg.bytVideoData[k] = (unsigned char)data_amp;
            rowSpeed[k] = rowSpeed[k + offset];
            rowAzi[k] = rowAzi[k + offset];
        }

        auto sendres = sendto(sendSocket, &videoMsg, sizeof(videoMsg), 0, (sockaddr *)&remoteAddr, sizeof(remoteAddr));
        if (sendres < 0) {
            std::cerr << "Detected video sendto() failed!" << std::endl;
            break;
        }
        plot.MainFun(reinterpret_cast<char *>(&videoMsg), sizeof(videoMsg), rowSpeed, rowAzi);
    }


}
