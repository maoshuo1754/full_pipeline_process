//
// Created by csic724 on 2024/12/5.
//

#include "SendVideo.h"
#include "../Config.h"

SendVideo::SendVideo() {
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
        std::cerr << "Send Socket init failed!";
    }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(multicast_port);  // 0x2001是8192  0x2002岁8194
    std::cout << "send_ip:      " << send_ip << std::endl;
    std::cout << "multicast_ip: " << multicast_ip << std::endl;
    addr.sin_addr.s_addr = inet_addr(multicast_ip.c_str());

    memset(&myaddr, 0, sizeof(myaddr));
    myaddr.sin_family = AF_INET;
    myaddr.sin_port = htons(send_port);  // 0x2001是8192  0x2002岁8194
    myaddr.sin_addr.s_addr = inet_addr(send_ip.c_str());

    if (bind(sendSocket, (sockaddr*)&myaddr, sizeof(myaddr)) < 0)
    {
        close(sendSocket);
        std::cerr << "bind error\n";
    } else {
        std::cout << "Socket bind success" << std::endl;
    }

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

void SendVideo::send(char *rawMessage, float2 *data, int numSamples, int rangeNum) {
//    msgInfo->AziVal = static_cast<unsigned short>(asind(j * lambda_0 / (WAVE_NUM * d)) * 65536.0 / 360.0);
    unsigned long dwTemp, dwTemp1;
    int nAzmCode;
    float rAzm;

    auto rawMsg = reinterpret_cast<int*>(rawMessage);
    int freqPoint = (rawMsg[12] & 0x00000fff);
    double lambda_0 = c_speed / ((freqPoint * 10 + 9600)*1e6);
    double data_amp;

    for (int ii = 0; ii < WAVE_NUM; ii++) {
        videoMsg.CommonHeader.wCOUNTER = htons(rawMsg[4] & 0xffff);  // 触发计数器
        dwTemp = (rawMsg[6] & 0x1fffffff); // FPGA时间
        dwTemp1 = dwTemp / 10000;

        //TRACE("FPGA TIME:%x\n", dwTemp);
        //TRACE("dwTxSecondTime:%x\n", dwTemp1);
        //TRACE("dwTxMicroSecondTime:%x\n", (dwTemp - dwTemp1 * 10000) * 100);

        dwTemp = (rawMsg[7] & 0x3fffffff) / 10;//0.1ms->1ms
        //TRACE("FPGA TIME ms:%d\n", dwTemp);
        int h = dwTemp / 1000 / 60 / 60;
        int min = (dwTemp - (h * 60 * 60 * 1000)) / 1000 / 60;
        int sec = dwTemp / 1000 % 60 + timeArray[ii];
        //TRACE("FPGA TIME h:%d;min:%d;sec:%d\n", h, min, sec);


        videoMsg.CommonHeader.dwTxSecondTime = htonl(dwTemp);        //FPGA Time
        //videoMsg.CommonHeader.dwTxMicroSecondTime = htonl((dwTemp - dwTemp1*10000)*100);		//FPGA Time
        videoMsg.CommonHeader.dwTxMicroSecondTime = htonl(dwTemp);        //FPGA Time



        videoMsg.RadarVideoHeader.dwTxAbsSecondTime = htonl(dwTemp);
        videoMsg.RadarVideoHeader.dwTxAbsMicroSecondTime = htonl(0);

        videoMsg.RadarVideoHeader.dwTxRelMilliSecondTime_H = htonl(0);
        videoMsg.RadarVideoHeader.dwTxRelMilliSecondTime_L = htonl(dwTemp);

        //nAzmCode = (rawMessage[8] & 0xffff);
        nAzmCode = (azi_table[31 - ii] & 0xffff);

        if (nAzmCode > 32768)
            nAzmCode -= 65536;

        //rAzm = asin((nAzmCode * BOCHANG) / (65536 * JIANJU))/3.1415926*180.0f;  //20200829
        //rAzm = 60 + asin((nAzmCode * bochang0) / (65536 * JIANJU)) / 3.1415926*180.0f;//״ﰲװƫ
        rAzm = 60 + asin((nAzmCode * lambda_0) / (65536 * d)) / 3.1415926 * 180.0f;//״ﰲװƫ
        //rAzm = 0 + asin((nAzmCode * bochang0) / (65536 * JIANJU)) / 3.1415926*180.0f;//״ﰲװƫ
        //rAzm = 122 + asin((nAzmCode * BOCHANG) / (65536 * JIANJU)) / 3.1415926*180.0f;//״ﰲװƫ
        //---------------------------------
        if (rAzm < 0)
            rAzm += 360.f;

        dwTemp = UINT16(rAzm / 360.0 * 65536.0f);
        videoMsg.RadarVideoHeader.wAziCode = htons(dwTemp);

        auto* rowData = data + ii * NFFT;
        for (int k = 0; k < unMinPRTLen; ++k) {
            // + system_delay
            data_amp = (double)(rowData[k + numSamples - 1].x);
//            data_amp = data_amp * 255 / 93.4;
            if(data_amp > 255)
                data_amp = 255;
            videoMsg.bytVideoData[k] = (unsigned char)data_amp;
        }

        dwTemp = UINT16(rAzm / 360.0 * 65536.0f);
        videoMsg.RadarVideoHeader.wAziCode = htons(dwTemp);
        auto sendres = sendto(sendSocket, &videoMsg, sizeof(videoMsg), 0, (sockaddr *)&addr, sizeof(addr));
        if (sendres < 0) {
            std::cerr << "sendto() failed!" << std::endl;
            break;
        }
    }


}
