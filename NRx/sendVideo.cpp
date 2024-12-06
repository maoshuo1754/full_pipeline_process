//
// Created by csic724 on 2024/12/5.
//

#include "SendVideo.h"

SendVideo::SendVideo(){
    m_sendBufOri = new char [1024 * 1024];
    unMinPRTLen = 7498;
    unTmpAzi = 0;
    c_speed = 2.99792458e8;
//    std::string filename("/home/csic724/CLionProjects/reader/NRx/azi_table.txt");
//    loadFromFile(filename);

    udpSender::addUdpSender("MTDVideo", "127.0.0.1", 7458);
}

SendVideo::~SendVideo() {
    delete[] m_sendBufOri;
}

//void SendVideo::loadFromFile(const std::string& filename) {
//    std::ifstream azi_file(filename);
//    if (!azi_file) {
//        std::cerr << "Can't open file: " << filename <<std::endl;
//        return;
//    }
//
//    for(auto & i : azi_table) {
//        if (!(azi_file >> i)) {
//            std::cerr << "Error when read file" << std::endl;
//            break;
//        }
//    }
//}

double SendVideo::asind(double x) {
    std::complex<double> z(x, 0.0);
    std::complex<double> result = std::asin(z) * 180.0 / M_PI;
    return result.real();
}

void SendVideo::send(char *rawMessage, float2 *data) {
    auto rawMsg = reinterpret_cast<int*>(rawMessage);
    int GpsTime = (rawMsg[7] & 0x3fffffff) / 10;  // 0.1ms -> 1ms
    int hour = GpsTime / 1000 / 60 / 60;
    int min = (GpsTime - (hour * 1000 * 3600)) / 1000 / 60;
    int sec = GpsTime / 1000 % 60;

    int freqPoint = (rawMsg[12] & 0x00000fff);
    freqPoint = 3;
    double lambda_0 = c_speed / ((freqPoint * 10 + 9600)*1e6);
    double d = 0.0135;

    double azi_table[WAVE_NUM];
    for(int i = 0; i < WAVE_NUM; i++) {
        int j = i - WAVE_NUM / 2;
        std::cout << j << " ";
        azi_table[i] = asind(j * lambda_0 / (WAVE_NUM * d));
    }


    auto* msgHead = reinterpret_cast<_tagMsgHead*>(m_sendBufOri);
    msgHead->ucDragCAT = 0xD2;
    msgHead->usDragLength = sizeof(_tagMsgHead) + sizeof(_EntireMessageInfo) + unMinPRTLen * sizeof(unsigned char) + sizeof(_tagMsgEnd);

    auto* msgInfo = reinterpret_cast<_EntireMessageInfo*>(msgHead + 1);
    msgInfo->PackKind = 4;
    msgInfo->DisCellNum = unMinPRTLen;


}
