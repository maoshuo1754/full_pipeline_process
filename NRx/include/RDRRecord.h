#ifndef RDRRECORD
#define RDRRECORD

#include <string>
#include <vector>

#ifdef BUILDRECORDDLL
#ifdef WIN32
# define RECORDDLLAPI __declspec(dllexport)
#else
#define RECORDDLLAPI __attribute__((visibility("default")))
#endif
#else
#ifdef WIN32
# define RECORDDLLAPI __declspec(dllimport)
#else
#define RECORDDLLAPI __attribute__((visibility("default")))
#endif
#endif

namespace rdrRecord {

using namespace std;

class RDRRecorderImpl;

///
/// \brief The RDRReplayer class, call udp localIP and dstIP before start()
///
class RECORDDLLAPI RDRRecorder
{
public:
    enum RDRDATATYPE {
        UNKNOWN,
        ORIVID,
        DETVID,
        PLOT,
        TRK,
        CTRL
    };
    enum RDRMSGFORMAT {
        NonNRx,
        NRx
    };

    RDRRecorder();
    ~RDRRecorder(void);
    void setSessionPath(const string& sessionPath); // set sessions path "xxx/xxx/.../Sessions/sessionFolder"
    int writeData(const char* data, int dataLen, RDRMSGFORMAT dataType, unsigned char chnID = 0);
    void start(const string& sessionPath); // start record thread
    void exit(void); // exit record thread
    bool isRunning(void); // if called start() return true; else(include called exit()) return false
    double getWriteSpeed (void); // get write speed MB/s
    double getCurLogFileSize(void); // get size of current log file MB
    string getCurLogFileName(void);
    long getCurLogFileStartTime(void); // UTC
    void setLogFileSizeLimit(long sizeLimit); // 1Byte
    void setLogFileTimeLimit(long timeLimit); // 1s
    void setRDRNRxVidDataCprFlag(bool isCpr);
    void setLogDataEnable(RDRDATATYPE dataType);
    void setLogDataDisable(RDRDATATYPE dataType);
    void setNRxVidLatitude(bool enable, double lat);
    void setNRxVidLongitude(bool enable, double lon);
    void setNRxVidHeight(bool enable, double height);
    void setNRxVidRadarID(bool enable, unsigned short radarID);
    void setNonNRxDataCprFlag(bool isCpr);

private:
    RDRRecorderImpl* m_recorder;
};

}

typedef rdrRecord::RDRRecorder::RDRDATATYPE RECDATATYPE;
typedef rdrRecord::RDRRecorder::RDRMSGFORMAT RECFORMAT;

#endif // RDRRECORD
