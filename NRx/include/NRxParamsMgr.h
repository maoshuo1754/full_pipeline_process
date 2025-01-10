#ifndef PARAMSSRC_H
#define PARAMSSRC_H

#include <string>
#include <vector>

#ifdef BUILDPARAMDLL
#ifdef WIN32
# define PARAMDLLAPI __declspec(dllexport)
#else
#define PARAMDLLAPI __attribute__((visibility("default")))
#endif
#else
#ifdef WIN32
# define PARAMDLLAPI __declspec(dllimport)
#else
#define PARAMDLLAPI __attribute__((visibility("default")))
#endif
#endif

namespace libParam {
using std::string;

static const string globalAreaName{"Global"};

// parameter type, int, double, string
enum NRXVARTYPE{
    TypeInt,
    TypeDouble,
    TypeStr,
    TypeWrong
};

// visibility level on display, to user or to developer
enum VISLV{
    VisUsr,
    VisDev,
    InVisible // not show in display
};

enum VARSCOPE{
    GlobalVar,
    AreaVar,
    AllVar
};

// basic info of a param
class PARAMDLLAPI AttrBase {
public:
    string m_name;
    VARSCOPE m_varScope;
    NRXVARTYPE m_varType;
    VISLV m_visLv;
    bool m_ismutable;
    string m_ctrlUnitName;
    int m_ctrlUnitVal;
    string m_tag;
    string m_note;
    int m_precision;

    AttrBase(const string& name = "", VARSCOPE varScope = GlobalVar, NRXVARTYPE varType = TypeInt,
             VISLV visLv = VisDev, bool ismutable = false, const string& ctrlUnitName = "",
             int ctrlUnitVal = 0, const string& tag = "", const string& note = "", int precision = 0);

    bool isGlobalVar(void) const {
        return (m_varScope == GlobalVar);
    }
};

// AttrInt
class PARAMDLLAPI AttrInt {
public:
    AttrBase m_attrBase;
    int m_defVal;
    int m_minVal;
    int m_maxVal;

    AttrInt(const int defVal = 0, const int minVal = 0, const int maxVal = 0,
            const string& name = "", VARSCOPE varScope = GlobalVar,
            VISLV vis = VisDev, bool isMutable = false, const string& ctrlUnitName = "",
            int ctrlUnitVal = 0, const string& tag = "", const string& note = "");
    AttrInt(const AttrBase base, const int defVal = 0, const int minVal = 0, const int maxVal = 0);
};

// AttrDouble
class PARAMDLLAPI AttrDouble {
public:
    AttrBase m_attrBase;
    double m_defVal;
    double m_minVal;
    double m_maxVal;

    AttrDouble(const double defVal = 0, const double minVal = 0, const double maxVal = 0,
               const string& name = "", VARSCOPE varScope = GlobalVar,
               VISLV vis = VisDev, bool isMutable = false, const string& ctrlUnitName = "",
               int ctrlUnitVal = 0, const string& tag = "", const string& note = "",
               int precision = 2);
    AttrDouble(const AttrBase base, const double defVal = 0, const double minVal = 0, const double maxVal = 0);
};

// AttrStr
class PARAMDLLAPI AttrStr {
public:
    AttrBase m_attrBase;
    string m_defVal;

    AttrStr(const string& defVal = "",
            const string& name = "", VARSCOPE varScope = GlobalVar,
            VISLV vis = VisDev, bool isMutable = false, const string& ctrlUnitName = "",
            int ctrlUnitVal = 0, const string& tag = "", const string& note = "");
    AttrStr(const AttrBase base, const string& defVal = "");
};

enum AREATYPE {
    AreaTypeFan,
    AreaTypePoly
};

class PARAMDLLAPI Point
{
public:
    double azi;
    double dis;
    double x;
    double y;

    Point(void);
    Point(const Point&);
    Point(double aziVal, double disVal);
    bool operator==(const Point& rhs)const;
};

typedef std::vector<Point> PointsVec;

extern PARAMDLLAPI int testGet1();

/// all funcs below is not multi-thread safe. Call iniConfigFile first to initial
/// std::vector<ParamManager*>, which hold all ParamManagers. Other funcs will not
/// check any more.

///
/// \brief iniConfigFileName, set default file path of param config file, and
///     load params from specific file. Call it first if you need specific
///     config file name.
/// \param filePath, file path to default config file
///
extern PARAMDLLAPI void iniConfigFile(const string& iniFilePath, size_t listIdx = 0);

//extern PARAMDLLAPI void setParamMgr(const string& iniFilePath);

///
/// \brief setAreaMapScale, set scale of area map, may consume much time,
///     recommand call at initial.
/// \param mapMaxDis, max dis in area map
/// \param mapDisSpan, dis span, dis cell num = mapMaxDis / mapDisSpan
/// \param mapAziNum, azi cell num, max dis is 360, azi span = 360 / mapAziNum
///
extern PARAMDLLAPI void setAreaMapScale(unsigned int mapMaxDis, size_t mapDisSpan,
                                                      size_t mapAziNum, size_t listIdx = 0);

///
/// \brief genAreaNameWithId, generate area name with assigned areaId.
/// \param areaId, area id number
/// \return
///     "usrArea-" + (areaId)
///
extern PARAMDLLAPI const string genAreaNameWithId(unsigned char areaId);

///
/// \brief resetParams
/// \param areaName
/// \param tagName
/// \param isGlobalParams
/// \param listIdx
///
extern PARAMDLLAPI void resetParams(VARSCOPE paramType, const string& areaName = "",
                                                  const string& tagName = "", size_t listIdx = 0);

///
/// \brief setConfigFilePath, set config file path, will change default
///     config file path, won't automatic update param in memory.
/// \param filePath, new file path
///
extern PARAMDLLAPI void setConfigFilePath(const string& filePath, size_t listIdx = 0);

///
/// \brief read, read params from current xml file
///
extern PARAMDLLAPI void read(size_t listIdx = 0);

///
/// \brief readFrom, read params from xml file with assigned filePath,
///     won't change default path.
/// \param filePath, file path to read from
///
extern PARAMDLLAPI void readFrom(const string& filePath, size_t listIdx = 0);

///
/// \brief save, save current parameters in memory to file
///
extern PARAMDLLAPI void save(size_t listIdx = 0);

///
/// \brief saveAs, save current parameters in memory to file at filePath,
///     won't change default path
/// \param filePath, file path to save to
///
extern PARAMDLLAPI void saveAs(const string& filePath, size_t listIdx = 0);

///
/// \brief getAreaNum, return current area num in areaMapMgr,
///     user area num + 1(global area)
///
extern PARAMDLLAPI int getAreaNum(size_t listIdx = 0);

///
/// \brief addArea, add assigned area.
///     Add area and recalculate area map if success add assigned area
///     NOTE:
///         1. after insert one area, it will create area params from
///            "Global"'s m_libAreaParams
/// \param pts, points build fan area or polygon area, std::vector<Point>
/// \param name, area name
/// \param type, area type, AreaTypeFan or AreaTypePoly
/// \return
///     true, if success add area into memory
///     false, if:
///         1. area in memory already have one area with same `name`;
///         2. `type` not recogonized;
///         3. `type` and `pts` not match, such as AreaTypeFan has 3 points
///             or AreaTypePoly has points less than 3.
///
extern PARAMDLLAPI bool addArea(const PointsVec& pts, const string& name, AREATYPE type, size_t listIdx = 0);

///
/// \brief delArea, delete assigned area with area name.
///     Delete area and recalculate area map if success delete assigned area.
///     NOTE:
///         1. CANNOT DELETE "Global" AREA! It will return false if you try
///            to delete "Global" area
/// \param name, name of area you want to delete
/// \return
///     true, if success delete assigned area in memory
///     false, if:
///         1. name is "Global"
///         2. name not found in areas which stored in memory
///
extern PARAMDLLAPI bool delArea(const string& name, size_t listIdx = 0);

///
/// \brief insertIntParam, insert assigned parameter.
///     If two parameter has same tagName, paramName, then they are treated
///     as same parameter, we'll omit second one, and return false.
///     NOTE:
///         1. all tagName, paramName are case sensitive.
///         2. after insert one param, all exist area will have this param
///         3. could insert three type param: int, double and str
///         4. tagName could not be omitted
/// \param param, param to insert
/// \return
///      true, if success add parameter into memory.
///      false, if:
///         1. Has repeat parameter
///         2. If param tagName or paramName is empty
///
extern PARAMDLLAPI bool insertIntParam(const AttrInt& param, size_t listIdx = 0);
extern PARAMDLLAPI bool insertDoubleParam(const AttrDouble& param, size_t listIdx = 0);
extern PARAMDLLAPI bool insertStrParam(const AttrStr& param, size_t listIdx = 0);

///
/// \brief getIntParam, get assigned area params.
///     Get assigned param with <tag, paramName> in area according to pos.
///     NOTE:
///         1. could get three type param: int, double and str
///         2. param type not consistant with called function
///         3. if has exception, will return parameter value in "Global" area
///         4. for more efficient usage, use param like [string a("name")] other than
///             temporary variables like [getIntParam("name")]
/// \param tag, tag name to find parameter
/// \param paramName, name of parameter to find
/// \param pos, pos to find assigned area in area map
/// \return
///     value of assigned parameter
///
extern PARAMDLLAPI int getIntAreaParam(const string& tag, const string& paramName,
                                                     const Point& pos, size_t listIdx = 0);
extern PARAMDLLAPI double getDoubleAreaParam(const string& tag, const string& paramName,
                                                           const Point& pos, size_t listIdx = 0);
extern PARAMDLLAPI string getStrAreaParam(const string& tag, const string& paramName,
                                                        const Point& pos, size_t listIdx = 0);

///
/// \brief getIntParam, get assigned global area params
///     Get assigned param with <tag, paramName> in "Global" area.
///     NOTE:
///         1. could get three type param: int, double and str
///         2. param type not consistant with called function will return 0 or ""
///         3. for more efficient usage, use param like [string a("name")] other than
///             temporary variables like [getIntParam("name")]
/// \param tag, tag name to find parameter
/// \param paramName, name of parameter to find
/// \return
///     value of assigned parameter
///
extern PARAMDLLAPI int getIntParam(const string& tag, const string& paramName, size_t listIdx = 0);
extern PARAMDLLAPI double getDoubleParam(const string& tag, const string& paramName, size_t listIdx = 0);
extern PARAMDLLAPI string getStrParam(const string& tag, const string& paramName, size_t listIdx = 0);

///
/// \brief setParam, set assigned param with assigned val.
///     NOTE:
///         1. if val outside range of assigned param, won't have effect
/// \param areaName, area to modify parameter values
/// \param tag, tag name to find parameter
/// \param paramName, name of parameter to find
/// \param val, new val to set
/// \return
///     true, if set val success;
///     false, if:
///         1. new val out of parameter range
///         2. new val doesn't match with parameter type
///
extern PARAMDLLAPI bool setParam(const string& areaName, const string& tag,
                                               const string& paramName, const string& val
                                               , size_t listIdx = 0);

///
/// \brief The ParamHolder class, used to return param address, not thread safe.
///
class PARAMDLLAPI ParamHolder
{
public:
    ParamHolder() {}
    virtual ~ParamHolder() {}
    virtual int getIntAreaParam(const string& tag, const string& paramName) = 0;
    virtual double getDoubleAreaParam(const string& tag, const string& paramName) = 0;
    virtual string getStrAreaParam(const string& tag, const string& paramName) = 0;
    virtual int getIntParam(const string& tag, const string& paramName) = 0;
    virtual double getDoubleParam(const string& tag, const string& paramName) = 0;
    virtual string getStrParam(const string& tag, const string& paramName) = 0;
};

extern PARAMDLLAPI std::vector<ParamHolder*>& getParamAtCol(size_t col, size_t listIdx = 0);

///
/// \brief getParamIdx, for easy trans on grid between signal process and param manager
/// \param dis, distance of param
/// \param listIdx, which paramManager to use
/// \return
///     dis idx of param manager
///
extern PARAMDLLAPI int getParamIdx(double dis, size_t listIdx = 0);

///
/// \brief getGlobalTags, get all param tags under <GeneralParams>
/// \return
///     tag name vector under tag <GeneralParams>
///
extern PARAMDLLAPI std::vector<string> getGlobalTags(size_t listIdx = 0);

///
/// \brief getAreaTags, get all param tags under <Global>
///     NOTE: all areas has same param tags
/// \return
///     param tag name vector under tag <Global>
///
extern PARAMDLLAPI std::vector<string> getAreaTags(size_t listIdx = 0);

///
/// \brief getGlobalTagParams, get all param names under assigned global tag
/// \param tagName, global tag to get all param names
/// \return
///     param names vector under assigned tagName
///
extern PARAMDLLAPI std::vector<string> getGlobalTagParams(const string& tagName, size_t listIdx = 0);

///
/// \brief getAreaTagParams, get all param names under assigned area tag
/// \param tagName, area tag to get all param names
/// \return
///     param names vector under assigned tagName
///
extern PARAMDLLAPI std::vector<string> getAreaTagParams(const string& tagName, size_t listIdx = 0);

///
/// \brief getParamType, get type of assigned param
/// \param tagName, area tag to get param type
/// \param paramName, param tag to get param type
/// \return
///     type of assigned param, <tagName, paramName>
///
extern PARAMDLLAPI NRXVARTYPE getParamType(const string& tagName, const string& paramName, size_t listIdx = 0);

///
/// \brief getIntParamInfo, get assigned params info.
///     Get assigned param info with <tag, paramName> in area with assigned areaName.
/// \param areaName, area name to find parameter
/// \param tagName, tag name to find parameter
/// \param paramName, name of parameter to find
/// \return
///     assigned param info
///
extern PARAMDLLAPI AttrInt getIntParamInfo(const string& areaName, const string& tagName, const string& paramName, size_t listIdx = 0);
extern PARAMDLLAPI AttrDouble getDoubleParamInfo(const string& areaName, const string& tagName, const string& paramName, size_t listIdx = 0);
extern PARAMDLLAPI AttrStr getStrParamInfo(const string& areaName, const string& tagName, const string& paramName, size_t listIdx = 0);

///
/// \brief getIntParam, get assigned params val.
///     Get assigned param val with <tag, paramName> in area with assigned areaName.
/// \param areaName, area name to find parameter
/// \param tag, tag name to find parameter
/// \param paramName, name of parameter to find
/// \return
///     assigned param val
///
extern PARAMDLLAPI int getIntParam(const string& areaName, const string& tag, const string& paramName, size_t listIdx = 0);
extern PARAMDLLAPI double getDoubleParam(const string& areaName, const string& tag, const string& paramName, size_t listIdx = 0);
extern PARAMDLLAPI string getStrParam(const string& areaName, const string& tag, const string& paramName, size_t listIdx = 0);

typedef struct AreaInfo {
    PointsVec pts;
    string name;
    AREATYPE type;
} AreaInfo;

///
/// \brief getAreasInfo, get all areas info
/// \return
///     AreaInfo vector, contain all areas info
///
extern PARAMDLLAPI std::vector<AreaInfo> getAreasInfo(size_t listIdx = 0);

} // namespace libParam

#endif // PARAMSSRC_H
