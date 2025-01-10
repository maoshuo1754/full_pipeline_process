/*
* NRxType.h
*
* Description:
* 		Defines types in NRxSDK.
*
* HINTS:
* 		20210512	0.0.1.	DDC.	Create the first version.
*/

#ifndef NRXTYPE_H
#define NRXTYPE_H

#include <assert.h>
#include <math.h>

// enviroment
#if defined(QT_CORE_LIB)
    #if !defined(QT_NO_DEBUG)
        #define DEBUG
    #else
        #define RELEASE
    #endif// QT_NO_DEBUG
#else
    // in eclipse DEBUG or RELEASE defined in Settings
#endif// QT_CORE_LIB


#ifdef DEBUG
#define ASSERT(x)  assert(x)
#else
#define ASSERT(x)  (void)(x)
#endif // DEBUG

/*********************************************************************
*
*   Constants
*
**********************************************************************/
/* Define the speed of light in metres per second. */
#define ACTUAL_LIGHT_SPEED_MPS	(299711000.0)	// 实际光速，单位：1m/s
#define COMMON_LIGHT_SPEED_MPS	(3.0e8)			// 常用光速，单位：1m/s
#define LIGHT_AMEND_COEF    (0.999036666666667) // 光速修正系数 = 实际光速 / 常用光速

/* Define default longitude and latitude */
// 启动试验场位置
#define DEFAULT_LONGITUDE (121.861559)
#define DEFAULT_LATITUDE (31.893568)

/* Define NM <---> M */
#define NM_2_METERS	(1852.0)
#define METER_2_NM	(1.0 / 1852.0)

/* Define M/S <---> KN */
#define KN_2_MPS	(NM_2_METERS / (60.0 * 60.0))
#define MPS_2_KN	(60.0 * 60.0 / NM_2_METERS)

/* Define M/S <---> KM/H */
#define KPH_2_MPS		(1000.0 / (60.0 * 60.0))
#define MPS_2_KPH		(60.0 * 60.0 / 1000.0)

#define EPSILON_6	(1e-6)
#define EPSILON_12	(1e-12)

static const double g_secOneDay = 86400; // 一天的秒数
static const double g_milliSecOneDay = 86400000; // 一天的毫秒数

#define NRX_1K	(1024u)
#define NRX_1M	(NRX_1K * NRX_1K)
#define NRX_1G	(NRX_1K * NRX_1M)

/* Assert an expression is true at compile time. */
template<bool> struct StaticAssert;
template<> struct StaticAssert<true> {};
#define STATIC_ASSERT(expr, msg)					\
struct msg								\
{									\
    StaticAssert<static_cast<bool>((expr))> StaticAssert__##msg;	\
}

/* "type" must be a template type. */
#define STATIC_ASSERT_ALWAYS(type, msg)				\
    STATIC_ASSERT(sizeof(type) == -1, msg)

// 检查指针是否为空。 如果为空，报错，将来改为 NRxError() .
#define NRX_REQUIRE_POINT_NO_RETURN(pointer) if(nullptr == (pointer)){ASSERT(false);}
#define NRX_REQUIRE_POINT_RETURN_EMPTY(pointer) if(nullptr == (pointer)){ASSERT(false);return;}
#define NRX_REQUIRE_POINT_RETURN_VAL(pointer, val) if(nullptr == (pointer)){ASSERT(false);return val;}

// 检查item是否为真 . 如果 false 报错，将来改为 NRxError() .
#define NRX_REQUIRE_ITEM_NO_RETURN(item) if(!(item)){ASSERT(false);}
#define NRX_REQUIRE_ITEM_RETURN_EMPTY(item) if(!(item)){ASSERT(false);return;}
#define NRX_REQUIRE_ITEM_RETURN_VAL(item, val) if(!(item)){ASSERT(false);return val;}

// 析构指针
#define NRX_DELETE_POINTER_WITH_ASSIGN_NULL(pointer) if(nullptr != (pointer)){delete (pointer); (pointer) = nullptr;}
#define NRX_DELETE_POINTER_WITHOUT_ASSIGN_NULL(pointer) if(nullptr != (pointer)){delete (pointer);}
#define NRX_DELETE_POINTER_ARRAY_WITH_ASSIGN_NULL(pointer) if(nullptr != (pointer)){delete[] (pointer); (pointer) = nullptr;}
#define NRX_DELETE_POINTER_ARRAY_WITHOUT_ASSIGN_NULL(pointer) if(nullptr != (pointer)){delete[] (pointer);}

// Avoid "unused parameter" warnings
#define NRX_UNUSED(x) (void)(x);

#define NRX_OBJ_PROTECT		(0xABCD5678u)	// const value to check


/*********************************************************************
*
*   Type definitions
*
**********************************************************************/

/* Define the common fixed-size types we use throughout.
* They also need to be checked at runtime
* by one of the Init functions to make sure their sizes are
* correct.
*/
typedef unsigned long long uint64;
typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;
typedef long long int64;
#ifndef XMD_H /* Xmd.h defines int32 as long. */
#ifdef __sun
typedef long int32;   /* Needs "long" on Solaris (clash with Xmd.h) */
#else
typedef int int32;
#endif
#endif
typedef short int16;
typedef signed char int8;	/* Needs "signed" for windows */
typedef unsigned char uchar;
typedef float real32;
typedef double real64;

/* Check sizes of types. */
STATIC_ASSERT(sizeof(uint64) == 8, uint64_is_incorrect_size);
STATIC_ASSERT(sizeof(uint32) == 4, uint32_is_incorrect_size);
STATIC_ASSERT(sizeof(uint16) == 2, uint16_is_incorrect_size);
STATIC_ASSERT(sizeof(uint8) == 1, uint8_is_incorrect_size);
STATIC_ASSERT(sizeof(int64) == 8, int64_is_incorrect_size);
STATIC_ASSERT(sizeof(int32) == 4, int32_is_incorrect_size);
STATIC_ASSERT(sizeof(int16) == 2, int16_is_incorrect_size);
STATIC_ASSERT(sizeof(int8) == 1, int8_is_incorrect_size);
STATIC_ASSERT(sizeof(uchar) == 1, uchar_is_incorrect_size);
STATIC_ASSERT(sizeof(real32) == 4, real32_is_incorrect_size);
STATIC_ASSERT(sizeof(real64) == 8, real64_is_incorrect_size);

/* Define TRUE and FALSE. */
//#ifndef FALSE
//#define	FALSE	(0)
//#else
//#error ("Macro 'FALSE' has been defined.")
//#endif

//#ifndef TRUE
//#define	TRUE	(1)
//#else
//#error ("Macro 'TRUE' has been defined.")
//#endif

//#ifndef BOOL
//#define BOOL    (int32)
//#else
//#error ("Macro 'BOOL' has been defined.")
//#endif

//#ifndef NULL
//#define NULL (0)
//#else
//#error ("Macro 'NULL' has been defined.")
//#endif

#endif// NRXTYPE_H
