/* *****************************************************************************
// [fsdbShr.h]
// Copyright and Proprietary Information Notice
// Copyright (c) 2015 Synopsys, Inc. All rights reserved. 
// This software and documentation contain confidential and proprietary information 
// that is the property of Synopsys, Inc. 
// The software and documentation are furnished under a license agreement and may be used 
// or copied only in accordance with the terms of the license agreement. 
// No part of the software and documentation may be reproduced, transmitted, or translated, 
// in any form or by any means, electronic, mechanical, manual, optical, or otherwise, 
// without prior written permission of Synopsys, Inc., or as expressly provided 
// by the license agreement. 
// The only license agreement under which you may obtain a license to this FSDB Writer 
// (library name libnffw.so) is the VC Apps Access Program Agreement 
// (or a future agreement that replaces the VC Apps Access Program Agreement) . 
// If you are interested in joining the VC Apps Access Program, please contact a Synopsys, Inc. 
// representative. 
// You are not permitted to use this FSDB Writer under other agreements, 
// including (without limitation) your Synopsys’ end user software license and maintenance agreement 
// or evaluation agreement.
// ****************************************************************************/
/* *****************************************************************************
// [fsdbShr.h]
// Copyright and Proprietary Information Notice
// Copyright (c) 2015 Synopsys, Inc. All rights reserved. 
// This software and documentation contain confidential and proprietary information 
// that is the property of Synopsys, Inc. The software and documentation are furnished under a license 
// agreement and may be used or copied only in accordance with the terms of the license agreement. 
// No part of the software and documentation may be reproduced, transmitted, or translated, 
// in any form or by any means, electronic, mechanical, manual, optical, or otherwise, 
// without prior written permission of Synopsys, Inc., or as expressly provided by the license agreement. 
// The only license agreement under which you may obtain a license to this FSDB Reader (library name libnffr.so) 
// is the VC Apps Access Program Agreement (or a future agreement that replaces the VC Apps Access Program Agreement) . 
// If you are interested in joining the VC Apps Access Program, please contact a Synopsys, Inc. representative. 
// You are not permitted to use this FSDB Reader under other agreements, including (without limitation) your Synopsys’ 
// end user software license and maintenance agreement or evaluation agreement.
// ****************************************************************************/
#ifndef fsdbShr_h
#define fsdbShr_h

#ifndef sysOS_h         /* --- begin of sysOS_h --- */
#ifndef str_T_DEFINED
#define str_T_DEFINED
typedef char*   str_T;
#endif
 
#ifdef __cplusplus
#define ET_BEGIN_FUNCS                  extern "C" {
#define ET_END_FUNCS                    }
#else
#define ET_BEGIN_FUNCS
#define ET_END_FUNCS
#endif
#endif                  /* --- end of sysOS_h --- */

#ifndef sysTypes_h
#ifndef bool_T_DEFINED
#define bool_T_DEFINED
typedef char    bool_T; 
#endif
#endif

#if !defined(__byte_T__)
#define __byte_T__
typedef unsigned char   byte_T;   
#endif
 
#if !defined(__ushort_T__)
#define __ushort_T__
typedef unsigned short  ushort_T; 
#endif
 
#if !defined(__uint_T__)
#define __uint_T__
typedef unsigned int    uint_T;   
#endif
 
#if !defined(__ulong_T__)
#define __ulong_T__
#ifdef _WIN64
typedef unsigned long long  ulong_T;
#else
typedef unsigned long   ulong_T;
#endif
#endif

#if !defined(__long_T__)
#define __long_T__
#ifdef _WIN64
// LLP64 
typedef long long  long_T;
#else
// LP64
typedef long long_T;
#endif //_WIN64
#endif //__long_T__

#if PORT_WINDOWS
#if !defined(__longlong_T__)
#define __longlong_T__
typedef signed __int64      longlong_T;
#endif
#if !defined(__ulonglong_T__)
#define __ulonglong_T__
typedef unsigned __int64    ulonglong_T;
#endif
#else /*PORT_WINDOWS*/
#if !defined(__longlong_T__)
#define __longlong_T__
typedef signed long long    longlong_T;
#endif
#if !defined(__ulonglong_T__)
#define __ulonglong_T__
typedef unsigned long long  ulonglong_T;
#endif
#endif /*PORT_WINDOWS*/

/*
** fsdb Return Code 
*/
enum fsdbRC {
    FSDB_RC_VAR_IDCODE_OVER_TWOG                = (-34),
    FSDB_RC_FILE_LOCK_FAIL_BY_SIMV              = (-33),
    FSDB_RC_FILE_UNLOCK_FAIL_BY_SIMV            = (-32),
    FSDB_RC_FILE_IS_LOCKED_BY_SIMV              = (-31),
    FSDB_RC_INVALID_OFFSET                      = (-30),
    FSDB_RC_INVALID_VAR_IDCODE                  = (-29),
    FSDB_RC_NAMING_VIOLATION                    = (-28),
    FSDB_RC_ENDIAN_TYPE_UNKNOWN                 = (-27),
    FSDB_RC_FSDB_VERSION_IS_NOT_VALID           = (-26),
    FSDB_RC_IMPOSSIBLE_CASE                     = (-25),
    FSDB_RC_INVALID_BYTES_PER_BIT               = (-24),
    FSDB_RC_INVALID_VAR_TYPE                    = (-23),
    FSDB_RC_SCOPE_TYPE_IS_OUT_OF_RANGE          = (-22),
    FSDB_RC_TREE_CREATION_IS_NOT_BALANCED       = (-21),
    FSDB_RC_INVALID_FILE_TYPE                   = (-20),
    FSDB_RC_FILE_IS_NOT_LOCKED                  = (-19),
    FSDB_RC_WRITE_MODE_IS_NOT_CORRECT           = (-18),
    FSDB_RC_DATA_TYPE_HANDLE_IS_NOT_FOUND       = (-17),
    FSDB_RC_VAR_IDCODE_IS_NOT_CONTINUOUS        = (-16),
    FSDB_RC_BIT_SIZE_IS_OUT_OF_RANGE            = (-15),
    FSDB_RC_BIT_NUM_IS_WRONG                    = (-14),
    FSDB_RC_ARGUMENT_IS_INVALID                 = (-13),
    FSDB_RC_ARGUMENT_IS_OUT_OF_RANGE            = (-12),
    FSDB_RC_VAR_DIR_IS_OUT_OF_RANGE             = (-11),
    FSDB_RC_VAR_TYPE_IS_OUT_OF_RANGE            = (-10),
    FSDB_RC_POINTER_IS_NULL                     = (-9),
    FSDB_RC_WRITE_OPERATION_FAILED              = (-8),
    FSDB_RC_FSEEK_FAILED                        = (-7),
    FSDB_RC_OBJECT_CREATION_FAILED              = (-6),
    FSDB_RC_ALL_VCS_HAVE_BEEN_READ              = (-5),
    FSDB_RC_ALL_LOAD_RANGE_VCS_HAVE_BEEN_READ   = (-4),
    FSDB_RC_MEMORY_IS_EXHAUSTED                 = (-3),
    FSDB_RC_RETURN_ANYWAY                       = (-2),

    FSDB_RC_FAILURE                             = (-1),
    FSDB_RC_SUCCESS                             = 0,
    FSDB_RC_EOF                                 = 1,
    FSDB_RC_UNEXPECTED_VALUE_FOUND              = 2,

    FSDB_RC_FILE_DOES_NOT_EXIST                 = 10,
    FSDB_RC_FILE_IS_A_DIRECTORY                 = 11,
    FSDB_RC_FILE_IS_EMPTY                       = 12,
    FSDB_RC_FILE_IS_NOT_A_FSDB_FILE             = 13,
    FSDB_RC_FILE_CONVERSION_FAILED              = 14,
    FSDB_RC_FILE_IS_THE_SAME                    = 15,
    FSDB_RC_FILE_IS_NOT_READABLE                = 16,
    FSDB_RC_FILE_IS_CORRUPT                     = 17,
    FSDB_RC_FILE_HAS_NOT_BEEN_MODIFIED          = 18,

    FSDB_RC_CANNOT_LOCATE_VAR_ID                = 21,
    FSDB_RC_CANNOT_LOCATE_VAR_IDCODE            = 22,
    FSDB_RC_SIGNAL_LIST_HAS_NOT_SIGNAL          = 24,
    FSDB_RC_ASSOC_MEM_NOT_ALLOCATED             = 25,
    FSDB_RC_FAILED_TO_OPEN_THE_FILE             = 26,
    FSDB_RC_FAILED_TO_GET_A_LINE                = 27,
    FSDB_RC_FFW_VAR_CREATION_FAILED             = 28,
    FSDB_RC_VAR_IDCODE_IS_ZERO                  = 29,
    FSDB_RC_INADEQUATE_USAGE                    = 30,
    FSDB_RC_UNSUPPORTED_FILE_TYPE               = 31,
    FSDB_RC_UNKOWN_FILE_TYPE                    = 32,
    FSDB_RC_UNKNOWN_FILE_TYPE                   = 32,
    FSDB_RC_IN_USE                              = 33,

    FSDB_RC_VCD_UNCONT_ID_STRING                = 80,
    FSDB_RC_VCD_END_OF_DEF                      = 81,
    FSDB_RC_VCD_TIME_IS_REWOUND                 = 82,
    FSDB_RC_VCD_UNEXPECTED_KEYWORD              = 83,
    FSDB_RC_VCD_WRONG_BIT_TYPE                  = 84,
    FSDB_RC_VCD_ID_STRING_IS_NOT_FOUND          = 85, 
    FSDB_RC_FILE_IS_NOT_A_VCD_FILE              = 86,

    FSDB_RC_THERE_IS_NO_TREE                    = 87,
    FSDB_RC_VCD_LONG_ID_STRING                  = 88,

    FSDB_RC_GET_USER_ABORT                      = 89,

    FSDB_RC_XTAG_IS_THE_SAME                    = 100,
    FSDB_RC_FTAG_IS_THE_SAME                    = 101,
    FSDB_RC_DTAG_IS_THE_SAME                    = 102,
    FSDB_RC_LTAG_IS_THE_SAME                    = 103,

    FSDB_RC_NO_MORE_DATA                        = 104,
    FSDB_RC_RECYCLE_FIFO_FAIL                   = 105,

    FSDB_RC_LIBRARY_INCOMPATIBLE                = 106,

    FSDB_RC_PROBE_FOUND                         = 107,
    FSDB_RC_PROBE_NOT_FOUND                     = 108,
    FSDB_RC_PROBE_ERROR                         = 109,

    FSDB_RC_TRANS_ID_NOT_FOUND                  = 110,
    FSDB_RC_RELATION_NOT_FOUND                  = 111,

    FSDB_RC_NO_SUCH_ADDR                        = 112,

    FSDB_RC_INVALID_XTAG_TYPE                   = 122,
    FSDB_RC_SCALE_UNIT_NOT_SET                  = 123,
    FSDB_RC_SYNC_CONTROL_IS_NOT_OKAY            = 124,
    FSDB_RC_INVALID_CKSUM                       = 125,
    FSDB_RC_NO_VALUE_CHANGE                     = 126,
    FSDB_RC_VIEW_WINDOW_NOT_SET                 = 127,
    FSDB_RC_FIELD_HAS_BEEN_SET_BEFORE           = 128,

    FSDB_RC_HAS_NO_X_VALUE                      = 129,
    FSDB_RC_HAS_X_VALUE_UNKNOWN                 = 130,

    FSDB_TOP_VAR_NOT_LOAD                       = 131,
    FSDB_RC_REMOVED_NAVDB                       = 132,

    FSDB_RC_FHDB_NETWORK_ERROR                  = 133
};
typedef enum fsdbRC                             fsdbRC;

enum fsdbVrtlFileType {
    FSDB_VRTL_FILE_TYPE_MIN             = 0x00,
    FSDB_VRTL_FILE_TYPE_NONE            = 0x00,
    FSDB_VRTL_FILE_TYPE_SWITCH          = 0x01,
    FSDB_VRTL_FILE_TYPE_SPLIT           = 0x02,
    FSDB_VRTL_FILE_TYPE_STITCH          = 0x03,
    FSDB_VRTL_FILE_TYPE_STREAMLINED     = 0x04,
    FSDB_VRTL_FILE_TYPE_MAX             = 0x04
};
typedef enum fsdbVrtlFileType       fsdbVrtlFileType; 


typedef float                   fsdbTagFloat;
typedef double                  fsdbTagDouble;

typedef unsigned int            fsdbTag32;
struct fsdbTag64 {
    fsdbTag32   H;      /* high part */
    fsdbTag32   L;      /* low part  */
};
typedef struct fsdbTag64        fsdbTag64;


typedef uint_T                  fsdbSeqNum;
typedef uint_T                  fsdbSeqNum_T;

#define FSDB_EMPTY_SEQ_NUM      0
#define FSDB_EMPTY_SEQNUM       0

#define FSDB_DFT_SCOPE_SEPARATOR        "/"
#define FSDB_DEFAULT_SCOPE_SEP          FSDB_DFT_SCOPE_SEPARATOR 

#define FSDB_DFT_CMP_METHOD     0

/*
** #define FSDB_MAX_BIT_SIZE       65516
**
** From Debussy 5.2 (fsdb v2.2), the max bit size is changed from
** 65516 to 256K.
**
** #define FSDB_MAX_BIT_SIZE       (256 * 1024) 
**
** From Debussy 5.3 (fsdb v2.3), the max bit size is changed from
** 256K to 2M.
*/
#define FSDB_MAX_BIT_SIZE       (2 * 1024 * 1024)
#define FSDB_MAX_BUF_SIZE       1024
#define FSDB_MAX_PATH           FSDB_MAX_BUF_SIZE

#define FSDB_MAX_SCOPE_TYPE     57      /* total 58 scope types */
#define FSDB_MAX_VAR_TYPE       63      /* total 64 var types */
 
 
#define FSDB_SIM_VERSION_LEN    32
#define FSDB_SIM_DATE_LEN       28
#define FSDB_SCOPE_SEP_LEN      4
#define FSDB_SCALE_UNIT_LEN     16

#define FSDB_ANNOTATION_LEN     512

#define FSDB_CURRENT_UNIT_LEN   16
#define FSDB_VOLTAGE_UNIT_LEN   16
#define FSDB_MATH_UNIT_LEN      16

enum fsdbFileType {
    FSDB_FT_VERILOG             = 0, 
    FSDB_FT_EPIC                = 1, 
    FSDB_FT_MENTOR              = 1, 
    FSDB_FT_NANOSIM             = 1, 

    /*
     * SPICE file type can only be accomodated by 1.6 format. Formal release
     * Verdi with latest FSDB (>2.x) writer does not have SPICE file type
     * support.
     */
    FSDB_FT_SPICE               = 2,

    FSDB_FT_XP                  = 3, 
    FSDB_FT_RAWFILE             = 4, 
    FSDB_FT_WFM                 = 5,
    FSDB_FT_VHDL                = 6, 
    FSDB_FT_VERILOG_VHDL        = 7,

    FSDB_FT_EBC_VERILOG         = 8,  
    /*
    ** verilog with Novas essential dumping
    */

    FSDB_FT_EBC_VHDL            = 9, 
    /*
    ** vhdl with Novas essential dumping
    */

    FSDB_FT_EBC_VERILOG_VHDL    = 10, 
    /*
    ** verilog and vhdl with Novas essential dumping
    */

    FSDB_FT_IKOS_VERILOG        = 11,  
    /*
    ** verilog with IKOS reconstruction engine
    */

    FSDB_FT_IKOS_VHDL           = 12, 
    /*
    ** vhdl with IKOS reconstruction engine 
    */

    FSDB_FT_IKOS_VERILOG_VHDL   = 13,
    /*
    ** verilog and vhdl with IKOS reconstruction engine 
    */

    FSDB_FT_VERILOG_NANOSIM     = 14,
    /* 
    ** used for merged verilog + nanosim(epic) fsdb 
    */

    FSDB_FT_FLE_VERILOG         = 15,
    FSDB_FT_FLE_VHDL            = 16,
    FSDB_FT_FLE_VERILOG_VHDL    = 17,

    /*
    ** Novas internal use only
    */
    FSDB_FT_UDDB                = 18,

    FSDB_FT_MAX                 = FSDB_FT_UDDB
    /*
    ** Mark boundary of value values
    */
};
typedef enum fsdbFileType       fsdbFileType;
#define FSDB_MAX_FILE_TYPE      FSDB_FT_MAX

#define fsdbDesignType          fsdbFileType
#define FSDB_MAX_DESIGN_TYPE    FSDB_MAX_FILE_TYPE

/*
** Make file status be within [0, 15] so that we 
** can store it with 4 bits. 
*/
enum fsdbFileStatus {
    FSDB_FS_FINISHED                    = 0x00,
    FSDB_FS_INTERACTIVE                 = 0x01,
    FSDB_FS_BATCH                       = 0x02,
    FSDB_FS_UNKNOWN                     = 0x0f
};
typedef enum fsdbFileStatus             fsdbFileStatus;
#define FSDB_MAX_VALID_FILE_STATUS      0x0f    

/*
** fsdbGlitchDetectionType should be placed in ffwAPI.h
*/ 
enum fsdbGlitchDetectionType {
    FSDB_GDT_CONTINUE_NEXT_VC, FSDB_GDT_STOP_NEXT_VC
};
typedef enum fsdbGlitchDetectionType      fsdbGlitchDetectionType;

enum fsdbScopeType {
    FSDB_ST_VCD_MODULE              = 0,
    FSDB_ST_VCD_TASK                = 1,
    FSDB_ST_VCD_FUNCTION            = 2,
    FSDB_ST_VCD_BEGIN               = 3,
    FSDB_ST_VCD_FORK                = 4,
    FSDB_ST_VCD_LIST                = 5,
    FSDB_ST_VCD_ARRAY_ELEM          = 6,
    FSDB_ST_VCD_GENERATE            = 7,

    FSDB_ST_VHDL_ARCHITECTURE       = 16,
    FSDB_ST_VHDL_PROCEDURE          = 17,
    FSDB_ST_VHDL_FUNCTION           = 18,
    FSDB_ST_VHDL_RECORD             = 19,
    FSDB_ST_VHDL_PROCESS            = 20,
    FSDB_ST_VHDL_BLOCK              = 21,
    FSDB_ST_VHDL_FOR_GENERATE       = 22,
    FSDB_ST_VHDL_IF_GENERATE        = 23,
    FSDB_ST_VHDL_GENERATE           = 24,
    

    /* below are for internal use only */

    FSDB_ST_SV_INTERFACE            = 31,
    FSDB_ST_SV_MODPORT              = 32,
    FSDB_ST_SV_INTERFACEPORT_REF    = 33,
    FSDB_ST_SV_MODPORT_REF          = 34,
    FSDB_ST_DESIGNWARE              = 35,
    FSDB_ST_SV_INTERFACE_PORT       = 36,
    FSDB_ST_SV_MODPORT_PORT         = 37,

    FSDB_ST_POWER_DOMAIN            = 45,
    FSDB_ST_POWER_SUPPLY_SET        = 46,
    FSDB_ST_POWER_STATE_TABLE       = 47,
    FSDB_ST_POWER_STATE_GROUP       = 48,
    FSDB_ST_POWER_SWITCH            = 49,
    FSDB_ST_POWER_SCOPE             = 50,
    FSDB_ST_POWER_STRATEGY_ISO      = 51,
    FSDB_ST_POWER_STRATEGY_RET      = 52,
    FSDB_ST_POWER_STRATEGY_LS       = 53,
    FSDB_ST_POWER_ROOT              = 54,

    FSDB_ST_SV_COMPUTATION_UNIT     = 63,
    FSDB_ST_SV_PACKAGE              = 64,
    FSDB_ST_SV_MODULE               = 65,
    FSDB_ST_SV_PROGRAM              = 66,
    FSDB_ST_SV_CLASS                = 67,

    FSDB_ST_SV_SUPER                = 68,
    FSDB_ST_SV_VERIFICATION_ROOT    = 69,
    
    FSDB_ST_SC_MODULE               = 81,
    
    FSDB_ST_SPECIAL_INDICATE_EVENT  = 90,

    FSDB_ST_SPICE                   = 91,
    FSDB_ST_SPICE_IE                = 92, // IE => INTERFACE_ELEMENT

    FSDB_ST_VERILOG_A               = 93,
    FSDB_ST_VERILOG_AMS             = 94,
    FSDB_ST_UNKNOWN                 = 99
};
typedef enum fsdbScopeType      fsdbScopeType;
#define FSDB_ST_MIN_VALUE       0
#define FSDB_ST_MAX_VALUE       57
#define FSDB_SCOPE_BLK_SC       FSDB_ST_SC_MODULE
/*
#ifndef FSDB_USE_32B_IDCODE
#define FSDB_USE_32B_IDCODE 1
#endif
*/
#ifdef WIN32
#ifndef _WIN64
#define FSDB_USE_32B_IDCODE 1
#endif // _WIN64
#endif // WIN32

#ifdef FSDB_USE_32B_IDCODE
typedef int                     fsdbVarIdcode;
typedef uint_T                  fsdbVarCount;
#else
typedef longlong_T              fsdbVarIdcode;
typedef ulonglong_T             fsdbVarCount;
#endif
typedef int                     fsdbInstanceIdcode;
        
#define FSDB_INVALID_VAR_IDCODE (-1)
#define FSDB_MIN_VAR_IDCODE     1       

#ifdef FSDB_USE_32B_IDCODE
#define FSDB_MAX_VAR_IDCODE     (INT_MAX-1)
#define FSDB_MAX_VF_VAR_IDCODE  (UINT_MAX)
#else
#define FSDB_MAX_VAR_IDCODE     (LLONG_MAX-1)
#define FSDB_MAX_VF_VAR_IDCODE  (LLONG_MAX)
#endif

#define FSDB_MAX_ALLOW_VAR_COUNT 17179869184ull

#define FSDB_INSTANCE_NULL_VALUE_IDCODE 0

enum fsdbVarType {

#define FSDB_VT_MIDDLE          64      /* 0100 0000 == 64  */
#define FSDB_VT_COMPUTED        128     /* 1000 0000 == 128 */  
#define FSDB_VT_MC_MASK         0xc0    /* 1100 0000 */
/* 
** FSDB_VT_MIDDLE and FSDB_VT_COMPUTED are used for 
** verilog and vhdl file type only, for other file types,
** FSDB_VT_MIDDLE and FSDB_VT_COMPUTED must not be used.
**
** FSDB_VT_MIDDLE and FSDB_VT_COMPUTED are used to mark a
** signal's type as MIDDLE or COMPUTED type. To mark it,
** just perform "logical or" onto the origianl var type and
** assign it back to signal's type; for example
**
** signal.type = FSDB_VT_VCD_WIRE | FSDB_VT_COMPUTED;
** signal.type = FSDB_VT_VCD_WIRE | FSDB_VT_MIDDLE;
**
** If var type of verilog/vhdl exceed 63, then we will get 
** problem and need to come out a new way to handle MIDDLE 
** and COMPUTED signal. 
**
** Note that only verilog/vhdl/verilog_vhdl file type fsdb
** has computed and middle type signals.
*/

    FSDB_VT_VCD_EVENT           = 0,
    FSDB_VT_VCD_INTEGER         = 1,
    FSDB_VT_VCD_PARAMETER       = 2,
    FSDB_VT_VCD_REAL            = 3,
    FSDB_VT_VCD_REG             = 4,
    FSDB_VT_VCD_SUPPLY0         = 5,
    FSDB_VT_VCD_SUPPLY1         = 6,
    FSDB_VT_VCD_TIME            = 7,
    FSDB_VT_VCD_TRI             = 8,
    FSDB_VT_VCD_TRIAND          = 9,
    FSDB_VT_VCD_TRIOR           = 10,
    FSDB_VT_VCD_TRIREG          = 11,
    FSDB_VT_VCD_TRI0            = 12,
    FSDB_VT_VCD_TRI1            = 13,
    FSDB_VT_VCD_WAND            = 14,
    FSDB_VT_VCD_WIRE            = 15,
    FSDB_VT_VCD_WOR             = 16,
    FSDB_VT_VCD_MEMORY          = 17,
    FSDB_VT_VCD_MEMORY_DEPTH    = 18,
    FSDB_VT_VCD_MEMORY_RANGE    = 18,
    FSDB_VT_VCD_PORT            = 19,   /* for extended vcd format */
    FSDB_VT_VCD_REG2            = 20,   /* for internal 6-value register */

    FSDB_VT_VCD_PACKED          = 21,   /* for internal use only */
    FSDB_VT_VCD_PACKED_ENUM     = 22,   /* for internal use only */

    FSDB_VT_EXT_VAR             = 23,   /* reserved for extended var type */

    FSDB_VT_AMS_SIGNAL          = 24,   /* for AMS */

    FSDB_VT_SV_VARIABLE         = 25,   /* for internal use only */
    FSDB_VT_SV_VARIABLE_THIS    = 26,   /* for internal use only */

    FSDB_VT_SV_MAX_DELTA        = 28,
    FSDB_VT_SV_METHOD           = 29,   /* for internal use only */
    FSDB_VT_SV_METHOD_SEQUENCE  = 30,   /* for internal use only */
    FSDB_VT_SV_THREAD_SEQUENCE  = 31,   /* for internal use only */

    FSDB_VT_VHDL_SIGNAL         = 32,
    FSDB_VT_VHDL_VARIABLE       = 33,
    FSDB_VT_VHDL_CONSTANT       = 34,
    FSDB_VT_VHDL_FILE           = 35,
    FSDB_VT_VHDL_MEMORY         = 36,
    FSDB_VT_VHDL_MEMORY_DEPTH   = 37,
    FSDB_VT_VHDL_MEMORY_RANGE   = 37,
    FSDB_VT_EVENT_VARIABLE      = 38,   /* for internal use only */

    FSDB_VT_SC_SIGNAL           = 39,   /* for internal use only */
    FSDB_VT_SC_PLAIN            = 40,   /* for internal use only */
    FSDB_VT_POWER_SIGNAL        = 41,   /* for internal use only */
    FSDB_VT_SV_INSTANCE	        = 42,   /* for internal use only */
    
    FSDB_VT_EXPR_RESULT         = 43,   /* for internal use only */

    FSDB_VT_REFERENCE           = 46,   /* for internal use only */
    FSDB_VT_INSTANCE            = 47,   /* for internal use only */
    FSDB_VT_MDA                 = 48,   /* Multiple dimensions array, for internal use only */
    FSDB_VT_STREAM              = 49,   /* Stream variable. For transaction */

    FSDB_VT_LOOP_MARKER	        = 50,	/* Loop marker. */
    FSDB_VT_COVERAGE            = 51,   /* Coverage var */ 
    FSDB_VT_LIST_OBSOLETE       = 52,   /* List var (obsolete) */
    FSDB_VT_LIST                = 53,   /* List var (new) */

    FSDB_VT_PROP_MIN            = 54,   /* Minium property var type */
    FSDB_VT_PROP_ASSUME         = 54,   /* Property assume(similiar to
                                        ** assert but shown in different way) */
    FSDB_VT_PROP_RESTRICT       = 55,   /* Property input constraint */
    FSDB_VT_PROP_COVER          = 56,   /* Property cover classe */
    FSDB_VT_PROP_LOCAL_MEMORY   = 57,   /* Property local memory clause */
    FSDB_VT_PROP_LOCAL          = 58,	/* Property local clause  */
    FSDB_VT_PROP_EVENT          = 59,	/* Property event clause  */
    FSDB_VT_PROP_BOOL           = 60,	/* Property bool clause   */
    FSDB_VT_PROP_ASSERT_FORBID  = 61,	/* Property assert forbid clause */
    FSDB_VT_PROP_ASSERT_CHECK   = 62,   /* Property assert check clause  */
    FSDB_VT_PROP_ASSERT         = 62,   /* for backward compatible */
    FSDB_VT_PROP_MAX            = 62,   /* Maxium Property var type */

    FSDB_VT_STRING              = 63,   /* string variable 		     */

    /*
    ** EPIC var type
    */
    FSDB_VT_EPIC_LOGIC                  = 0,        /* byte */
    FSDB_VT_EPIC_VOLTAGE                = (int)'v', /* float */
    FSDB_VT_EPIC_INSTANTANEOUS_CURRENT  = (int)'i', /* float */
    FSDB_VT_EPIC_AVERAGE_RMS_CURRENT    = (int)'I', /* float */
    FSDB_VT_EPIC_DI_DT                  = (int)'d', /* float */
    FSDB_VT_EPIC_MATHEMATICS            = (int)'m', /* float */
    FSDB_VT_EPIC_POWER                  = (int)'p', /* double */

    /*
    ** Nanosim var type. They are substitues for EPIC var types since the 
    ** name EPIC is obsolete.
    */
    FSDB_VT_NANOSIM_LOGIC                   = 0,        /* byte */
    FSDB_VT_NANOSIM_VOLTAGE                 = (int)'v', /* float */
    FSDB_VT_NANOSIM_INSTANTANEOUS_CURRENT   = (int)'i', /* float */
    FSDB_VT_NANOSIM_AVERAGE_RMS_CURRENT     = (int)'I', /* float */
    FSDB_VT_NANOSIM_DI_DT                   = (int)'d', /* float */
    FSDB_VT_NANOSIM_MATHEMATICS             = (int)'m', /* float */
    FSDB_VT_NANOSIM_POWER                   = (int)'p', /* double */

    /*
    ** HSPICE var type
    */
    FSDB_VT_HSPICE_VOLTAGE      = 1,
    FSDB_VT_HSPICE_VOLTS_MAG    = 2,
    FSDB_VT_HSPICE_VOLTS_REAL   = 3,
    FSDB_VT_HSPICE_VOLTS_IMAG   = 4,
    FSDB_VT_HSPICE_PHASE        = 5,
    FSDB_VT_HSPICE_VOLTS_DB     = 6,
    FSDB_VT_HSPICE_VOLTS_TDLY   = 7,
    FSDB_VT_HSPICE_CURRENT      = 8,
    FSDB_VT_HSPICE_I_MAG        = 9,
    FSDB_VT_HSPICE_I_REAL       = 10,
    FSDB_VT_HSPICE_I_IMAG       = 11,
    FSDB_VT_HSPICE_I_PHASE      = 12,
    FSDB_VT_HSPICE_I_DECIBEL    = 13,
    FSDB_VT_HSPICE_I_TDLY       = 14,
    FSDB_VT_HSPICE_CURRENT_1    = 15,
    FSDB_VT_HSPICE_I_MAG_1      = 16,
    FSDB_VT_HSPICE_I_REAL_1     = 17,
    FSDB_VT_HSPICE_I_IMAG_1     = 18,
    FSDB_VT_HSPICE_I_PHASE_1    = 19,
    FSDB_VT_HSPICE_I_DECIBEL_1  = 20,
    FSDB_VT_HSPICE_I_TDLY_1     = 21,
    FSDB_VT_HSPICE_CURRENT_2    = 22,
    FSDB_VT_HSPICE_I_MAG_2      = 23,
    FSDB_VT_HSPICE_I_REAL_2     = 24,
    FSDB_VT_HSPICE_I_IMAG_2     = 25,
    FSDB_VT_HSPICE_I_PHASE_2    = 26,
    FSDB_VT_HSPICE_I_DECIBEL_2  = 27,
    FSDB_VT_HSPICE_I_TDLY_2     = 28,
    FSDB_VT_HSPICE_CURRENT_3    = 29,
    FSDB_VT_HSPICE_I_MAG_3      = 30,
    FSDB_VT_HSPICE_I_REAL_3     = 31,
    FSDB_VT_HSPICE_I_IMAG_3     = 32,
    FSDB_VT_HSPICE_I_PHASE_3    = 33,
    FSDB_VT_HSPICE_I_DECIBEL_3  = 34,
    FSDB_VT_HSPICE_I_TDLY_3     = 35,
    FSDB_VT_HSPICE_CURRENT_4    = 36,
    FSDB_VT_HSPICE_I_MAG_4      = 37,
    FSDB_VT_HSPICE_I_REAL_4     = 38,
    FSDB_VT_HSPICE_I_IMAG_4     = 39,
    FSDB_VT_HSPICE_I_PHASE_4    = 40,
    FSDB_VT_HSPICE_I_DECIBEL_4  = 41,
    FSDB_VT_HSPICE_I_TDLY_4     = 42,
    FSDB_VT_HSPICE_POWER        = 43,
    FSDB_VT_HSPICE_P_M          = 44,
    FSDB_VT_HSPICE_P_REAL       = 45,
    FSDB_VT_HSPICE_P_IMAG       = 46,
    FSDB_VT_HSPICE_P_PHASE      = 47,
    FSDB_VT_HSPICE_P_DB         = 48,
    FSDB_VT_HSPICE_P_TDLY       = 49,
    FSDB_VT_HSPICE_TPOWRD       = 50,
    FSDB_VT_HSPICE_NOI_V_SQ_HZ  = 51,
    FSDB_VT_HSPICE_INOISE       = 52,
    FSDB_VT_HSPICE_HD2          = 53,
    FSDB_VT_HSPICE_HD3          = 54,
    FSDB_VT_HSPICE_DIM2         = 55,
    FSDB_VT_HSPICE_SIM2         = 56,
    FSDB_VT_HSPICE_DIM3         = 57,
    FSDB_VT_HSPICE_Z11          = 58,
    FSDB_VT_HSPICE_Z21          = 59,
    FSDB_VT_HSPICE_Z12          = 60,
    FSDB_VT_HSPICE_Z22          = 61,
    FSDB_VT_HSPICE_ZIN          = 62,
    FSDB_VT_HSPICE_ZOUT         = 63,
    FSDB_VT_HSPICE_Y11          = 64,
    FSDB_VT_HSPICE_Y21          = 65,
    FSDB_VT_HSPICE_Y12          = 66,
    FSDB_VT_HSPICE_Y22          = 67,
    FSDB_VT_HSPICE_YIN          = 68,
    FSDB_VT_HSPICE_YOUT         = 69,
    FSDB_VT_HSPICE_H11          = 70,
    FSDB_VT_HSPICE_H21          = 71,
    FSDB_VT_HSPICE_H12          = 72,
    FSDB_VT_HSPICE_H22          = 73,
    FSDB_VT_HSPICE_S11          = 74,
    FSDB_VT_HSPICE_S21          = 75,
    FSDB_VT_HSPICE_S12          = 76,
    FSDB_VT_HSPICE_S22          = 77,
    FSDB_VT_HSPICE_PARAM        = 78,
    FSDB_VT_HSPICE_LX           = 79,
    FSDB_VT_HSPICE_LV           = 80, 
    FSDB_VT_HSPICE_SWEEP        = 81,
    FSDB_VT_HSPICE_MATH         = 82
    
};
typedef enum fsdbVarType        fsdbVarType;

#define FSDB_VT_MIN_VALUE       0   /* minimum value of var type */
#define FSDB_VT_MAX_VALUE       255 /* maximum value of var type */

enum fsdbVarDir {
    FSDB_VD_IMPLICIT    = 0,
    FSDB_VD_INPUT       = 1,    
    FSDB_VD_OUTPUT      = 2,
    FSDB_VD_INOUT       = 3,
    FSDB_VD_BUFFER      = 4,
    FSDB_VD_LINKAGE     = 5
};
typedef enum fsdbVarDir         fsdbVarDir;
#define FSDB_VD_MIN_VALUE       0    /* minimum value of var direction */
#define FSDB_VD_MAX_VALUE       5   /* maximum value of var direction */
    

typedef void*                   fsdbVarHandle;

enum fsdbValueChangeDataType {
    FSDB_VC_DT_UNKNOWN   = 0,
    FSDB_VC_DT_BYTE      = 1, 
    FSDB_VC_DT_SHORT     = 2, 
    FSDB_VC_DT_INT       = 3, 
    FSDB_VC_DT_LONG      = 4, 
    FSDB_VC_DT_FLOAT     = 5, 
    FSDB_VC_DT_DOUBLE    = 6,
    FSDB_VC_DT_HL_INT    = 7,
    FSDB_VC_DT_PHYSICAL  = 8
};
typedef enum fsdbValueChangeDataType    fsdbValueChangeDataType;

/*
** valid data type idcode is between 0 and 65535(ushort_T)
*/
typedef int                     fsdbDataTypeIdcode;
typedef fsdbDataTypeIdcode      fsdbDatatypeIdcode;

/*
** predefined data type idcode
*/
enum fsdbDataType {
    FSDB_DT_INVALID     = -1,

    /*
    ** FSDB_DT_BYTE, FSDB_DT_SHORT, 
    ** FSDB_DT_INT, FSDB_DT_INT_H_N_L,
    ** FSDB_DT_LONG_INT, FSDB_DT_FLOAT and
    ** FSDB_DT_DOUBLE are added from fsdb v2.0
    */
    FSDB_DT_BYTE        = 0,
    FSDB_DT_SHORT       = 511,
    FSDB_DT_INT         = 510,
    FSDB_DT_FLOAT       = 509,
    FSDB_DT_HL_INT      = 508,
    FSDB_DT_LONG        = 507,
    FSDB_DT_DOUBLE      = 506,


    FSDB_DT_VERILOG_STANDARD            = 0,    /* {0, 1, X, Z} + strength */
    FSDB_DT_VERILOG_REAL                = 1,    /* for real type value change */
    FSDB_DT_VERILOG_INTEGER             = 2,    /* for memory depth value change */
    /*
    ** [0, 7] are reserved for the possible data type extension of verilog
    */

    FSDB_DT_EPIC_DIGITAL                = 8,    /* digital */
    FSDB_DT_EPIC_ANALOG                 = 9,    /* digital */
    FSDB_DT_HSPICE                      = 10,   /* analog only */
    FSDB_DT_XP                          = 11,   /* analog only */
    FSDB_DT_RAWFILE                     = 12,   /* analog only */
    FSDB_DT_WFM                         = 13,   /* digital only */
    /*
    ** [8, 31] are reserved for misc data types.
    **
    ** [32, 511] are reserved for VHDL predefined data types.
    */
    FSDB_DT_VHDL_BOOLEAN                = 32,
    FSDB_DT_VHDL_BIT                    = 33,
    FSDB_DT_VHDL_BIT_VECTOR             = 34,
    FSDB_DT_VHDL_STD_ULOGIC             = 35,
    FSDB_DT_VHDL_STD_ULOGIC_VECTOR      = 36,
    FSDB_DT_VHDL_STD_LOGIC              = 37,
    FSDB_DT_VHDL_STD_LOGIC_VECTOR       = 38,
    FSDB_DT_VHDL_UNSIGNED               = 39,
    FSDB_DT_VHDL_SIGNED                 = 40,
    FSDB_DT_VHDL_INTEGER                = 41,
    FSDB_DT_VHDL_REAL                   = 42,
    FSDB_DT_VHDL_NATURAL                = 43,
    FSDB_DT_VHDL_POSITIVE               = 44,
    FSDB_DT_VHDL_TIME                   = 45,
    FSDB_DT_VHDL_CHARACTER              = 46,
    FSDB_DT_VHDL_STRING                 = 47,

    FSDB_DT_VHDL_STDLOGIC_1D            = 48,
    FSDB_DT_VHDL_STDLOGIC_TABLE         = 49,
    FSDB_DT_VHDL_LOGIC_X01_TABLE        = 50,
    FSDB_DT_VHDL_LOGIC_X01Z_TABLE       = 51,
    FSDB_DT_VHDL_LOGIC_UX01_TABLE       = 52,

    FSDB_DT_VHDL_X01                    = 53,
    FSDB_DT_VHDL_X01Z                   = 54,
    FSDB_DT_VHDL_UX01                   = 55,
    FSDB_DT_VHDL_UX01Z                  = 56,

    FSDB_DT_VHDL_SEVERITY_LEVEL         = 57,
    FSDB_DT_VHDL_DELAY_LENGTH           = 58,

    FSDB_DT_VHDL_LINE                   = 59,
    FSDB_DT_VHDL_TEXT                   = 60,
    FSDB_DT_VHDL_SIDE                   = 61,
    FSDB_DT_VHDL_WIDTH                  = 62, 

    FSDB_DT_VHDL_FILE_OPEN_KIND         = 63,
    FSDB_DT_VHDL_FILE_OPEN_STATUS       = 64,

    /* VHDL Xprop support */
    FSDB_DT_VHDL_BOOLEAN_X_VALUE	= 65,
    FSDB_DT_VHDL_BIT_X_VALUE	        = 66,
    FSDB_DT_VHDL_BIT_VECTOR_X_VALUE     = 67,
    FSDB_DT_VHDL_INTEGER_X_VALUE	= 68,

    /* For AMS data type [224~255] reverse to AMS data type*/
    /*
    ** AMS unify dumping
    */
    FSDB_DT_NANOSIM_LOGIC                 = 248,
    FSDB_DT_NANOSIM_VOLTAGE               = 249,  
    FSDB_DT_NANOSIM_INSTANTANEOUS_CURRENT = 250,
    FSDB_DT_NANOSIM_AVERAGE_RMS_CURRENT   = 251,
    FSDB_DT_NANOSIM_DI_DT                 = 252,
    FSDB_DT_NANOSIM_MATHEMATICS           = 253,
    FSDB_DT_NANOSIM_POWER                 = 254,

    FSDB_DT_AMS_WREAL                   = 255,
    
    /* System Verilog data types */
    FSDB_DT_SV_LOGIC                    = 256, FSDB_DT_SV_MIN = 256,
    FSDB_DT_SV_REG                      = 257,
    FSDB_DT_SV_BIT                      = 258,
    FSDB_DT_SV_LONG_INT                 = 259,
    FSDB_DT_SV_LONG_UINT                = 260,
    FSDB_DT_SV_INT                      = 261,
    FSDB_DT_SV_UINT                     = 262,
    FSDB_DT_SV_INTEGER                  = 263,
    FSDB_DT_SV_UINTEGER                 = 264,
    FSDB_DT_SV_SHORT_INT                = 265,
    FSDB_DT_SV_SHORT_UINT               = 266,
    FSDB_DT_SV_BYTE_INT                 = 267,
    FSDB_DT_SV_BYTE_UINT                = 268,
    FSDB_DT_SV_REAL                     = 269,
    FSDB_DT_SV_SHORT_REAL               = 270,
    FSDB_DT_SV_TIME                     = 271,
    FSDB_DT_SV_REFERENCE                = 272, /* internal use only */
    FSDB_DT_SV_STRING                   = 273,
    FSDB_DT_SV_CHAR                     = 274, /* removed at IEEE LRM */
    FSDB_DT_SV_EVENT                    = 275,
    FSDB_DT_SV_MDA                      = 276, FSDB_DT_SV_MAX = 276, /* internal use only */
    FSDB_DT_SV_PACKED_STRUCT            = 277, /* internal use only */
    FSDB_DT_SV_PACKED_ARRAY             = 278, /* internal use only */
    FSDB_DT_SV_PACKED_UNION             = 279, /* internal use only */
    
    FSDB_DT_SV_CLASS_ATTR               = 285, /* internal use only */
    
    FSDB_DT_SC_PROC_DYNA_SLIST          = 300, FSDB_DT_SC_MIN = 300, /* internal use only */
    FSDB_DT_SC_EVENT_DYNA_SLIST         = 301, FSDB_DT_SC_MAX = 301, /* internal use only */

    FSDB_DT_EVENT_STREAM                = 302, FSDB_DT_EVENT_MIN = 302,
    FSDB_DT_EVENT_RELATION              = 303,
    FSDB_DT_EVENT_INDEX                 = 304, FSDB_DT_EVENT_MAX = 304,

    FSDB_DT_FRD_VAR                     = 312,  /* internal use only */
    FSDB_DT_FIELD_TYPE                  = 313,  /* internal use only */
    FSDB_DT_COUNTER_INTEGER             = 314,  /* internal use only */
    FSDB_DT_COUNTER_WIRE                = 315   /* internal use only */

};
typedef enum fsdbDataType               fsdbDataType;
#define FSDB_DT_MAX_IDCODE              65535
#define FSDB_DT_MAX_RESERVED_IDCODE     511     /* 0 ~ 511 are reserved for fsdb */
#define FSDB_INVALID_DATA_TYPE_IDCODE   (-1)
/*
** type handle of predefined data type(Pedefined Data Type Handle)
**
** FSDB_DT_HDL_BYTE, FSDB_DT_HDL_SHORT, 
** FSDB_DT_HDL_INT, FSDB_DT_HDL_INT_H_N_L,
** FSDB_DT_HDL_LONG_INT, FSDB_DT_HDL_FLOAT and
** FSDB_DT_HDL_DOUBLE are added from fsdb v2.0
*/
#define FSDB_DT_HDL_BYTE                ((void*)FSDB_DT_BYTE)
#define FSDB_DT_HDL_SHORT               ((void*)FSDB_DT_SHORT)
#define FSDB_DT_HDL_INT                 ((void*)FSDB_DT_INT)
#define FSDB_DT_HDL_FLOAT               ((void*)FSDB_DT_FLOAT)
#define FSDB_DT_HDL_HL_INT              ((void*)FSDB_DT_HL_INT)
#define FSDB_DT_HDL_LONG                ((void*)FSDB_DT_LONG)
#define FSDB_DT_HDL_DOUBLE              ((void*)FSDB_DT_DOUBLE)

#define FSDB_DT_HDL_VERILOG_STANDARD    ((void*)FSDB_DT_VERILOG_STANDARD)
#define FSDB_DT_HDL_VERILOG_REAL        ((void*)FSDB_DT_VERILOG_REAL)
#define FSDB_DT_HDL_VERILOG_INTEGER     ((void*)FSDB_DT_VERILOG_INTEGER)

#define FSDB_DT_HDL_EPIC_DIGITAL        ((void*)FSDB_DT_EPIC_DIGITAL)
#define FSDB_DT_HDL_EPIC_ANALOG         ((void*)FSDB_DT_EPIC_ANALOG)
#define FSDB_DT_HDL_HSPICE              ((void*)FSDB_DT_HSPICE)
#define FSDB_DT_HDL_XP                  ((void*)FSDB_DT_XP)
#define FSDB_DT_HDL_RAWFILE             ((void*)FSDB_DT_RAWFILE)
#define FSDB_DT_HDL_WFM                 ((void*)FSDB_DT_WFM)

/* VHDL Xprop support */
#define FSDB_DT_HDL_VHDL_BOOLEAN_X_VALUE        ((void*)FSDB_DT_VHDL_BOOLEAN_X_VALUE)
#define FSDB_DT_HDL_VHDL_BIT_X_VALUE            ((void*)FSDB_DT_VHDL_BIT_X_VALUE)
#define FSDB_DT_HDL_VHDL_BIT_VECTOR_X_VALUE     ((void*)FSDB_DT_VHDL_BIT_VECTOR_X_VALUE)
#define FSDB_DT_HDL_VHDL_INTEGER_X_VALUE        ((void*)FSDB_DT_VHDL_INTEGER_X_VALUE)

#define FSDB_DT_HDL_VHDL_BOOLEAN                ((void*)FSDB_DT_VHDL_BOOLEAN)
#define FSDB_DT_HDL_VHDL_BIT                    ((void*)FSDB_DT_VHDL_BIT)
#define FSDB_DT_HDL_VHDL_BIT_VECTOR             ((void*)FSDB_DT_VHDL_BIT_VECTOR)
#define FSDB_DT_HDL_VHDL_STD_ULOGIC             ((void*)FSDB_DT_VHDL_STD_ULOGIC)
#define FSDB_DT_HDL_VHDL_STD_ULOGIC_VECTOR      ((void*)FSDB_DT_VHDL_STD_ULOGIC_VECTOR)
#define FSDB_DT_HDL_VHDL_STD_LOGIC              ((void*)FSDB_DT_VHDL_STD_LOGIC)
#define FSDB_DT_HDL_VHDL_STD_LOGIC_VECTOR       ((void*)FSDB_DT_VHDL_STD_LOGIC_VECTOR)
#define FSDB_DT_HDL_VHDL_UNSIGNED               ((void*)FSDB_DT_VHDL_UNSIGNED)
#define FSDB_DT_HDL_VHDL_SIGNED                 ((void*)FSDB_DT_VHDL_SIGNED)
#define FSDB_DT_HDL_VHDL_INTEGER                ((void*)FSDB_DT_VHDL_INTEGER)
#define FSDB_DT_HDL_VHDL_REAL                   ((void*)FSDB_DT_VHDL_REAL)
#define FSDB_DT_HDL_VHDL_NATURAL                ((void*)FSDB_DT_VHDL_NATURAL)
#define FSDB_DT_HDL_VHDL_POSITIVE               ((void*)FSDB_DT_VHDL_POSITIVE)
#define FSDB_DT_HDL_VHDL_TIME                   ((void*)FSDB_DT_VHDL_TIME)
#define FSDB_DT_HDL_VHDL_CHARACTER              ((void*)FSDB_DT_VHDL_CHARACTER)
#define FSDB_DT_HDL_VHDL_STRING                 ((void*)FSDB_DT_VHDL_STRING)

#define FSDB_DT_HDL_VHDL_STDLOGIC_1D            ((void*)FSDB_DT_VHDL_STDLOGIC_1D)
#define FSDB_DT_HDL_VHDL_STDLOGIC_TABLE         ((void*)FSDB_DT_VHDL_STDLOGIC_TABLE)
#define FSDB_DT_HDL_VHDL_LOGIC_X01_TABLE        ((void*)FSDB_DT_VHDL_LOGIC_X01_TABLE)
#define FSDB_DT_HDL_VHDL_LOGIC_X01Z_TABLE       ((void*)FSDB_DT_VHDL_LOGIC_X01Z_TABLE)
#define FSDB_DT_HDL_VHDL_LOGIC_UX01_TABLE       ((void*)FSDB_DT_VHDL_LOGIC_UX01_TABLE)

#define FSDB_DT_HDL_VHDL_X01            ((void*)FSDB_DT_VHDL_X01)
#define FSDB_DT_HDL_VHDL_X01Z           ((void*)FSDB_DT_VHDL_X01Z)
#define FSDB_DT_HDL_VHDL_UX01           ((void*)FSDB_DT_VHDL_UX01)
#define FSDB_DT_HDL_VHDL_UX01Z          ((void*)FSDB_DT_VHDL_UX01Z)

#define FSDB_DT_HDL_VHDL_SEVERITY_LEVEL ((void*)FSDB_DT_VHDL_SEVERITY_LEVEL)
#define FSDB_DT_HDL_VHDL_DELAY_LENGTH   ((void*)FSDB_DT_VHDL_DELAY_LENGTH)

#define FSDB_DT_HDL_VHDL_LINE           ((void*)FSDB_DT_VHDL_LINE)
#define FSDB_DT_HDL_VHDL_TEXT           ((void*)FSDB_DT_VHDL_TEXT)
#define FSDB_DT_HDL_VHDL_SIDE           ((void*)FSDB_DT_VHDL_SIDE)
#define FSDB_DT_HDL_VHDL_WIDTH          ((void*)FSDB_DT_VHDL_WIDTH)

#define FSDB_DT_HDL_VHDL_FILE_OPEN_KIND         ((void*)FSDB_DT_VHDL_FILE_OPEN_KIND)
#define FSDB_DT_HDL_VHDL_FILE_OPEN_STATUS       ((void*)FSDB_DT_VHDL_FILE_OPEN_STATUS) 


/*
** reserve these data type handles to keep backward compatibility
*/
#define FSDB_DT_HANDLE_VERILOG_STANDARD         ((void*)FSDB_DT_VERILOG_STANDARD)
#define FSDB_DT_HANDLE_VERILOG_REAL             ((void*)FSDB_DT_VERILOG_REAL)
#define FSDB_DT_HANDLE_VERILOG_INTEGER          ((void*)FSDB_DT_VERILOG_INTEGER)

#define FSDB_DT_HANDLE_EPIC_DIGITAL             ((void*)FSDB_DT_EPIC_DIGITAL)
#define FSDB_DT_HANDLE_EPIC_ANALOG              ((void*)FSDB_DT_EPIC_ANALOG)
#define FSDB_DT_HANDLE_HSPICE                   ((void*)FSDB_DT_HSPICE)
#define FSDB_DT_HANDLE_XP                       ((void*)FSDB_DT_XP)
#define FSDB_DT_HANDLE_RAWFILE                  ((void*)FSDB_DT_RAWFILE)
#define FSDB_DT_HANDLE_WFM                      ((void*)FSDB_DT_WFM)

#define FSDB_DT_HANDLE_VHDL_BOOLEAN             ((void*)FSDB_DT_VHDL_BOOLEAN)
#define FSDB_DT_HANDLE_VHDL_BIT                 ((void*)FSDB_DT_VHDL_BIT)
#define FSDB_DT_HANDLE_VHDL_BIT_VECTOR          ((void*)FSDB_DT_VHDL_BIT_VECTOR)
#define FSDB_DT_HANDLE_VHDL_STD_ULOGIC          ((void*)FSDB_DT_VHDL_STD_ULOGIC)
#define FSDB_DT_HANDLE_VHDL_STD_ULOGIC_VECTOR   ((void*)FSDB_DT_VHDL_STD_ULOGIC_VECTOR)
#define FSDB_DT_HANDLE_VHDL_STD_LOGIC           ((void*)FSDB_DT_VHDL_STD_LOGIC)
#define FSDB_DT_HANDLE_VHDL_STD_LOGIC_VECTOR    ((void*)FSDB_DT_VHDL_STD_LOGIC_VECTOR)
#define FSDB_DT_HANDLE_VHDL_UNSIGNED            ((void*)FSDB_DT_VHDL_UNSIGNED)
#define FSDB_DT_HANDLE_VHDL_SIGNED              ((void*)FSDB_DT_VHDL_SIGNED)
#define FSDB_DT_HANDLE_VHDL_INTEGER             ((void*)FSDB_DT_VHDL_INTEGER)
#define FSDB_DT_HANDLE_VHDL_REAL                ((void*)FSDB_DT_VHDL_REAL)
#define FSDB_DT_HANDLE_VHDL_NATURAL             ((void*)FSDB_DT_VHDL_NATURAL)
#define FSDB_DT_HANDLE_VHDL_POSITIVE            ((void*)FSDB_DT_VHDL_POSITIVE)
#define FSDB_DT_HANDLE_VHDL_TIME                ((void*)FSDB_DT_VHDL_TIME)
#define FSDB_DT_HANDLE_VHDL_CHARACTER           ((void*)FSDB_DT_VHDL_CHARACTER)
#define FSDB_DT_HANDLE_VHDL_STRING              ((void*)FSDB_DT_VHDL_STRING)

#define FSDB_DT_HANDLE_VHDL_STDLOGIC_1D         ((void*)FSDB_DT_VHDL_STDLOGIC_1D)
#define FSDB_DT_HANDLE_VHDL_STDLOGIC_TABLE      ((void*)FSDB_DT_VHDL_STDLOGIC_TABLE)
#define FSDB_DT_HANDLE_VHDL_LOGIC_X01_TABLE     ((void*)FSDB_DT_VHDL_LOGIC_X01_TABLE)
#define FSDB_DT_HANDLE_VHDL_LOGIC_X01Z_TABLE    ((void*)FSDB_DT_VHDL_LOGIC_X01Z_TABLE)
#define FSDB_DT_HANDLE_VHDL_LOGIC_UX01_TABLE    ((void*)FSDB_DT_VHDL_LOGIC_UX01_TABLE)

#define FSDB_DT_HANDLE_VHDL_X01                 ((void*)FSDB_DT_VHDL_X01)
#define FSDB_DT_HANDLE_VHDL_X01Z                ((void*)FSDB_DT_VHDL_X01Z)
#define FSDB_DT_HANDLE_VHDL_UX01                ((void*)FSDB_DT_VHDL_UX01)
#define FSDB_DT_HANDLE_VHDL_UX01Z               ((void*)FSDB_DT_VHDL_UX01Z)

#define FSDB_DT_HANDLE_VHDL_SEVERITY_LEVEL      ((void*)FSDB_DT_VHDL_SEVERITY_LEVEL)
#define FSDB_DT_HANDLE_VHDL_DELAY_LENGTH        ((void*)FSDB_DT_VHDL_DELAY_LENGTH)

#define FSDB_DT_HANDLE_VHDL_LINE                ((void*)FSDB_DT_VHDL_LINE)
#define FSDB_DT_HANDLE_VHDL_TEXT                ((void*)FSDB_DT_VHDL_TEXT)
#define FSDB_DT_HANDLE_VHDL_SIDE                ((void*)FSDB_DT_VHDL_SIDE)
#define FSDB_DT_HANDLE_VHDL_WIDTH               ((void*)FSDB_DT_VHDL_WIDTH)

#define FSDB_DT_HANDLE_VHDL_FILE_OPEN_KIND      ((void*)FSDB_DT_VHDL_FILE_OPEN_KIND)
#define FSDB_DT_HANDLE_VHDL_FILE_OPEN_STATUS    ((void*)FSDB_DT_VHDL_FILE_OPEN_STATUS) 

/* AMS data types */
#define FSDB_DT_HANDLE_AMS_WREAL                ((void*)FSDB_DT_AMS_WREAL)

#define FSDB_DT_HANDLE_NANOSIM_LOGIC            ((void*)FSDB_DT_NANOSIM_LOGIC)
#define FSDB_DT_HANDLE_NANOSIM_VOLTAGE          ((void*)FSDB_DT_NANOSIM_VOLTAGE)
#define FSDB_DT_HANDLE_NANOSIM_INSTANTANEOUS_CURRENT              ((void*)FSDB_DT_NANOSIM_INSTANTANEOUS_CURRENT)
#define FSDB_DT_HANDLE_NANOSIM_AVERAGE_RMS_CURRENT                ((void*)FSDB_DT_NANOSIM_AVERAGE_RMS_CURRENT)
#define FSDB_DT_HANDLE_NANOSIM_DI_DT            ((void*)FSDB_DT_NANOSIM_DI_DT)
#define FSDB_DT_HANDLE_NANOSIM_MATHEMATICS      ((void*)FSDB_DT_NANOSIM_MATHEMATICS)
#define FSDB_DT_HANDLE_NANOSIM_POWER            ((void*)FSDB_DT_NANOSIM_POWER)

/* System Verilog data types */
#define FSDB_DT_HANDLE_SV_LOGIC                 ((void*)FSDB_DT_SV_LOGIC)
#define FSDB_DT_HANDLE_SV_REG                   ((void*)FSDB_DT_SV_REG)
#define FSDB_DT_HANDLE_SV_BIT                   ((void*)FSDB_DT_SV_BIT)
#define FSDB_DT_HANDLE_SV_LONG_INT              ((void*)FSDB_DT_SV_LONG_INT)
#define FSDB_DT_HANDLE_SV_LONG_UINT             ((void*)FSDB_DT_SV_LONG_UINT)
#define FSDB_DT_HANDLE_SV_INT                   ((void*)FSDB_DT_SV_INT)
#define FSDB_DT_HANDLE_SV_UINT                  ((void*)FSDB_DT_SV_UINT)
#define FSDB_DT_HANDLE_SV_INTEGER               ((void*)FSDB_DT_SV_INTEGER)
#define FSDB_DT_HANDLE_SV_UINTEGER              ((void*)FSDB_DT_SV_UINTEGER)
#define FSDB_DT_HANDLE_SV_SHORT_INT             ((void*)FSDB_DT_SV_SHORT_INT)
#define FSDB_DT_HANDLE_SV_SHORT_UINT            ((void*)FSDB_DT_SV_SHORT_UINT)
#define FSDB_DT_HANDLE_SV_BYTE_INT              ((void*)FSDB_DT_SV_BYTE_INT)
#define FSDB_DT_HANDLE_SV_BYTE_UINT             ((void*)FSDB_DT_SV_BYTE_UINT)
#define FSDB_DT_HANDLE_SV_REAL                  ((void*)FSDB_DT_SV_REAL)
#define FSDB_DT_HANDLE_SV_SHORT_REAL            ((void*)FSDB_DT_SV_SHORT_REAL)
#define FSDB_DT_HANDLE_SV_TIME                  ((void*)FSDB_DT_SV_TIME)
#define FSDB_DT_HANDLE_SV_REFERENCE             ((void*)FSDB_DT_SV_REFERENCE)
#define FSDB_DT_HANDLE_SV_STRING                ((void*)FSDB_DT_SV_STRING)
#define FSDB_DT_HANDLE_SV_CHAR                  ((void*)FSDB_DT_SV_CHAR)
#define FSDB_DT_HANDLE_SV_EVENT                 ((void*)FSDB_DT_SV_EVENT)
#define FSDB_DT_HANDLE_SV_MDA                   ((void*)FSDB_DT_SV_MDA)
#define FSDB_DT_HANDLE_SV_CLASS_ATTR            ((void*)FSDB_DT_SV_CLASS_ATTR)

#define FSDB_DT_HANDLE_SC_PROC_DYNA_SLIST       ((void*)FSDB_DT_SC_PROC_DYNA_SLIST)
#define FSDB_DT_HANDLE_SC_EVENT_DYNA_SLIST      ((void*)FSDB_DT_SC_EVENT_DYNA_SLIST)

#define FSDB_DT_HANDLE_EVENT_STREAM             ((void*)FSDB_DT_EVENT_STREAM)
#define FSDB_DT_HANDLE_EVENT_RELATION           ((void*)FSDB_DT_EVENT_RELATION)
#define FSDB_DT_HANDLE_EVENT_INDEX              ((void*)FSDB_DT_EVENT_INDEX)

#define FSDB_DT_HANDLE_FRD_VAR                  ((void*)FSDB_DT_FRD_VAR)
#define FSDB_DT_HANDLE_FIELD_TYPE               ((void*)FSDB_DT_FIELD_TYPE)

#define FSDB_DT_HANDLE_COUNTER_INTEGER          ((void*)FSDB_DT_COUNTER_INTEGER)
#define FSDB_DT_HANDLE_COUNTER_WIRE             ((void*)FSDB_DT_COUNTER_WIRE)
/*
** predefined bit type
*/
enum fsdbBitType {
    /*
    ** fsdb Writer embeds a special signal called "dump off signal", it stores
    ** the dump off time range. With this mechanism and the dump off value(255), we
    ** can tell the current value is a real value or dump off value.
    */
    FSDB_BT_VCD_0  = 0,
    FSDB_BT_VCD_1  = 1,
    FSDB_BT_VCD_X  = 2,
    FSDB_BT_VCD_Z  = 3,
    FSDB_BT_VCD_NC = 4, /* for internal use only */
    FSDB_BT_VCD_NV = 5, /* for internal use only */

    FSDB_BT_EVCD_L      = 0,    /* for extended vcd format */
    FSDB_BT_EVCD_l      = 1,
    FSDB_BT_EVCD_H      = 2,
    FSDB_BT_EVCD_h      = 3,
    FSDB_BT_EVCD_X      = 4,
    FSDB_BT_EVCD_x      = 5,
    FSDB_BT_EVCD_T      = 6,
    FSDB_BT_EVCD_D      = 7,
    FSDB_BT_EVCD_d      = 8,
    FSDB_BT_EVCD_U      = 9,
    FSDB_BT_EVCD_u      = 10,
    FSDB_BT_EVCD_N      = 11,
    FSDB_BT_EVCD_n      = 12,
    FSDB_BT_EVCD_Z      = 13,
    FSDB_BT_EVCD_QSTN   = 14,
    FSDB_BT_EVCD_0      = 15,
    FSDB_BT_EVCD_1      = 16,
    FSDB_BT_EVCD_A      = 17,
    FSDB_BT_EVCD_a      = 18,
    FSDB_BT_EVCD_B      = 19,
    FSDB_BT_EVCD_b      = 20,
    FSDB_BT_EVCD_C      = 21,
    FSDB_BT_EVCD_c      = 22,
    FSDB_BT_EVCD_F      = 23,
    FSDB_BT_EVCD_f      = 24,
    FSDB_BT_EVCD_NC     = 25, 
    FSDB_BT_EVCD_NV     = 26, 

    FSDB_BT_VHDL_BOOLEAN_FALSE = 0,
    FSDB_BT_VHDL_BOOLEAN_TRUE  = 1,

    FSDB_BT_VHDL_BIT_0  = 0,
    FSDB_BT_VHDL_BIT_1  = 1,

    FSDB_BT_VHDL_SEVERITY_LEVEL_NOTE    = 0,
    FSDB_BT_VHDL_SEVERITY_LEVEL_WARNING = 1,
    FSDB_BT_VHDL_SEVERITY_LEVEL_ERROR   = 2,
    FSDB_BT_VHDL_SEVERITY_LEVEL_FAILURE = 3,

    FSDB_BT_VHDL_STD_ULOGIC_U           = 0,
    FSDB_BT_VHDL_STD_ULOGIC_X           = 1,
    FSDB_BT_VHDL_STD_ULOGIC_0           = 2,
    FSDB_BT_VHDL_STD_ULOGIC_1           = 3,
    FSDB_BT_VHDL_STD_ULOGIC_Z           = 4,
    FSDB_BT_VHDL_STD_ULOGIC_W           = 5,
    FSDB_BT_VHDL_STD_ULOGIC_L           = 6,
    FSDB_BT_VHDL_STD_ULOGIC_H           = 7,
    FSDB_BT_VHDL_STD_ULOGIC_DASH        = 8,

    FSDB_BT_VHDL_STD_LOGIC_U            = 0,
    FSDB_BT_VHDL_STD_LOGIC_X            = 1,
    FSDB_BT_VHDL_STD_LOGIC_0            = 2,
    FSDB_BT_VHDL_STD_LOGIC_1            = 3,
    FSDB_BT_VHDL_STD_LOGIC_Z            = 4,
    FSDB_BT_VHDL_STD_LOGIC_W            = 5,
    FSDB_BT_VHDL_STD_LOGIC_L            = 6,
    FSDB_BT_VHDL_STD_LOGIC_H            = 7,
    FSDB_BT_VHDL_STD_LOGIC_DASH         = 8,

    FSDB_BT_VHDL_FILE_OPEN_KIND_READ_MODE       = 0,
    FSDB_BT_VHDL_FILE_OPEN_KIND_WRITE_MODE      = 1,
    FSDB_BT_VHDL_FILE_OPEN_KIND_APPEND_MODE     = 2,

    FSDB_BT_VHDL_FILE_OPEN_STATUS_OPEN_OK       = 0,
    FSDB_BT_VHDL_FILE_OPEN_STATUS_STATUS_ERROR  = 1,
    FSDB_BT_VHDL_FILE_OPEN_STATUS_NAME_ERROR    = 2,
    FSDB_BT_VHDL_FILE_OPEN_STATUS_MODE_ERROR    = 3,

    FSDB_BT_EPIC_0                      = 0, 
    FSDB_BT_EPIC_1                      = 1, 
    FSDB_BT_EPIC_U                      = 2, 
    FSDB_BT_EPIC_L                      = 3, 
    FSDB_BT_EPIC_H                      = 4, 
    FSDB_BT_EPIC_Z                      = 5, 
    FSDB_BT_EPIC_N                      = 6, 
    FSDB_BT_EPIC_T                      = 7, 
    FSDB_BT_EPIC_Y                      = 8, 
    FSDB_BT_EPIC_QSTN                   = 9,    /* it means "?" symbol */
    FSDB_BT_EPIC_X                      = 10, 

    FSDB_BT_MC_VCD_NC                   = 4,    /* Not Computed */
    FSDB_BT_MC_VHDL_STD_ULOGIC_NC       = 9,
    FSDB_BT_MC_VHDL_STD_LOGIC_NC        = 9,

    FSDB_BT_TRANS_LOGIC_0 = 0,
    FSDB_BT_TRANS_LOGIC_1 = 1,
    FSDB_BT_TRANS_LOGIC_X = 2,
    FSDB_BT_TRANS_LOGIC_Z = 3,

    FSDB_BT_TRANS_BOOL_FALSE = 0,
    FSDB_BT_TRANS_BOOL_TRUE  = 1
};
typedef enum fsdbBitType        fsdbBitType;
#define FSDB_MAX_VCD_BIT_TYPE   3
#define FSDB_NUM_VCD_BIT_TYPE   4

/*
**      Strength Type
*/
enum fsdbStrengthType {
    FSDB_ST_HIGHZ       = 0,
    FSDB_ST_SMALL       = 1,
    FSDB_ST_MEDIUM      = 2,
    FSDB_ST_WEAK        = 3,
    FSDB_ST_LARGE       = 4,
    FSDB_ST_PULL        = 5,
    FSDB_ST_STRONG      = 6,
    FSDB_ST_SUPPLY      = 7
};
typedef enum fsdbStrengthType   fsdbStrengthType;


/* 
** fsdb writer use 1B to store bytesPerBit information
*/
enum fsdbBytesPerBit {
    FSDB_BYTES_PER_BIT_1B       =   0,
    FSDB_BYTES_PER_BIT_2B       =   1,
    FSDB_BYTES_PER_BIT_4B       =   2,
    FSDB_BYTES_PER_BIT_8B       =   3,
    FSDB_BYTES_PER_BIT_UNKNOWN  = 255
};
typedef enum fsdbBytesPerBit    fsdbBytesPerBit;


struct fsdbVCRec {
    ushort_T    bitSize;
    ushort_T    padShort;
    char        *bits;
    union       _vc {
        int     idcode;
        void    *handle;
    } u;
};
typedef struct fsdbVCRec        fsdbVCRec;
 
struct fsdbExeRec {
    byte_T      type;
    int         lineNo;
    str_T       fileName;
    str_T       fullScope;
    str_T       opt;
};
typedef struct fsdbExeRec       fsdbExeRec;

/* The table is for internal use*/
enum fsdbFieldPreDefineType {
        FSDB_FIELD_MIN           = 0, 
        /* support for SV */
        FSDB_FIELD_SV_LOGIC      = 0, FSDB_FIELD_SV_MIN = 0, 
        FSDB_FIELD_SV_ULOGIC     = 1, 
        FSDB_FIELD_SV_REG        = 2,
        FSDB_FIELD_SV_UREG       = 3,
        FSDB_FIELD_SV_BIT        = 4, 
        FSDB_FIELD_SV_UBIT       = 5, 
        FSDB_FIELD_SV_LONG_INT   = 6, 
        FSDB_FIELD_SV_LONG_UINT  = 7, 
        FSDB_FIELD_SV_INT        = 8, 
        FSDB_FIELD_SV_UINT       = 9, 
        FSDB_FIELD_SV_INTEGER    = 10, 
        FSDB_FIELD_SV_UINTEGER   = 11, 
        FSDB_FIELD_SV_SHORT_INT  = 12, 
        FSDB_FIELD_SV_SHORT_UINT = 13, 
        FSDB_FIELD_SV_BYTE_INT   = 14,
        FSDB_FIELD_SV_BYTE_UINT  = 15, 
        FSDB_FIELD_SV_REAL       = 16, 
        FSDB_FIELD_SV_SHORT_REAL = 17, 
        FSDB_FIELD_SV_TIME       = 18, 
        FSDB_FIELD_SV_STRING     = 19, 
        FSDB_FIELD_SV_EVENT      = 20, 
        FSDB_FIELD_SV_REALTIME   = 21, FSDB_FIELD_SV_MAX = 21,
        
        /* support for C */
        FSDB_FIELD_C_LONG        = 22, FSDB_FIELD_C_MIN = 22,
        FSDB_FIELD_C_ULONG       = 23, 
        FSDB_FIELD_C_INT         = 24, 
        FSDB_FIELD_C_UINT        = 25, 
        FSDB_FIELD_C_SHORT       = 26, 
        FSDB_FIELD_C_USHORT      = 27, 
        FSDB_FIELD_C_CHAR        = 28, 
        FSDB_FIELD_C_UCHAR       = 29, 
        FSDB_FIELD_C_BOOL        = 30, 
        FSDB_FIELD_C_STRING      = 31, 
        FSDB_FIELD_C_DOUBLE      = 32, 
        FSDB_FIELD_C_FLOAT       = 33, 
        
        FSDB_FIELD_C_INT64       = 35,
        FSDB_FIELD_C_UINT64      = 36,
        FSDB_FIELD_C_LONGDOUBLE  = 37, 

        FSDB_FIELD_C_LONGLONG    = 38,
        FSDB_FIELD_C_ULONGLONG   = 39,
        FSDB_FIELD_C_POINTER32   = 40,
        FSDB_FIELD_C_POINTER64   = 41,
        FSDB_FIELD_C_STRUCT      = 42,
        FSDB_FIELD_C_UNION       = 43,
        FSDB_FIELD_C_ARRAY       = 44, FSDB_FIELD_C_MAX  = 44,

        /* Tag*/
        FSDB_FIELD_TAG           = 34,

        FSDB_FIELD_SC_LOGIC      = 45, FSDB_FIELD_SC_MIN  = 45,

        FSDB_FIELD_SC_INT_BASE   = 46,
        FSDB_FIELD_SC_UINT_BASE  = 47,
        FSDB_FIELD_SC_SIGNED     = 48,
        FSDB_FIELD_SC_UNSIGNED   = 49,
        FSDB_FIELD_SC_LV_BASE    = 50,
        FSDB_FIELD_SC_BV_BASE    = 51,

        FSDB_FIELD_SC_FIX        = 52,
        FSDB_FIELD_SC_UFIX       = 53,
        FSDB_FIELD_SC_FXVAL      = 54,

        FSDB_FIELD_SC_FIXFAST    = 55,
        FSDB_FIELD_SC_UFIXFAST   = 56,
        FSDB_FIELD_SC_FXVALFAST  = 57,

        FSDB_FIELD_SC_FIFO       = 58,
        FSDB_FIELD_SC_TLM_FIFO   = 59, FSDB_FIELD_SC_MAX      = 59,

        /* Verilog*/

        FSDB_FIELD_VERILOG_REG   = 60, FSDB_FIELD_VERILOG_MIN  = 60,
        FSDB_FIELD_VERILOG_REG2  = 61,
        FSDB_FIELD_VERILOG_STRUCT= 62,
        FSDB_FIELD_VERILOG_UNION = 63, FSDB_FIELD_VERILOG_MAX = 63,

        FSDB_FIELD_AMS_WREAL     = 64,

        /* extend SV*/
        FSDB_FIELD_SV_VOID       = 65,
        
        /* VHDL */ 
        FSDB_FIELD_VHDL_BIT      = 66, FSDB_FIELD_VHDL_MIN  = 66,
        FSDB_FIELD_VHDL_BOOLEAN  = 67,
        FSDB_FIELD_VHDL_STDLOGIC = 68, 
        FSDB_FIELD_VHDL_INTEGER  = 69,
        FSDB_FIELD_VHDL_TIME     = 70,
        FSDB_FIELD_VHDL_REAL     = 71,
        FSDB_FIELD_VHDL_RECORD   = 72, FSDB_FIELD_VHDL_MAX  = 72,

        FSDB_FIELD_MAX          = 72 
};
typedef enum fsdbFieldPreDefineType fsdbFieldPreDefineType;
typedef enum fsdbFieldPreDefineType fsdbFieldBlkType ;

#define FSDB_INVALID_FIELD_TYPE   (-1)
#define FSDB_INVALID_FIELD_IDCODE (-1)
typedef int                 fsdbFieldIdcode;
typedef int                 fsdbFieldType;
/* 
** -------------------- fsdb v2.0 -------------------- 
*/
#define FSDB_MAX_SCOPE_DEPTH            512

#define FSDB_LEN_SCOPE_SEPARATOR        4
#define FSDB_LEN_SCALE_UNIT             20      
#define FSDB_LEN_SIMULATION_DATE        32
#define FSDB_LEN_SIMULATOR_VERSION      64

union fsdbLUOff {
    byte_T array[8];
    double d;
};
typedef union fsdbLUOff         fsdbLUOff;

struct fsdbLTag {
    uint_T      L;
};
typedef struct fsdbLTag         fsdbLTag;

struct fsdbHLTag {
    uint_T      H;
    uint_T      L;
};
typedef struct fsdbHLTag        fsdbHLTag;

union fsdbXTag {
    fsdbLTag    ltag;
    fsdbHLTag   hltag;
    float       ftag;
    double      dtag;
};
typedef union fsdbXTag          fsdbXTag;
typedef union fsdbXTag          fsdbUTag;

struct fsdbXTagRange {
    fsdbXTag begin;
    fsdbXTag end;
};
typedef struct fsdbXTagRange fsdbXTagRange;

enum fsdbXTagType {
    FSDB_XTAG_TYPE_L            = 0,
    FSDB_XTAG_TYPE_HL           = 1,
    FSDB_XTAG_TYPE_FLOAT        = 2,
    FSDB_XTAG_TYPE_DOUBLE       = 3
};
typedef enum fsdbXTagType       fsdbXTagType;


/*
** fsdbDumpOffRange was added from Debussy v5.2
*/
struct fsdbDumpOffRange {
    fsdbXTag begin;
    fsdbXTag end;
};
typedef struct fsdbDumpOffRange fsdbDumpOffRange;


/*
** fsdb Tree Callback Type(for fsdb Reader) 
*/
enum fsdbTreeCBType {
    /*
    ** scope and var
    */ 
    FSDB_TREE_CBT_BEGIN_TREE    = 0, 
    FSDB_TREE_CBT_SCOPE         = 1,
    FSDB_TREE_CBT_VAR           = 2,
    FSDB_TREE_CBT_UPSCOPE       = 3,
    FSDB_TREE_CBT_END_TREE      = 4,
    FSDB_TREE_CBT_END_ALL_TREE  = 5, /* there is no more tree */


    /*
    ** data type
    */ 
    FSDB_TREE_CBT_DT_ENUM       = 6,
    FSDB_TREE_CBT_DT_INT        = 7,
    FSDB_TREE_CBT_DT_FLOAT      = 8,
    FSDB_TREE_CBT_DT_FLOATING   = 8,
    FSDB_TREE_CBT_DT_PHYSICAL_BEGIN = 9,
    FSDB_TREE_CBT_DT_PHY_BEGIN      = 9,
    FSDB_TREE_CBT_DT_PHYSICAL_SECONDARY_UNIT = 10,
    FSDB_TREE_CBT_DT_PHY_SEC_UNIT            = 10,
    FSDB_TREE_CBT_DT_PHYSICAL_END = 11,
    FSDB_TREE_CBT_DT_PHY_END      = 11,
    FSDB_TREE_CBT_DT_INT_H_N_L    = 12,


    /*
    ** signal grouping 
    */ 
    FSDB_TREE_CBT_ARRAY_BEGIN       = 13,
    FSDB_TREE_CBT_ARRAY_END         = 14,
    FSDB_TREE_CBT_RECORD_BEGIN      = 15,
    FSDB_TREE_CBT_RECORD_END        = 16,
 
    FSDB_TREE_CBT_BUS_BEGIN         = 17,
    FSDB_TREE_CBT_BUS_ELEMENT       = 18,
    FSDB_TREE_CBT_BUS_END           = 19,


    /*
    ** others
    */
    FSDB_TREE_CBT_FILE_TYPE         = 20,
    FSDB_TREE_CBT_SIMULATOR_VERSION = 21,
    FSDB_TREE_CBT_SIMULATION_DATE   = 22,
    FSDB_TREE_CBT_X_AXIS_SCALE      = 23,

    /*
    ** signal grouping
    */
    FSDB_TREE_CBT_STRUCT_BEGIN      = 24,
    FSDB_TREE_CBT_STRUCT_END        = 25,

    /*
    ** transaction attributes
    */
    FSDB_TREE_CBT_DT_ATTR_UNKNOWN_TYPE  = 26,
    FSDB_TREE_CBT_DT_ATTR_USER_DEFINED  = 27,
    FSDB_TREE_CBT_DT_ATTR_LOGIC         = 28,
    FSDB_TREE_CBT_DT_ATTR_BOOL          = 29,
    FSDB_TREE_CBT_DT_ATTR_CHAR          = 30,
    FSDB_TREE_CBT_DT_ATTR_STRING        = 31,
    FSDB_TREE_CBT_DT_ATTR_INT32         = 32,
    FSDB_TREE_CBT_DT_ATTR_INT64         = 33,
    FSDB_TREE_CBT_DT_ATTR_UINT32        = 34,
    FSDB_TREE_CBT_DT_ATTR_UINT64        = 35,
    FSDB_TREE_CBT_DT_ATTR_FLOAT         = 36,
    FSDB_TREE_CBT_DT_ATTR_DOUBLE        = 37,
    FSDB_TREE_CBT_DT_ATTR_ENUM          = 38,
    FSDB_TREE_CBT_DT_ATTR_REFERENCE     = 39,
    FSDB_TREE_CBT_DT_ATTR_EVENT         = 40,
    FSDB_TREE_CBT_DT_ATTR_SV_ENUM       = 41,
    FSDB_TREE_CBT_DT_ATTR_MDA           = 42,

    /*
    ** data type
    */
    FSDB_TREE_CBT_DT_ENUM2              = 51,
    FSDB_TREE_CBT_DT_ENUM3              = 52,
    FSDB_TREE_CBT_DT_CLASS              = 53,
    FSDB_TREE_CBT_DT_PACKED_STRUCT      = 54,
    FSDB_TREE_CBT_DT_PACKED_ARRAY       = 55,
    FSDB_TREE_CBT_DT_PACKED_UNION       = 56,
    
    /* System Verilog data types */
    FSDB_TREE_CBT_DT_ATTR_SV_LOGIC      = 64, FSDB_TREE_CBT_DT_ATTR_SV_MIN = 64,
    FSDB_TREE_CBT_DT_ATTR_SV_REG        = 65,
    FSDB_TREE_CBT_DT_ATTR_SV_BIT        = 66,
    FSDB_TREE_CBT_DT_ATTR_SV_LONG_INT   = 67,
    FSDB_TREE_CBT_DT_ATTR_SV_LONG_UINT  = 68,
    FSDB_TREE_CBT_DT_ATTR_SV_INT        = 69,
    FSDB_TREE_CBT_DT_ATTR_SV_UINT       = 70,
    FSDB_TREE_CBT_DT_ATTR_SV_INTEGER    = 71,
    FSDB_TREE_CBT_DT_ATTR_SV_UINTEGER   = 72,
    FSDB_TREE_CBT_DT_ATTR_SV_SHORT_INT  = 73,
    FSDB_TREE_CBT_DT_ATTR_SV_SHORT_UINT = 74,
    FSDB_TREE_CBT_DT_ATTR_SV_BYTE_INT   = 75,
    FSDB_TREE_CBT_DT_ATTR_SV_BYTE_UINT  = 76,
    FSDB_TREE_CBT_DT_ATTR_SV_REAL       = 77,
    FSDB_TREE_CBT_DT_ATTR_SV_SHORT_REAL = 78,
    FSDB_TREE_CBT_DT_ATTR_SV_TIME       = 79,
    FSDB_TREE_CBT_DT_ATTR_SV_REFERENCE  = 80,
    FSDB_TREE_CBT_DT_ATTR_SV_STRING     = 81,
    FSDB_TREE_CBT_DT_ATTR_SV_CHAR       = 82,
    FSDB_TREE_CBT_DT_ATTR_SV_EVENT      = 83,
    FSDB_TREE_CBT_DT_ATTR_SV_CLASS_ATTR = 84,
    FSDB_TREE_CBT_DT_ATTR_SV_MDA        = 85, FSDB_TREE_CBT_DT_ATTR_SV_MAX = 85,
    FSDB_TREE_CBT_DT_ATTR_RAW_STRING    = 86,
    FSDB_TREE_CBT_DT_ATTR_SV_PACKED_STRUCT_ATTR = 87,
    FSDB_TREE_CBT_DT_ATTR_SV_PACKED_ARRAY_ATTR  = 88,
    FSDB_TREE_CBT_DT_ATTR_SV_PACKED_UNION_ATTR  = 89,

    /* Function Recording DT CB  */
    FSDB_TREE_CBT_DT_PARTIAL_CLASS          = 96,
    FSDB_TREE_CBT_DT_CLASS_BEGIN            = 97,
    FSDB_TREE_CBT_DT_COMPLETE_CLASS_BEGIN   = 98,
    FSDB_TREE_CBT_DT_CLASS_MEMBER           = 99,       
    FSDB_TREE_CBT_DT_CLASS_END              = 100,
    FSDB_TREE_CBT_DT_STRUCT_BEGIN           = 101,
    FSDB_TREE_CBT_DT_STRUCT_FIELD           = 102,
    FSDB_TREE_CBT_DT_STRUCT_END             = 103,
    FSDB_TREE_CBT_DT_ARRAY                  = 104,
    FSDB_TREE_CBT_DT_UNION_BEGIN            = 105,
    FSDB_TREE_CBT_DT_UNION_FIELD            = 106,
    FSDB_TREE_CBT_DT_UNION_END              = 107,
    FSDB_TREE_CBT_DT_QUEUE                  = 108,
    FSDB_TREE_CBT_DT_ASSOCIATE_ARRAY        = 109,
    FSDB_TREE_CBT_DT_DYNAMIC_ARRAY          = 110,
    

    /* Function Recording Var CB  */

    FSDB_TREE_CBT_SV_VAR                = 113,
    FSDB_TREE_CBT_SV_VAR_THIS           = 114,
    FSDB_TREE_CBT_METHOD_VAR            = 115,
    FSDB_TREE_CBT_METHOD_END            = 116,
    FSDB_TREE_CBT_METHOD_THIS_VAR       = 117,
    FSDB_TREE_CBT_METHOD_RETURN_VAR     = 118,
    FSDB_TREE_CBT_METHOD_ARGUMENT_VAR   = 119,
    FSDB_TREE_CBT_METHOD_LOCAL_VAR      = 120,

    /*
    ** event tree scope and var
    */
    FSDB_TREE_CBT_BEGIN_EVENT_TREE    = 121,
    FSDB_TREE_CBT_EVENT_SCOPE         = 122,
    FSDB_TREE_CBT_EVENT_VAR           = 123,
    FSDB_TREE_CBT_EVENT_UPSCOPE       = 124,
    FSDB_TREE_CBT_END_EVENT_TREE      = 125,

    /* Field Type CB */
    FSDB_TREE_CBT_FIELD                      = 130,
    FSDB_TREE_CBT_FIELD_ENUM                 = 131,
    FSDB_TREE_CBT_FIELD_PACKED_ARRAY         = 132,
    FSDB_TREE_CBT_FIELD_SC_BASE              = 133,
    FSDB_TREE_CBT_FIELD_STRUCT               = 134,
    FSDB_TREE_CBT_FIELD_FIX                  = 135,
    FSDB_TREE_CBT_FIELD_ARRAY                = 136,
    FSDB_TREE_CBT_FIELD_UNION                = 137,
    FSDB_TREE_CBT_FIELD_FIFO                 = 138,
    FSDB_TREE_CBT_FIELD_TLM_FIFO             = 139,
    FSDB_TREE_CBT_FIELD_SC_ENUM              = 140,
    FSDB_TREE_CBT_FIELD_UNPACKED_ARRAY       = 141,
    FSDB_TREE_CBT_FIELD_VLOG_UNPACKED_ARRAY  = 141,
    FSDB_TREE_CBT_FIELD_VERILOG_UNION        = 142,
    FSDB_TREE_CBT_FIELD_UNPACKED_STRUCT      = 143,
    FSDB_TREE_CBT_FIELD_VLOG_UNPACKED_STRUCT = 143,
    FSDB_TREE_CBT_FIELD_VECTOR               = 144,
    FSDB_TREE_CBT_FIELD_VLOG_PACKED_ARRAY    = 145,
     
    /* SystemC CB  */
    FSDB_TREE_CBT_DT_ARRAY_BEGIN        = 146,
    FSDB_TREE_CBT_DT_ARRAY_END          = 147,
    FSDB_TREE_CBT_DT_FIFO_BEGIN         = 148,
    FSDB_TREE_CBT_DT_FIFO_END           = 149,       

    FSDB_TREE_CBT_OBJECT_BEGIN        = 150,
    FSDB_TREE_CBT_OBJECT_END          = 151,
    
    FSDB_TREE_CBT_DT_TLM_FIFO_BEGIN     = 152,
    FSDB_TREE_CBT_DT_TLM_FIFO_END       = 153,
    
    /*continue Field type CB*/       
    FSDB_TREE_CBT_FIELD_VLOG_PACKED_STRUCT = 160,
    FSDB_TREE_CBT_FIELD_VLOG_TAGGED_UNION  = 161,
    FSDB_TREE_CBT_FIELD_MDF_SCOPE          = 162, 
    
    /* formbus for SLVF */
    FSDB_TREE_CBT_FORMBUS_BEGIN     = 163,
    FSDB_TREE_CBT_FORMBUS_END       = 164,

    /* Module base Scope*/ 
    FSDB_TREE_CBT_MDF_SCOPE         = 165,
    FSDB_TREE_CBT_MDF_VAR           = 166,

    /* Field type*/
    FSDB_TREE_CBT_FIELD_VHDL_ARRAY     = 167,
    FSDB_TREE_CBT_FIELD_VHDL_RECORD    = 168,
    FSDB_TREE_CBT_FIELD_VHDL_PHYSICAL  = 169,

    /*Module base composite var*/
    FSDB_TREE_CBT_MDF_COMPOSITE_VAR    = 170,

    /* SLVF formbus for enum var, internal used */
    FSDB_TREE_CBT_ENUM_VAR = 171
};
typedef enum fsdbTreeCBType     fsdbTreeCBType;

enum fsdbEnumValueType {
    FSDB_ENUM_VALUE_TYPE_INTEGER            = 0,
    FSDB_ENUM_VALUE_TYPE_LOGIC              = 1,
    FSDB_ENUM_VALUE_TYPE_TRANS_LOGIC        = 1,    /* For transaction */
    FSDB_ENUM_VALUE_TYPE_VERILOG_LOGIC      = 2,
    FSDB_ENUM_VALUE_TYPE_VHDL_LOGIC         = 3,
    /* Following types are internal only */
    FSDB_ENUM_VALUE_TYPE_LONG               = 4,
    FSDB_ENUM_VALUE_TYPE_SHORT              = 5,
    FSDB_ENUM_VALUE_TYPE_BYTE               = 6,
    FSDB_ENUM_VALUE_TYPE_UNSIGN_LONG        = 7,
    FSDB_ENUM_VALUE_TYPE_UNSIGN_INTEGER     = 8,
    FSDB_ENUM_VALUE_TYPE_UNSIGN_SHORT       = 9,
    FSDB_ENUM_VALUE_TYPE_UNSIGN_BYTE        = 10,
    FSDB_ENUM_VALUE_TYPE_SV_LONG            = 11,
    FSDB_ENUM_VALUE_TYPE_SV_INTEGER         = 12,
    FSDB_ENUM_VALUE_TYPE_SV_SHORT           = 13,
    FSDB_ENUM_VALUE_TYPE_SV_BYTE            = 14,
    FSDB_ENUM_VALUE_TYPE_SV_UNSIGN_LONG     = 15,
    FSDB_ENUM_VALUE_TYPE_SV_UNSIGN_INTEGER  = 16,
    FSDB_ENUM_VALUE_TYPE_SV_UNSIGN_SHORT    = 17,
    FSDB_ENUM_VALUE_TYPE_SV_UNSIGN_BYTE     = 18,
    FSDB_ENUM_VALUE_TYPE_NONE               = 19
};
typedef enum fsdbEnumValueType fsdbEnumValueType;

typedef bool_T  
(*fsdbTreeCBFunc)(fsdbTreeCBType tree_cb_type, 
                  void *tree_client_data, 
                  void *tree_cb_data);

typedef bool_T 
(*ffr_dt_consumer)(fsdbTreeCBType type, 
                    void* client_data, 
                    void* call_data);

typedef bool_T 
(*ffr_dt_chk_consumer)(fsdbTreeCBType type, 
                    void* client_data, 
                    void* call_data);

struct fsdbTreeCBDataBeginTree {
    fsdbLUOff   var_start_log_uoff;
    fsdbLUOff   sub_scope_off; 

    uint_T      real_obj_idx:16; 
    uint_T      has_memory_var:1;
    uint_T      has_property_var:1;
    uint_T      :14;

    uint_T      pad;
};
typedef struct fsdbTreeCBDataBeginTree  fsdbTreeCBDataBeginTree;

struct fsdbTreeCBDataScope {
    fsdbLUOff   var_start_log_uoff;
    str_T       name;                   /* scope name */
    str_T       module;                 /* module name */
    str_T       scale_unit;             /* scale unit */
    str_T       time_unit;              /* time unit */

    uint_T      real_obj_idx:16; 
    uint_T      type:8;                 /* scope type */
    uint_T      has_memory_var:1;
    uint_T      case_insensitive:1;
		/*
		** If it's true, then scope name is case insensitive.
		** If it's false, then scope name is case sensitive.
		*/
    uint_T      is_hidden_scope:1;
    uint_T      has_property_var:1;
    uint_T      is_xa_dummy_scope:1;    /* spice dummy scope for AMS */
    uint_T      :3; 

    uint_T      pad;
    str_T       full_decl_name;
};
typedef struct fsdbTreeCBDataScope      fsdbTreeCBDataScope;
typedef fsdbTreeCBDataScope             fsdbScopeRec;

struct fsdbTreeCBDataUpscope {
    fsdbLUOff   var_start_log_uoff;
    fsdbLUOff   sub_scope_off; 

    uint_T      real_obj_idx:16; 
    uint_T      has_memory_var:1;
    uint_T      has_property_var:1;
    uint_T      :14;

    uint_T      pad;
};
typedef struct fsdbTreeCBDataUpscope    fsdbTreeCBDataUpscope;

#if 0
/*
** From Debussy 5.2 (fsdb v2.2), data type of lbitnum/rbitnum
** is changed from ushort_T to int.
*/
struct fsdbTreeCBDataVar {
    str_T       name;           /* var name */
    union       _var {
        int     idcode;         /* var idcode */
        void    *handle;        /* var handle */
    } u;
    ushort_T    lbitnum;        /* left bit number */
    ushort_T    rbitnum;        /* right bit number */
    ushort_T    dtidcode;       /* data type idcode */
    byte_T      direction;      /* var direction */
    byte_T      type;           /* var type */

    byte_T      bytes_per_bit;  /* fsdbBytesPerBit */
    byte_T      vc_dt;          /* fsdbValueChangeDataType */
    byte_T      pad_byte[2];

    uint_T      len_descriptor;
    byte_T      *descriptor;
};
typedef struct fsdbTreeCBDataVar        fsdbTreeCBDataVar;
typedef fsdbTreeCBDataVar               fsdbVarRec;
#endif

/*
** From Debussy 5.2 (fsdb v2.2), data type of lbitnum/rbitnum
** is changed from ushort_T to int.
**
** From Debussy 5.3, extra_inf is added, due to property variables.
*/
struct fsdbTreeCBDataVar {
    str_T       name;           /* var name */
    union       _var {
        fsdbVarIdcode     idcode;         /* var idcode */
        void              *handle;        /* var handle */
    } u;
    int         lbitnum;        /* left bit number */
    int         rbitnum;        /* right bit number */

    ushort_T    dtidcode;       /* data type idcode */
    byte_T      direction;      /* var direction */
    byte_T      type;           /* var type */

    byte_T      bytes_per_bit;        /* fsdbBytesPerBit */
    byte_T      vc_dt;                /* fsdbValueChangeDataType */
    byte_T      is_dummy_var;       /* for internal use only*/
    byte_T      is_class_in_obj;    /* for internal use only*/
    byte_T      is_void;            /* for internal use only*/
    byte_T      not_used[3];           /* for internal use only*/

    uint_T      len_descriptor;
    byte_T      *descriptor;

    void	*extra_inf;	/* added for special type variable */
};
typedef struct fsdbTreeCBDataVar        fsdbTreeCBDataVar;
typedef fsdbTreeCBDataVar               fsdbVarRec;

/*
** fsdbPropVarExtraInf is added from Debussy 5.3
*/
struct fsdbPropVarExtraInf {
    str_T	assoc_clk_name;
    char	**assoc_var_name;
    uint_T	count;
    str_T	text_description;
};
typedef struct fsdbPropVarExtraInf	fsdbPropVarExtraInf;


struct fsdbTreeCBDataEndTree {
    void        *fsdb_reader_obj;
};
typedef struct fsdbTreeCBDataEndTree    fsdbTreeCBDataEndTree;

struct fsdbTreeCBDataEnum {
    ushort_T    idcode;         /* data type idcode */
    ushort_T    numLiteral;
    char        **arrLiteral;
    bool_T      is_has_xprop_value;
};
typedef struct fsdbTreeCBDataEnum       fsdbTreeCBDataEnum;
typedef fsdbTreeCBDataEnum              fsdbDTEnumRec;

struct fsdbTreeCBDataEnum2 {
    ushort_T    idcode;         /* data type idcode */
    byte_T      val_type;       /* type of val_arr. It's fsdbEnumValueType */
    byte_T      pad;
    str_T       name;
    uint_T      val_len;
    ushort_T    literal_count;
    byte_T      pad1[2];
    char        **literal_arr;
    byte_T      **val_arr;
};
typedef struct fsdbTreeCBDataEnum2  fsdbTreeCBDataEnum2;
typedef fsdbTreeCBDataEnum2         fsdbDTEnumRec2;

struct fsdbTreeCBDataEnum3 {
    ushort_T    idcode;         /* data type idcode */
    byte_T      val_type;       /* type of val_arr. It's fsdbEnumValueType */
    byte_T      pad;
    str_T       name;
    int         lbitnum; 
    int         rbitnum;        
    uint_T      val_len;        /* val_len = ABS(lbitnum-rbitnum) + 1 */ 
    ushort_T    literal_count;
    byte_T      pad1[2];
    char        **literal_arr;
    byte_T      **val_arr;
};
typedef struct fsdbTreeCBDataEnum3  fsdbTreeCBDataEnum3;
typedef fsdbTreeCBDataEnum3         fsdbDTEnumRec3;

struct fsdbTreeCBDataInt {
    ushort_T    idcode;         /* data type idcode */
    ushort_T    padShort;
    int         leftBound;
    int         rightBound;
};
typedef struct fsdbTreeCBDataInt        fsdbTreeCBDataInt;
typedef fsdbTreeCBDataInt               fsdbDTIntRec;

struct fsdbTreeCBDataIntHnL{
    ushort_T    idcode;         /* data type idcode */
    ushort_T    padShort;
};
typedef struct fsdbTreeCBDataIntHnL     fsdbTreeCBDataIntHnL;
typedef fsdbTreeCBDataIntHnL            fsdbDTIntHnLRec;

struct fsdbTreeCBDataFloating {
    ushort_T    idcode;         /* data type idcode */
    ushort_T    padShort;
    double      leftBound;
    double      rightBound;
};
typedef struct fsdbTreeCBDataFloating   fsdbTreeCBDataFloating;
typedef fsdbTreeCBDataFloating          fsdbDTFloatRec;
   
/*
** Physical
*/ 
struct fsdbTreeCBDataPhyBegin {
    ushort_T    idcode;         /* data type idcode */
    ushort_T    padShort;
    double      leftBound;
    double      rightBound;
    str_T       baseUnit;
};
typedef struct fsdbTreeCBDataPhyBegin   fsdbTreeCBDataPhyBegin;
typedef fsdbTreeCBDataPhyBegin          fsdbDTPhysicalRec;

/*
** Physical Secondary Unit
*/
struct fsdbTreeCBDataPhySecUnit {
    double      multiple;
    str_T       identifier;
};
typedef struct fsdbTreeCBDataPhySecUnit fsdbTreeCBDataPhySecUnit;
typedef fsdbTreeCBDataPhySecUnit        fsdbDTPhysicalSecondaryUnitRec;

// We declaire local variables of fsdbTreeCBDataArrayBegin fsdbDTArrayRec 
// in some API. As add a new flag, please be sure the flag is initilized
struct fsdbTreeCBDataArrayBegin {
    str_T       name;           /* array name */ 
    uint_T      size;           /* array size */
    byte_T      is_partial_dumped;
    byte_T      is_packed_array;
    byte_T      pad[2];
};  
typedef struct fsdbTreeCBDataArrayBegin fsdbTreeCBDataArrayBegin;
typedef fsdbTreeCBDataArrayBegin        fsdbDTArrayRec;
    
struct fsdbTreeCBDataRecordBegin {
    str_T       name;           /* record name */ 
    uint_T      fieldCount;     /* field count (number of field) */
    byte_T      is_partial_dumped;
    byte_T      pad[3];
};  
typedef struct fsdbTreeCBDataRecordBegin        fsdbTreeCBDataRecordBegin;
typedef fsdbTreeCBDataRecordBegin               fsdbDTRecordRec;

struct fsdbTreeCBDataBusBegin {
    str_T       name;           /* bus name */
};
typedef struct fsdbTreeCBDataBusBegin   fsdbTreeCBDataBusBegin;

struct fsdbTreeCBDataBusElement {
    str_T       name;           /* full path bus element name */
};
typedef struct fsdbTreeCBDataBusElement fsdbTreeCBDataBusElement;

enum fsdbStructType {
    FSDB_STRUCT_TYPE_VHDL_RECORD            = 0,
    FSDB_STRUCT_TYPE_HVL_DL                 = 1,    /* HVL driver/load */

    FSDB_STRUCT_TYPE_UNPACKED_STRUCT        = 10,   /* SV unpacked struct */
    FSDB_STRUCT_TYPE_PACKED_STRUCT          = 11,   /* SV packed struct   */
    FSDB_STRUCT_TYPE_SUPPLY_NET             = 12,   /* Power supply net packed struct, internal use only */
    FSDB_STRUCT_TYPE_SUPPLY_PORT            = 13,   /* Power supply port packed struct, internal use only */
                                                    /* 14 is reserved     */ 
    FSDB_STRUCT_TYPE_UNPACKED_UNION         = 15,   /* SV unpacked union  */
    FSDB_STRUCT_TYPE_PACKED_UNION           = 16,   /* SV packed union    */
    FSDB_STRUCT_TYPE_TAGGED_PACKED_UNION    = 17    /* SV tagged union    */
                                                    /* 18-20 reserved     */
};  
typedef enum fsdbStructType fsdbStructType;

struct fsdbTreeCBDataStructBegin {
    str_T   name;
    uint_T  fieldCount;     /* field count (number of field) */
    byte_T  type;           /* type is defined by fsdbStructType */
    byte_T  is_partial_dumped;
    byte_T  pad[2];
};
typedef struct fsdbTreeCBDataStructBegin fsdbTreeCBDataStructBegin;
typedef fsdbTreeCBDataStructBegin fsdbDTStructRec;

/*
** Types of class data type 
*/
enum fsdbClassType {
    FSDB_CLASS_TYPE_NONE = 0,
    FSDB_CLASS_TYPE_CLASS, 
    FSDB_CLASS_TYPE_STRUCT,
    FSDB_CLASS_TYPE_UNION
};
typedef enum fsdbClassType fsdbClassType;

struct fsdbTreeCBDataClass {
    ushort_T        idcode;         /* data type idcode */
    byte_T          pad[2];

    str_T           name;         
    fsdbClassType   class_type;     

    uint_T          attr_count;      
    ushort_T        *attr_dtidcode_arr;     
};
typedef struct fsdbTreeCBDataClass fsdbTreeCBDataClass;
typedef fsdbTreeCBDataClass fsdbDTClassRec;

struct fsdbTreeCBDataDTQueue {
    ushort_T dt_idcode;
    ushort_T sub_elem_dtidcode;
    uint_T   elem_count;
};
typedef struct fsdbTreeCBDataDTQueue fsdbTreeCBDataDTQueue;

struct fsdbTreeCBDataDTDynamicArray {  
    ushort_T dt_idcode;
    ushort_T sub_elem_dtidcode; 
    uint_T   elem_count; 
};  
typedef struct fsdbTreeCBDataDTDynamicArray fsdbTreeCBDataDTDynamicArray;

struct fsdbTreeCBDataDTAssocArray {
    ushort_T  dt_idcode;
    ushort_T  index_dtidcode;
    ushort_T  value_dtidcode;
    uint_T    elem_count;
};

typedef struct fsdbTreeCBDataDTAssocArray fsdbTreeCBDataDTAssocArray;

 
/*
** Support information 
** NOTE : Remember to update FSDB_SUPINFO_USER_DEFINED_LEN if any other len is
** changed. 
*/
#define FSDB_SUPINFO_LEN                2048
#define FSDB_SUPINFO_CREATION_TIME_LEN  30
#define FSDB_SUPINFO_HOST_NAME_LEN      20
#define FSDB_SUPINFO_ARCHITECTURE_LEN   20
#define FSDB_SUPINFO_OS_TYPE_LEN        12
#define FSDB_SUPINFO_OS_VERSION_LEN     50
#define FSDB_SUPINFO_GLITCH_NUM_LEN     2       /* sizeof(ushort_T) = 2 */
#define FSDB_SUPINFO_ENV_VAR_LEN        200
#define FSDB_SUPINFO_IDENT_LEN          120
#define FSDB_SUPINFO_PRODUCER_LEN       64
#define FSDB_SUPINFO_USER_DEFINED_LEN 1372
#define FSDB_SUPINFO_RELEASE_VERSION  30
#define FSDB_SUPINFO_DUMPER_VERSION     128

#define FSDB_DUMP_ENV_VAR_LEN   900
/* 
** DFTINFO are those default info filled by fsdb writer. That is,
** all info except producer and user_defined. 
*/
#define FSDB_SUPINFO_DFTINFO_LEN        FSDB_SUPINFO_LEN - \
                                        FSDB_SUPINFO_USER_DEFINED_LEN - \
                                        FSDB_SUPINFO_PRODUCER_LEN - \
                                        FSDB_SUPINFO_DUMPER_VERSION

typedef struct fsdbSupInfo  fsdbSupInfo;
struct fsdbSupInfo {
    char        creation_time[FSDB_SUPINFO_CREATION_TIME_LEN];
    char        host_name[FSDB_SUPINFO_HOST_NAME_LEN];
    char        architecture[FSDB_SUPINFO_ARCHITECTURE_LEN];
    char        os_type[FSDB_SUPINFO_OS_TYPE_LEN];
    char        os_version[FSDB_SUPINFO_OS_VERSION_LEN];
    ushort_T    glitch_num;
    char        env_var[FSDB_SUPINFO_ENV_VAR_LEN];
    char        ident[FSDB_SUPINFO_IDENT_LEN];
    char        producer[FSDB_SUPINFO_PRODUCER_LEN];
    char        user_defined[FSDB_SUPINFO_USER_DEFINED_LEN];
    char        release_version[FSDB_SUPINFO_RELEASE_VERSION];
    char        dumper_version[FSDB_SUPINFO_DUMPER_VERSION];
};

struct fsdbABBusVal {
    int  aval;
    int  bval;
};

typedef struct fsdbABBusVal fsdbABBusVal;

struct fsdbCDBusVal {
    int  cval;
    int  dval;
};

typedef struct fsdbCDBusVal fsdbCDBusVal;

/*
** Loop marker
*/
#define FSDB_MAX_LOOP_MARKER_COUNT 64

/*
** Transaction Recording
*/
typedef int             fsdbStreamHdl;
typedef fsdbDataType    fsdbAttrHdl;
typedef uint_T          fsdbTransId;
typedef fsdbTransId     fsdbMessageId;
typedef fsdbTransId     fsdbFsdbLogId;
typedef uint_T          fsdbRelationHdl;

#define FSDB_INVALID_STREAM_HDL     FSDB_INVALID_VAR_IDCODE
#define FSDB_INVALID_ATTR_HDL       FSDB_INVALID_DATA_TYPE_IDCODE
#define FSDB_INVALID_TRANS_ID       ((uint_T)0xFFFFFFFF)
#define FSDB_INVALID_MESSAGE_ID     ((uint_T)0xFFFFFFFF)
#define FSDB_INVALID_RELATION_HDL   ((uint_T)0xFFFFFFFF)

enum fsdbTransStatus {
    FSDB_TRANS_STATUS_UNKNOWN,
    FSDB_TRANS_STATUS_PENDING,
    FSDB_TRANS_STATUS_FINISHED,
    FSDB_TRANS_STATUS_CANCELLED
};
typedef enum fsdbTransStatus fsdbTransStatus;

enum fsdbAttrDataType {
    FSDB_ATTR_DT_UNKNOWN_TYPE       =  0, /* unknown type */
    FSDB_ATTR_DT_USER_DEFINED       =  1, /* user defined */
    FSDB_ATTR_DT_LOGIC              =  2, /* 4-state 1/0/X/Z */
    FSDB_ATTR_DT_BOOL               =  3, /* Boolean  */
    FSDB_ATTR_DT_CHAR               =  4, /* one character */
    FSDB_ATTR_DT_STRING             =  5, /* string of characters */
    FSDB_ATTR_DT_INT32              =  6, /* 32 bit signed integer */
    FSDB_ATTR_DT_INT64              =  7, /* 64 bit signed integer */
    FSDB_ATTR_DT_UINT32             =  8, /* 32 bit unsigned integer */
    FSDB_ATTR_DT_UINT64             =  9, /* 64 bit unsigned integer */
    FSDB_ATTR_DT_FLOAT              = 10, /* 32 bit float */
    FSDB_ATTR_DT_DOUBLE             = 11, /* 64 bit double */
    FSDB_ATTR_DT_ENUM               = 12, /* Enumeration */
    FSDB_ATTR_DT_REFERENCE          = 13, /* Reference */
    FSDB_ATTR_DT_EVENT              = 14, /* Event (0/1)*/ 
    FSDB_ATTR_DT_SV_ENUM            = 15, /* System Verilog Enumeration */
    FSDB_ATTR_DT_MDA                = 16, /* sparse MDA */
    
    /* System Verilog data types */
    FSDB_ATTR_DT_SV_MIN             = 17,
    FSDB_ATTR_DT_SV_LOGIC           = 17, /* Logic (4-state ary) */
    FSDB_ATTR_DT_SV_REG             = 18, /* Register (4-state ary) */
    FSDB_ATTR_DT_SV_BIT             = 19, /* Bit (2-state ary) */
    FSDB_ATTR_DT_SV_LONG_INT        = 20, /* Long Integer (64bit, 2-state ary) */
    FSDB_ATTR_DT_SV_LONG_UINT       = 21, /* Unsigned Long Integer (64bit, 2-state ary) */
    FSDB_ATTR_DT_SV_INT             = 22, /* Integer (32bit, 2-state ary) */
    FSDB_ATTR_DT_SV_UINT            = 23, /* Unsigned Integer (32bit, 2-state ary) */
    FSDB_ATTR_DT_SV_INTEGER         = 24, /* Integer (32bit, 4-state ary) */
    FSDB_ATTR_DT_SV_UINTEGER        = 25, /* Unsigned Integer (32bit, 4-state ary) */
    FSDB_ATTR_DT_SV_SHORT_INT       = 26, /* Short Integer (16bit, 2-state ary) */
    FSDB_ATTR_DT_SV_SHORT_UINT      = 27, /* Unsigned Short Integer (16bit, 2-state ary) */
    FSDB_ATTR_DT_SV_BYTE_INT        = 28, /* Byte Integer (8bit, 2-state ary) */
    FSDB_ATTR_DT_SV_BYTE_UINT       = 29, /* Unsigned Byte Integer (8bit, 2-state ary) */
    FSDB_ATTR_DT_SV_REAL            = 30, /* Real (64bit) */
    FSDB_ATTR_DT_SV_SHORT_REAL      = 31, /* Short Real (32bit) */
    FSDB_ATTR_DT_SV_TIME            = 32, /* Time (64bit, 4-state ary) */
    FSDB_ATTR_DT_SV_REFERENCE       = 33, /* Reference */
    FSDB_ATTR_DT_SV_STRING          = 34, /* String */
    FSDB_ATTR_DT_SV_CHAR            = 35, /* Char */
    FSDB_ATTR_DT_SV_EVENT           = 36, /* Event */
    FSDB_ATTR_DT_SV_MDA             = 37, /* sparse MDA */
    FSDB_ATTR_DT_SV_CLASS_ATTR      = 38,
    FSDB_ATTR_DT_SV_MAX             = 38,
    
    /* fsdb_log */
    FSDB_ATTR_DT_RAW_STRING         = 39,  /* string */

    FSDB_ATTR_DT_SV_PACKED_STRUCT_ATTR  = 40,
    FSDB_ATTR_DT_SV_PACKED_ARRAY_ATTR   = 41,
    FSDB_ATTR_DT_SV_PACKED_UNION_ATTR   = 42
};
typedef enum fsdbAttrDataType  fsdbAttrDataType;

#define FSDB_EVENT_ATTR_VALUE_FALSE   0 
#define FSDB_EVENT_ATTR_VALUE_TRUE    1 

enum fsdbAttrRadix {
    FSDB_ATTR_RADIX_MIN      = 0,
    FSDB_ATTR_RADIX_NONE     = 0,
    FSDB_ATTR_RADIX_BIN      = 1,
    FSDB_ATTR_RADIX_DEC      = 2,
    FSDB_ATTR_RADIX_UNSIGNED = 3, 
    FSDB_ATTR_RADIX_OCT      = 4,
    FSDB_ATTR_RADIX_HEX      = 5,
    FSDB_ATTR_RADIX_STRING   = 6,
    FSDB_ATTR_RADIX_MAX      = 6
};
typedef enum fsdbAttrRadix fsdbAttrRadix;

struct fsdbTreeCBDataAttr {
    ushort_T        idcode;         /* data type idcode */
    byte_T          hidden;
    byte_T          padByte;
    str_T           name;
    int             larridx;
    int             rarridx;
    ushort_T        meta_info;
    byte_T          pad[2];
    fsdbAttrRadix   radix;
};
typedef struct fsdbTreeCBDataAttr fsdbTreeCBDataAttr;
typedef fsdbTreeCBDataAttr        fsdbAttrDTRec;

#define fsdbTreeCBDataAttrLogic   fsdbTreeCBDataAttr
#define fsdbAttrDTLogicRec        fsdbAttrDTRec

struct fsdbTreeCBDataEnumAttr {
    ushort_T        idcode;         /* data type idcode */
    byte_T          hidden;
    byte_T          val_type;       /* type of val_arr. It's fsdbEnumValueType */
    fsdbAttrRadix   radix;
    str_T           name;
    int             larridx;
    int             rarridx;
    uint_T          val_len;
    ushort_T        literal_count;
    ushort_T        meta_info;
    char            **literal_arr;
    byte_T          **val_arr;
};
typedef struct fsdbTreeCBDataEnumAttr   fsdbTreeCBDataEnumAttr;
typedef fsdbTreeCBDataEnumAttr          fsdbEnumAttrDTRec;

struct fsdbTreeCBDataSVEnumAttr {
    ushort_T        idcode;         /* data type idcode */
    byte_T          hidden;
    byte_T          val_type;       /* type of val_arr. It's fsdbEnumValueType */
    fsdbAttrRadix   radix;
    str_T           name;
    int             lbitnum;
    int             rbitnum;
    uint_T          val_len; 
    ushort_T        literal_count;
    ushort_T        meta_info;
    char            **literal_arr;
    byte_T          **val_arr;
};
typedef struct fsdbTreeCBDataSVEnumAttr   fsdbTreeCBDataSVEnumAttr;
typedef fsdbTreeCBDataSVEnumAttr          fsdbSVEnumAttrDTRec;

/*
** (handle, value) pair of an attribute value
*/
struct fsdbAttrHdlVal {
    fsdbAttrHdl hdl;
    byte_T      *value;
};
typedef struct fsdbAttrHdlVal fsdbAttrHdlVal;

enum fsdbStreamType {
    FSDB_STREAM_TYPE_NORMAL,
    FSDB_STREAM_TYPE_LOCAL_VAR, 
    FSDB_STREAM_TYPE_MESSAGE,
    FSDB_STREAM_TYPE_FSDB_LOG,
    FSDB_STREAM_TYPE_OVM_TRANSACTION
};
typedef enum fsdbStreamType fsdbStreamType;

/*
** Transaction Color
*/
typedef enum fsdbColorId {
    FSDB_COLOR_ID_MIN               = 0,
    FSDB_COLOR_ID_WHITE             = 0,
    FSDB_COLOR_ID_LAVENDER          = 1,
    FSDB_COLOR_ID_LIGHTYELLOW2      = 2,
    FSDB_COLOR_ID_MISTYROSE         = 3,
    FSDB_COLOR_ID_GRAY              = 4,
    FSDB_COLOR_ID_SLATEGRAY         = 5,
    FSDB_COLOR_ID_RED               = 6,
    FSDB_COLOR_ID_ORANGERED         = 7,
    FSDB_COLOR_ID_TOMATO            = 8,
    FSDB_COLOR_ID_RED3              = 9,
    FSDB_COLOR_ID_ORANGE            = 10,
    FSDB_COLOR_ID_DARKORANGE        = 11,
    FSDB_COLOR_ID_YELLOW            = 12,
    FSDB_COLOR_ID_OLIVEDRAB1        = 13,
    FSDB_COLOR_ID_DARKOLIVEGREEN2   = 14,
    FSDB_COLOR_ID_GOLD              = 15,
    FSDB_COLOR_ID_LIGHTGOLDENROD    = 16,
    FSDB_COLOR_ID_YELLOWGREEN       = 17,
    FSDB_COLOR_ID_GREEN             = 18,
    FSDB_COLOR_ID_MEDIUMSPRINGGREEN = 19,
    FSDB_COLOR_ID_SPRINGGREEN2      = 20,
    FSDB_COLOR_ID_SEAGREEN3         = 21,
    FSDB_COLOR_ID_LIMEGREEN         = 22,
    FSDB_COLOR_ID_FORESTGREEN       = 23,
    FSDB_COLOR_ID_BLUE              = 24,
    FSDB_COLOR_ID_CORNFLOWERBLUE    = 25,
    FSDB_COLOR_ID_MEDIUMSLATEBLUE   = 26,
    FSDB_COLOR_ID_DEEPSKYBLUE       = 27,
    FSDB_COLOR_ID_LIGHTSKYBLUE      = 28,
    FSDB_COLOR_ID_CADETBLUE2        = 29,
    FSDB_COLOR_ID_CYAN              = 30,
    FSDB_COLOR_ID_PALETURQUOISE     = 31,
    FSDB_COLOR_ID_DARKTURQUOISE     = 32,
    FSDB_COLOR_ID_MEDIUMAQUAMARINE  = 33,
    FSDB_COLOR_ID_AQUAMARINE        = 34,
    FSDB_COLOR_ID_PURPLE            = 35,
    FSDB_COLOR_ID_DARKKHAKI         = 36,
    FSDB_COLOR_ID_SIENNA            = 37,
    FSDB_COLOR_ID_PERU              = 38,
    FSDB_COLOR_ID_BURLYWOOD         = 39,
    FSDB_COLOR_ID_SANDYBROWN        = 40,
    FSDB_COLOR_ID_CHOCOLATE         = 41,
    FSDB_COLOR_ID_BROWN             = 42,
    FSDB_COLOR_ID_SALMON            = 43,
    FSDB_COLOR_ID_HOTPINK           = 44,
    FSDB_COLOR_ID_DEEPPINK          = 45,
    FSDB_COLOR_ID_LIGHTPINK         = 46,
    FSDB_COLOR_ID_WHEAT2            = 47,
    FSDB_COLOR_ID_MAGENTA           = 48,
    FSDB_COLOR_ID_VIOLET            = 49,
    FSDB_COLOR_ID_AQUAMARINE1       = 50,
    FSDB_COLOR_ID_CHARTREUSE2       = 51,
    FSDB_COLOR_ID_PINK2             = 52,
    FSDB_COLOR_ID_GRAY80            = 53,
    FSDB_COLOR_ID_BLACK             = 54,
    FSDB_COLOR_ID_SPRINGGREEN       = 55,
    FSDB_COLOR_ID_OLIVEDRAB         = 56,
    FSDB_COLOR_ID_DARKSEAGREEN      = 57,
    FSDB_COLOR_ID_DARKSEAGREEN2     = 58,
    FSDB_COLOR_ID_SEAGREEN          = 59,
    FSDB_COLOR_ID_MEDIUMSEAGREEN    = 60,
    FSDB_COLOR_ID_LIGHTSEAGREEN     = 61,
    FSDB_COLOR_ID_PALEGREEN         = 62,
    FSDB_COLOR_ID_PINK              = 63,
    FSDB_COLOR_ID_WHEAT             = 64,
    FSDB_COLOR_ID_MAX               = 64
} fsdbColorId;

/*
** User could define his/her own message hooks so that warning and error
** messages could be redirected to these hooks.
*/
typedef int (*fsdbMsgCBFunc) (const char *,...);

/*
** Bus Protocol Support
*/
#define AHB_PROTOCOL            "AHB"
#define APB_PROTOCOL            "APB"
#define AXI_PROTOCOL            "AXI"
#define PCI_PROTOCOL            "PCI"
#define USB20_PROTOCOL          "USB20"
#define PCIX_PROTOCOL           "PCI-X"
#define PCIE_PROTOCOL           "PCI-E"
#define STB_PROTOCOL            "STBus"
#define HT_PROTOCOL             "HyperTransport"
#define RAPIDIO_PROTOCOL        "RapidIO"
#define DDR_SDRAM_PROTOCOL      "DDR-SDRAM"
#define DDR2_SDRAM_PROTOCOL     "DDR2-SDRAM"

#define FSDB_INVALID_BUS_HDL    FSDB_INVALID_TRANS_ID

typedef fsdbTransId             fsdbBusHdl;

enum fsdbClockType {
    FSDB_CLOCK_POSEDGE = 0,
    FSDB_CLOCK_NEGEDGE,
    FSDB_CLOCK_ANYEDGE
};
typedef enum fsdbClockType      fsdbClockType;

enum fsdbBusDataType {
    FSDB_BDT_GENERIC = 0,
    FSDB_BDT_COMMAND,
    FSDB_BDT_ADDRESS,
    FSDB_BDT_MASTER,
    FSDB_BDT_SLAVE,
    FSDB_BDT_DATA,
    FSDB_BDT_POLARITY			/* read/write */
};
typedef enum fsdbBusDataType    fsdbBusDataType;

/*
** Coverage Type
*/
enum fsdbCoverageType {
    FSDB_COVERAGE_OVERALL = 0,
    FSDB_COVERAGE_STRUCT,
    FSDB_COVERAGE_GROUP,
    FSDB_COVERAGE_SIMPLE_ITEM,
    FSDB_COVERAGE_CROSS_ITEM,
    FSDB_COVERAGE_TRANSITION_ITEM,
    FSDB_COVERAGE_BUCKET
};
typedef enum fsdbCoverageType fsdbCoverageType; 

enum fsdbIndptVar
{
    FSDB_INDPT_VAR_MIN          = 0,    /* minium indepent variable type */
    FSDB_INDPT_VAR_NONE         = 0,    /* indepent variable type is invalid
                                           or not set */
    FSDB_INDPT_VAR_TIME_DOMAIN  = 1,    /* time domain, transient analysis */
    FSDB_INDPT_VAR_FREQ_DOMAIN  = 2,    /* frequancy domain, AC analysis */
    FSDB_INDPT_VAR_VOLTAGE      = 3,    /* voltage, DC analysis */
    FSDB_INDPT_VAR_CURRENT      = 4,    /* current, DC analysis */
    FSDB_INDPT_VAR_TEMPERATURE  = 5,    /* temperature, DC analysis */
    FSDB_INDPT_VAR_PARAMETER    = 6,    /* parameter, DC analysis */
    FSDB_INDPT_VAR_PARAMETER1   = 7,    /* parameter1, DC analysis */
    FSDB_INDPT_VAR_MAX          = 7     /* maxium indepent variable type */
};
typedef enum fsdbIndptVar fsdbIndptVar;

enum fsdbListVarType {
    FSDB_LIST_LIST = 0,
    FSDB_LIST_DYNAMIC_ARRAY,
    FSDB_LIST_QUEUE,
    FSDB_LIST_MAILBOX,
    FSDB_LIST_FIFO
};
typedef enum fsdbListVarType fsdbListVarType;

struct fsdbMultiSweepCondition {
    str_T               key;
    str_T               value;
};
typedef struct fsdbMultiSweepCondition  fsdbMultiSweepCondition;

#define AB_BUS_RECORD_COUNT(bus_width)  (((((bus_width) - 1) / 32) + 1))
#define AB_BUS_BYTE_COUNT(bus_width)    (AB_BUS_RECORD_COUNT(bus_width) * (sizeof(fsdbABBusVal)))

struct fsdbArrayRange {
    int                         lbit;
    int                         rbit;
};
typedef struct fsdbArrayRange fsdbArrayRange;

struct fsdbTreeCBDataFieldUnpackedArray  {
    fsdbFieldType     idcode;  
    fsdbFieldType     base_field_type;
    uint_T            dim; 
    fsdbArrayRange    *ranges;
};
typedef struct fsdbTreeCBDataFieldUnpackedArray fsdbTreeCBDataFieldUnpackedArray;

struct fsdbTreeCBDataUnpackedStruct {
    fsdbFieldIdcode   idcode;
    uint_T            member_count;
    fsdbFieldIdcode   *member_dt_arr;
    str_T             *member_name_arr;
}; 
typedef struct fsdbTreeCBDataUnpackedStruct fsdbTreeCBDataUnpackedStruct;
typedef struct fsdbTreeCBDataUnpackedStruct fsdbTreeCBDataVerilogUnion;

struct fsdbTreeCBDataFieldVector  {
    fsdbFieldType     idcode;  
    fsdbFieldType     base_field_type;
    int               lbit; 
    int               rbit;
};
typedef struct fsdbTreeCBDataFieldVector fsdbTreeCBDataFieldVector;

struct fsdbTreeCBDataFormbusBegin {
    str_T   name;
    int     lbit;
    int     rbit;
};
typedef struct fsdbTreeCBDataFormbusBegin fsdbTreeCBDataFormbusBegin;


enum fsdbPowerType {
  FSDB_NO_POWER         = 0,    /* for error protect */
  FSDB_POWER_VCS_PPI    = 1,    /* VCS NLP 1.0 */
  FSDB_POWER_VCS_PLATO  = 2,    /* VCS NLP 2.0 */
  FSDB_POWER_VCS_PPI_WITH_POSTFIX = 3,
  FSDB_POWER_ZEBU_VCS_PPI = 4,
  FSDB_POWER_UNKNOWN      = 5,  
  FSDB_POWER_TYPE_MAX   = 5
};
typedef enum fsdbPowerType fsdbPowerType;


#ifdef NOVAS_FSDB
#include "fsdbShrPrivate.h"
#endif

#ifdef FSDB_INTERNAL_HEADER
#include "fsdbShr_Int.h"
#endif

#endif /* fsdbShr_h */
