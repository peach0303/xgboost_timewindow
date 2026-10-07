/* *****************************************************************************
// [ffrAPI.h]
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
#ifndef ffrAPI_h
#define ffrAPI_h


#define FFR_API_VERSION_STR 5.7
/*
** FFR_API_VERSION_STR is the same as fsdb version
*/

/*
** Unlike 5.3, ffrVCTrvsHdl must be defined before ffrWrapDS.h is included
** since ffrTimeBasedVCTrvsHdl needs it.
*/
typedef class ffrVCIterOne*             ffrVCTraverseHandle;
typedef class ffrVCIterOne*             ffrVCTrvsHdl;

typedef class ffrTimeBasedVCIterOne*    ffrTimeBasedVCTrvsHdl;

#ifdef NOVAS_FSDB
#include "ffrWrapDS.h"
#endif

#include "ffrKit.h"

#ifndef NULL
#define NULL    0
#endif

/*
**  ffrObject object
*/

typedef struct ffrObjData   ffrObjData;
typedef struct ffrObjData20 ffrObjData20;

union ffrXTag {
    fsdbTag64   t64;
    float       f;
    double      d;
};
typedef union  ffrXTag                  ffrXTag;

// recorde FSDB file information
#define FSDB_VERSION_STR_LEN    4
struct ffrFSDBInfo {
    fsdbFileType    file_type;
    fsdbFileStatus  file_status;
    char        version[FSDB_VERSION_STR_LEN];
    char        scope_separator[FSDB_LEN_SCOPE_SEPARATOR];
    char        scale_unit[FSDB_LEN_SCALE_UNIT];
    char        simulation_date[FSDB_LEN_SIMULATION_DATE];
    char        simulator_version[FSDB_LEN_SIMULATOR_VERSION];
                
    fsdbXTag    min_xtag;
    fsdbXTag    max_xtag;
                
    fsdbVarCount      unique_var_count;
    fsdbVarCount      total_var_count;
    uint_T      total_tree_count;
    ushort_T    max_glitch_num;
    ushort_T    pad_short;
                
    bool_T      is_view_window_available;
    bool_T      pad_bool[3];
};
typedef struct ffrFSDBInfo  ffrFSDBInfo;

// recorde var information
struct ffrVarInfo {
    byte_T      type;   /* var type */
    byte_T      bpb;
    byte_T      not_used[2];

    ushort_T    lbitnum;
    ushort_T    rbitnum;
    uint_T      byte_count;

    uint_T      len_descriptor;
    byte_T      *descriptor;

    byte_T      *extra_info;
};
typedef struct ffrVarInfo   ffrVarInfo;

// recorded Info about flush sessions. Useful for partial load
typedef struct ffrSessionInfo   ffrSessionInfo;

struct ffrSessionInfo {
    ffrSessionInfo  *next;
    fsdbXTag        start_xtag;
    fsdbXTag        close_xtag;
};

// recored var xtag and value information
struct ffrXTagValSeqNum {
    fsdbXTag    xtag;
    byte_T      *val;
    fsdbSeqNum  seq_num;
};
typedef struct ffrXTagValSeqNum     ffrXTagValSeqNum;

// Property Statistics
struct ffrPropStatistics {
    uint_T  total_success;
    uint_T  total_failure;
};
typedef struct ffrPropStatistics    ffrPropStatistics;

struct ffrPropVarStatInfo {
    uint_T      success_count;
    uint_T      failure_count;
    uint_T      vsuccess_count;
    uint_T      incomplete_count;
    uint_T      asserton_count;
    uint_T      assertoff_count;
    uint_T      kill_count;

    fsdbXTag    first_watched_begin_time;
    fsdbXTag    first_watched_end_time;
};
typedef struct ffrPropVarStatInfo   ffrPropVarStatInfo;

//recored clock information
struct ffrClockInfo {
    fsdbXTag    start_xtag; // start time
    uint_T      half_cycle;     // half cycle
    byte_T      first_val;      // first value
    byte_T      second_val;     // second value
    byte_T      not_used[2];
    uint_T      total_vc_count; // total value change count
};
typedef struct ffrClockInfo ffrClockInfo;

// recorde var extra MDA information
struct ffrVarInfo_Extra_MDA {
    ushort_T    dim;
    ushort_T    packed_arr_begin_pos;
    int         *left_idx_arr;
    int         *right_idx_arr;
    byte_T      *init_val_a_bit;
};
typedef struct ffrVarInfo_Extra_MDA ffrVarInfo_Extra_MDA;

// Coverage Value
struct ffrCoverageVarValue {
    double  grade_val;
    uint_T  hit_val;
};
typedef struct ffrCoverageVarValue ffrCoverageVarValue;

// OEM function prototype
struct ffrOEMOverrideData {
    fsdbXTag    min_xtag;
    fsdbXTag    max_xtag;
    str_T       scope_separator;
    str_T       scale_unit;
    str_T       simulation_date;
    str_T       simulator_version;
};
typedef struct ffrOEMOverrideData   ffrOEMOverrideData;

// For compressed FSDB file
struct ffrCmprsFileArg {
    str_T file_name;            // File hierarchy name
    str_T decompress_file_loc;  // The location of 
                                // decompressed FSDB
    str_T decompressor_name;    //If is NULL, decopressor is
                                // gzip. 
    str_T decompressor_option;  // The option of the
                                // decopressor.
    uint_T seconds_to_wait;     // The seconds to wait before 
                                // kill the decompression. If is
                                // 0, will always wait till finished.
    str_T  ret_msg_buf;       // The buffer for return mesage.
    uint_T ret_msg_buf_size;  // the buffer size, if the real msaage length is 
                              // larger than the given size, it will be truncated.
};
typedef struct ffrCmprsFileArg ffrCmprsFileArg;

// For ffrOEMFuncTbl
typedef fsdbRC (*ffrOEMOpenFile)(str_T fname, ffrFSDBInfo *fsdb_info,
            void **ret_oem_file_hdl);
typedef fsdbRC (*ffrOEMCloseFile)(void *oem_file_hdl, uint_T reference_count);
typedef fsdbRC (*ffrOEMSetFile)(void *oem_file_hdl);
typedef fsdbRC (*ffrOEMSetViewWindow)(fsdbXTag *begin, fsdbXTag *end);
typedef fsdbRC (*ffrOEMAddToSignalList)(void *oem_file_hdl,
            str_T full_path_var_name, fsdbVarIdcode var_idcode,
            ffrVarInfo *var_info, void **ret_oem_var_hdl);
typedef fsdbRC (*ffrOEMLoadSignals)(void *oem_file_hdl);
typedef fsdbRC (*ffrOEMGetOverrideData)(void *oem_file_hdl,
            ffrOEMOverrideData **ret_data);
typedef fsdbRC (*ffrOEMAskVC)(void *oem_file_hdl, bool_T &has_new_vc);
typedef fsdbRC (*ffrOEMSaveToFSDB)(void *oem_file_hdl,
            str_T output_filename, fsdbVarIdcode var_idcode_arr[], 
            uint_T count);
typedef fsdbRC (*ffrOEMKillSaveToFSDBProcess)();
// For ffrToOEMFuncTbl
typedef void (*ffrOEMBusyOn)(const char *, ...);
typedef int  (*ffrOEMBusyGetUserAbort)(void);
typedef void (*ffrOEMBusyOff)(void);
typedef fsdbRC (*ffrInsertOEMComputedVC)(void *oem_var_hdl, fsdbXTag *time,
            byte_T *vc);
typedef fsdbRC (*ffrInsertOEMComputedPropVC)(void *oem_var_hdl,
            fsdbXTag *begin_time, fsdbXTag *end_time, str_T result_str);
typedef fsdbRC (*ffrInsertOEMComputedTransVC)(void *oem_var_hdl,
            fsdbXTag *begin_time, fsdbXTag *end_time, str_T label,
            fsdbAttrHdlVal attr_arr[], ushort_T attr_count, 
            fsdbTransStatus status);
typedef fsdbRC (*ffrInsertOEMComputedMDAVC)(void *oem_var_hdl, fsdbXTag *time,
            int address, byte_T *val);
typedef fsdbRC (*ffrOEMEnableTBInsertVC)();

// OEM function table
struct ffrOEMFuncTbl {
    ffrOEMOpenFile              oem_open_file;
    ffrOEMCloseFile             oem_close_file;
    ffrOEMSetFile               oem_set_file;
    ffrOEMSetViewWindow         oem_set_view_window;
    ffrOEMAddToSignalList       oem_add_to_signal_list;
    ffrOEMLoadSignals           oem_load_signals;
    ffrOEMGetOverrideData       oem_get_override_data;
    ffrOEMAskVC                 oem_ask_vc;
    ffrOEMSaveToFSDB            oem_save_to_fsdb;
    ffrOEMKillSaveToFSDBProcess oem_kill_save_to_fsdb_process;
};
typedef struct ffrOEMFuncTbl    ffrOEMFuncTbl;

struct ffrToOEMFuncTbl {
    ffrOEMBusyOn                oem_busy_on;
    ffrOEMBusyGetUserAbort      oem_busy_get_user_abort;
    ffrOEMBusyOff               oem_busy_off;
    ffrInsertOEMComputedVC      oem_insert_computed_vc;
    ffrInsertOEMComputedPropVC  oem_insert_computed_prop_vc;
    ffrInsertOEMComputedTransVC oem_insert_computed_trans_vc;
    ffrInsertOEMComputedMDAVC   oem_insert_computed_mda_vc;
    ffrOEMEnableTBInsertVC      oem_enable_tb_insert_vc;
};
typedef struct ffrToOEMFuncTbl    ffrToOEMFuncTbl;

// These functions are defined on OEM side and called by FSDB to pass data
// and function pointer tables between these two sides.

extern "C" {
    void ffrOEMInit(int argc, char *argv[], void *xt_app_context,
    ffrOEMFuncTbl           *ret_oem_func_tbl,
    ffrInsertOEMComputedVC  insert_oem_computed_vc);

    void ffrOEMInit2(int argc, char *argv[], void *xt_app_context,
    ffrOEMFuncTbl   *ret_oem_func_tbl,
    ffrToOEMFuncTbl *ret_to_oem_func_tbl);
};
// --------------------------------------------------
// end of OEM declaration
//

//
// For Non-blocking Managerial layer
//
class JobMap;
class JobQueue; 
class PriJobStack; 

class VarManager;
class OneLevelArray;
class TwoLevelArray;
class ThreeLevelArray;
class ffrObject;
class ffrRealObject;

#ifdef  WIN32
#if WIN32 && !USE_LIB_FFR
#if NFFR_EXPORTS
class __declspec(dllexport) ffrVCIterOne
#else // NFFR_EXPORTS
class __declspec(dllimport) ffrVCIterOne
#endif // NFFR_EXPORTS
#else
class ffrVCIterOne
#endif // WIN32 && !USE_LIB_FFR
#else // WIN32 
class ffrVCIterOne
#endif // WIN32 
{

  friend class          ffrObject;
  friend class          ffrTimeBasedVCIterOne;
  friend class          ffrRealObject;

  protected:

    void    *dm;    // data member
    uint_T      idx_in_split_vrtl_file;

#ifdef NOVAS_FSDB
#include "ffrVCTrvsHdlAPI.h"
#endif

  public:

    //
    // APIs for third parties
    //

    //
    // vc_trvs_handle API
    //---------------------------------------------------
    fsdbRC ffrGotoXTag(void *xtag, int *ret_glitch_num = NULL);
    /*
    purpose:
        Point the value change traverse handle to the last value change at xtag.
        If there is no value change at xtag, the value change traverse handle is
        aligned based on the jump behavior defined by fsdb reader and xtag will
        be updated based on the aligned value change. For details, please refer
        to Jump Behavior section. If there is(are) value change(s) at xtag, the 
        traverse handle is moved to the last value change at xtag and the number
        of glitches at xtag is calculated and returned to application.
    input/output:
	*xtag: The specified x coordinate point. 
               The value of (*xtag) will be changed to time point at LHS of *xtag, if no vc at (*xtag).  

    return:
    	FSDB_RC_SUCCESS if successful; 
    	FSDB_RC_FAILURE otherwise.
    */
    fsdbRC ffrGotoNextVC();
    /*
    purpose:
	    Move the traverse point to next value change.

    return:
    	FSDB_RC_FAILURE on failure, there is no value change behind current one. 
    	FSDB_RC_SUCCESS on success
	*/
    fsdbRC ffrGotoPrevVC();
    /*
    purpose:
	    Move the traverse point to the previous value change.

    return:
    	FSDB_RC_FAILURE on failure, there is no value change before current one. 
    	FSDB_RC_SUCCESS on success
    */

    fsdbRC ffrGetXTagVCSeqNum(fsdbXTag *xtag, byte_T **vc, fsdbSeqNum *seq_num);
    /*
    Purpose
        Get xtag, value, and sequence number of the value change this traverse 
        handle points to now.
    If an application needs to retrieve the value of seq_num, it should call 
    ffrObject::ffrIsThereSeqNum to detect if the fsdb file contains sequence 
    number or not first. If the fsdb file does not contain sequence number and 
    applications calls this function, the value of *seq_num will be 0.
    
    This function only supports fsdb file version > 2.3.
    Arguments
        xtag    : A pointer to fsdbXTag representing the returned xtag.
        vc      : A pointer to byte_T* representing the value.
        seq_num : A pointer to fsdbSeqNum representing the sequence number. 
                  fsdbSeqNum is defined in fsdbShr.h.
    Return Value
        FSDB_RC_SUCCESS if successful.
        FSDB_RC_FAILURE when the signal does not have any value change or the 
        fsdb file version is < 2.0
    */
    void   ffrFree();
    /*
    purpose:
	    Free the data structures used by this traverse handle.    
    */
    fsdbRC ffrGotoTheFirstVC();
    fsdbRC ffrGetXTag(void *ret_xtag);
    fsdbRC ffrGetGlitchNum(int *ret_glitch_num);
    fsdbRC ffrGetVC(byte_T **ret_vc);
    fsdbRC ffrGetSeqNum(fsdbSeqNum *ret_seq_num);

    fsdbRC ffrGetMinXTag(void *ret_xtag);
    fsdbRC ffrGetMaxXTag(void *ret_xtag);

    bool_T ffrHasIncoreVC();
    uint_T ffrGetRefCount() const;

    //---------------------------------------------------
    // vc_trvs_handle API end
    //

    //
    // file base API
    // ---------------------------------------------------
    uint_T          ffrGetBitSize();
    uint_T          ffrGetByteCount();
    fsdbBytesPerBit ffrGetBytesPerBit();
    fsdbVarType     ffrGetVarType();
    fsdbVarType     ffrGetPowerVarRealType();
    bool_T          ffrGetVwIsSet();
    ffrObject*      ffrGetReaderObj();
    // --------------------------------------------------
    // file base API end
    //

    //
    // transaction info API, only used for fsdb file in 2.x format
    // --------------------------------------------------
    fsdbRC  ffrGetTransInfo(ffrTransInfo *&info);
    /*
    Purpose
        Gets the values of the transaction specified by trans_id. The result 
        will be filled in info by fsdb reader.
    Arguments
        trans_id : The id of the transaction.
        info     : An ffrTransInfo that will hold the result returned by fsdb 
            reader.
    Return Value
        FSDB_RC_SUCCESS on success; 
        FSDB_RC_FAILURE otherwise.
    */
    fsdbRC  ffrGetTransVal(fsdbTransId &trans_id, str_T &label,
                fsdbXTag &begin_time, fsdbXTag &end_time,
                fsdbTransStatus &status, ushort_T &attr_count,
                ushort_T *&attr_hdl_arr, byte_T **&attr_val_arr,
                fsdbSeqNum &seq_num);
    // -----------------------------------------------------
    // transaction info API end
    //
};

#ifdef  WIN32
#if WIN32 && !USE_LIB_FFR
#if NFFR_EXPORTS
class __declspec(dllexport) ffrTimeBasedVCIterOne
#else // NFFR_EXPORTS
class __declspec(dllimport) ffrTimeBasedVCIterOne
#endif // NFFR_EXPORTS
#else
class ffrTimeBasedVCIterOne
#endif // WIN32 && !USE_LIB_FFR
#else // WIN32
class ffrTimeBasedVCIterOne
#endif // WIN32 
{

  friend class          ffrObject;
  friend class          ffrRealObject;

  protected:

    void            *dm;    // data member

#ifdef NOVAS_FSDB
#include "tb_vc_trvs_hdl.h"
#endif

  public:

    //
    // Those are used for fsdb file in 1.x format and 2.x format
    //

    //
    // Get current signal information
    // ---------------------------------
    uint_T          ffrGetBitSize();
    fsdbBytesPerBit ffrGetBytesPerBit();
    uint_T          ffrGetByteCount();
    fsdbVarType     ffrGetVarType();

    // the 2 APIs are used for fsdb file in 2.3 format
    bool_T          ffrIsItPropVar() const;
    bool_T          ffrIsItStreamVar() const;

    // ---------------------------------
    // signal information API end
    //

    // Traversal
    fsdbRC  ffrGotoNextVC();
    /*
    Purpose
        Point the traverse handle to next value change. Since this is a 
        time-based traverse handle, the value change may beglong to another 
        signal. Application must notice this.
        The function is similar to ffrVCIterOne::ffrGotoNextVC.
    Return Value
        FSDB_RC_SUCCESS if successful; FSDB_RC_FAILURE otherwise.
    */

    fsdbRC  ffrGetVarIdcodeXTagVCSeqNum(fsdbVarIdcode *var_idcode,
                fsdbXTag *xtag, byte_T **vc, fsdbSeqNum *seq_num);
    /*
    Purpose
        When the traverse handle stops at certain value change of certain 
        signal, get idcode, xtag, value, and sequence number of the signal. It 
        is similar to ffrVCIterOne::ffrGetXTagVCSeqNum.
    Arguments
        var_idcode  : A ponter to fsdbVarIdcode representing the returned idcode
                 of the var.
        xtag        : A pointer to fsdbXTag representing the returned xtag.
        vc          : A pointer to byte_T* representing the value.
        seq_num     : A pointer to fsdbSeqNum representing the sequence number. 
                 fsdbSeqNum is defined in fsdbShr.h.
    Return Value
        FSDB_RC_SUCCESS if successful. This function returns FSDB_RC_FAILURE 
        when the traverse handle does not point to any value change or the fsdb 
        file version is < 2.0
    */

    fsdbRC  ffrGetTransInfo(ffrTransInfo *&info);
    
    void    ffrFree();
    /*
    purpose:
	    Free the data structures used by this traverse handle.    
    */

    fsdbRC  ffrGetVarIdcode(fsdbVarIdcode *ret_var_idcode);
    fsdbRC  ffrGetXTag(void *ret_xtag);
    fsdbRC  ffrGetVC(byte_T **ret_vc);
    fsdbRC  ffrGetSeqNum(fsdbSeqNum *ret_seq_num);
    fsdbRC  ffrGetGlitchNum(int *ret_glitch_num);
    fsdbRC  ffrGetTransVal(fsdbTransId &trans_id, str_T &label,
                fsdbXTag &begin_time, fsdbXTag &end_time, fsdbTransStatus &status,
                ushort_T &attr_count, ushort_T *&attr_hdl_arr, byte_T **&attr_val_arr,
                fsdbSeqNum &seq_num);

    // ------------------------------------
    // tb_vc_trvs_hdl API end
    //
};

#ifdef NOVAS_FSDB
#include "ffrVCTrvsHdlDS.h"
#endif

class IJoinedFile;
#ifdef WIN32
#ifndef NOVAS_FSDB
class ffrVar;
typedef ffrVar* ffrVarPtr;
class ffrPKVar;
typedef ffrPKVar* ffrPKVarPtr;
#endif
#endif
//
// Those are used for fsdb file in 2.x format
// --------------------------------------------------
#ifdef  WIN32
#if WIN32 && !USE_LIB_FFR
#if NFFR_EXPORTS
class __declspec(dllexport) ffrObject
#else // NFFR_EXPORTS
class __declspec(dllimport) ffrObject
#endif // NFFR_EXPORTS
#else
class ffrObject
#endif // WIN32 && !USE_LIB_FFR
#else // WIN32 
class ffrObject
#endif // WIN32 
{
  
  protected:

    static ffrOEMFuncTbl    oem_func_tbl;
    static ffrObject        *oem_active_obj;

    static ffrRealObject    *head_obj_list;
    static bool_T           resolve_sym_link;

    // For user-defined message callback function
    static fsdbMsgCBFunc    warn_cb_func;
    static fsdbMsgCBFunc    error_cb_func;
    static fsdbMsgCBFunc    info_cb_func;
  
  protected:
    ffrObject()
    {
    };

    ffrObject(str_T fsdb_fname, bool_T is_child_thread, IJoinedFile* pIJoinedFile)
    {
    };

    virtual 
        ~ffrObject()
    {
    }
    ;

  public:

    //
    // ---------- Begin of APIs for 3rd Parties ----------
    //

    //
    // file API
    // --------------------------------------------------
    static bool_T   ffrIsFSDB(str_T file_name);
    /*
    purpose:
        To see if this file is a fsdb file or not.
    
    return value:
        TRUE if it is a fsdb file
        FALSE if it's not a fsdb file
	*/

    static ffrObject*   ffrOpenNonSharedObj(str_T fname);
    /*
    purpose:
        Open a non-shared fsdb file
    
    argument:
        fname: a pointer to the file name string
 
    return:
        NULL on failure
        NON-NULL on success
 	*/


    static ffrObject*   ffrOpen3(str_T fname);
    /*
    purpose:
        Open a fsdb file
    
    argument:
        fname: a pointer to the file name string
 
    return:
        NULL on failure
        NON-NULL on success
 	*/

    virtual void
        ffrClose() = 0;
    /*
    purpose: 
        To stop accessing and close the FSDB file
    */

    // --------------------------------------------------
    // end file API
    //

    //
    // call-back function API
    // --------------------------------------------------
    virtual fsdbRC  
        ffrSetTreeCBFunc(fsdbTreeCBFunc cb_func,
                void *tree_client_data) = 0;
    /*
    purpose:
        Register file tree callback function and client data. The fsdb reader
        will read each signal definition and pass its information back via the 
        registered tree callback function.
        
        callback function:
        
            typedef bool_T  (*fsdbTreeCBFunc)(fsdbTreeCBType type, 
                          void *tree_client_data, 
        				  void *tree_cb_data);
        
        The tree_cb_data would be the pointer to fsdbVarRec, fsdbScopeRec ... 
        etc, depends on the callback type; you need to perform type cast on 
        tree_cb_data based on the callback type. 
    
    argument:
        tree_cb_func: 		function pointer of the tree callback
        tree_client_data:   tree client data 
    
    return:
        NULL on failure
        NON-NULL on success
    */
    virtual fsdbRC  
        ffrGetTreeCBFunc(fsdbTreeCBFunc *ret_tree_cb_func,
                void **ret_tree_client_data) = 0;
    /*
    purpose:
        Get file tree callback function and client data.
        
        The ret_tree_cb_data would be the pointer to fsdbVarRec, 
        fsdbScopeRec ... etc, depends on the callback type; 
        you need to perform type cast on tree_cb_data based on the callback 
        type. 
    
    argument:
        ret_tree_cb_func: 		function pointer of the tree callback
        ret_tree_client_data:   tree client data 
    
    return:
        NULL on failure
        NON-NULL on success
    */
    
    // --------------------------------------------------
    // end call-back function API
    //


    //
    // file information API
    // --------------------------------------------------
    virtual fsdbVarIdcode   
        ffrGetMaxVarIdcode()    const = 0;
    /*
    purpose:
    	Get the maximum signal idcode of this fsdb file. 

    return:
    	The maximum signal idcode of this fsdb file. The return value also means
    	that how many unique signals are there in this fsdb file.

    */
   
    virtual str_T           
        ffrGetScaleUnit()   const = 0;
    /*
    purpose: 
        Get the scale unit of xtag.
    
    return
        a char pointer to the scale unit
	*/
    
    virtual fsdbFileType    
        ffrGetFileType() = 0;

    static fsdbRC   ffrGetFSDBInfo(str_T fsdb_fname, ffrFSDBInfo &fsdb_info);
                    
    virtual bool_T          
        ffrIsThereSeqNum() = 0;
        
    virtual ffrSessionInfo* 
        ffrGetSessionListInfo() const = 0;

    virtual str_T           
        ffrGetFileDumperType()  const = 0;
        
    virtual str_T           
        ffrGetFileSimType()  const = 0;

    virtual str_T           
        ffrGetSimVersion()  const = 0;

    virtual str_T           
        ffrGetSimDate()     const = 0;

    virtual str_T           
        ffrGetScopeSeparator()  const = 0;

    virtual str_T           
        ffrGetAnnotation(uint_T &len)   const = 0;

    virtual void            
        ffrSetFsdbTag64LoadRange(fsdbTag64 *start, 
                            fsdbTag64 *end) = 0;

    // --------------------------------------------------
    // end file information API
    //

    //
    // add var API
    // --------------------------------------------------
    virtual fsdbRC  
        ffrAddToSignalList(fsdbVarIdcode idcode,
            ffrMinMaxAnalogValRec *min_max_analog_val = NULL,
            str_T full_path_var_name = NULL) = 0;
    /*
    purpose:
        To add a signal's idcode into the signal list
    
    argument:
        idcode: The idcode of interested signal. Note that the application can 
                get the idcode of a signal in the tree callback.
        min_max_analog_val:
                
        full_path_var_name:
                
    
    return
        FSDB_RC_FAILURE on failure
        FSDB_RC_SUCCESS on success
    */
    virtual fsdbRC  
        ffrResetSignalList() = 0;
    /*
    purpose:
        To clean the signal list.
    
    return
        FSDB_RC_FAILURE on failure
        FSDB_RC_SUCCESS on success
    */
    virtual fsdbRC  
        ffrLoadSignals() = 0;
    /*
    purpose:
        To load all the value changes of the signals in signal list into memory.
    
    return:
        FSDB_RC_FAILURE on failure
        FSDB_RC_SUCCESS on success
    */
    virtual fsdbRC  
        ffrUnloadSignals() = 0;
    /*
    purpose:
        Unload all the memory used by those signals which are in signal list. 
    
    return
        FSDB_RC_FAILURE on failure
        FSDB_RC_SUCCESS on success
    */
    virtual bool_T  
        ffrIsInSignalList(fsdbVarIdcode idcode) = 0;

    virtual fsdbRC  
        ffrUnloadSignals(fsdbVarIdcode var_idcode_arr[], fsdbVarCount count) = 0;

    virtual fsdbRC  
        ffrUnloadSignals(fsdbVarIdcode var_idcode) = 0;

    // --------------------------------------------------
    // end add var API
    //

    //
    // read scope/ datatype/ var API
    // --------------------------------------------------
    virtual fsdbRC 
        ffrReadScopeVarTree() = 0;
    /*
    Purpose
        Read both scope/signal information in the fsdb file. Fsdb reader will 
        activate the tree callback function and pass scope/signal information to
        application through the function.

    Return Value
        FSDB_RC_SUCCESS if successful; 
        FSDB_RC_FAILURE otherwise.
    */
    virtual fsdbRC 
        ffrReadScopeVarTree2(fsdbTreeCBFunc tree_cb_func, 
                   void *tree_client_data) = 0;
    /*
    Purpose
        This function is similar to ffrObject::ffrReadScopeVarTree with the 
        difference that caller could ask the function to use tree_cb_func and 
        tree_client_data instead of that of the fsdb reader object to handle 
        scope/signal information. This is useful when caller needs to handle 
        scope/signal information in its own way but does not want to change the 
        tree function pointer and tree callback data of the fsdb reader object.
        Note that the tree function pointer and tree callback data of the fsdb 
        reader object remains intact after the function finishes.
    Arguments
        tree_cb_func:
            Tree callback function pointer that will be used in this function.
        tree_client_data:
            Tree client data that will be used in this function.
    Return Value
        FSDB_RC_SUCCESS if successful; 
        FSDB_RC_FAILURE otherwise.
    */
    virtual fsdbRC 
        ffrReadVarByLogUOff(fsdbLUOff *var_start_log_uoff) = 0;
    /*
    Purpose
        When application needs to read all signals of a scope, it should call 
        this function and supply it with the var_start_log_uoff of that scope 
        which it kept previously. In this way, all signals of that scope will be
        read from the fsdb file and the tree callback function will be activated
        for each signal. When the tree callback function is activated, the 
        callback type will be FSDB_TREE_CBT_VAR. Application should cast the 
        callback data to fsdbTreeCBDataVar in order to get the information of 
        the returned signal. After the function returns, all signals of the 
        desired scope will all be retrieved.
    Arguments
        var_start_log_uoff:
            Start logical unsigned offset of the signals belonging to the 
            desired scope.
    Return Value
        FSDB_RC_SUCCESS if successful; 
        FSDB_RC_FAILURE otherwise.
    */
    virtual fsdbRC 
        ffrReadVarByLogUOff(fsdbLUOff   *var_start_log_uoff, 
                                            ushort_T    real_file_obj_idx) = 0;
    /*
    Purpose
        This API is only for virtual file. When application needs to read all 
        signals of a scope, it should call this function and supply it with 
        the var_start_log_uoff of that scope which it kept previously. In 
        this way, all signals of that scope will be read from the fsdb file 
        and the tree callback function will be activated for each signal. 
        When the tree callback function is activated, the callback type will 
        be FSDB_TREE_CBT_VAR. Application should cast the callback data to 
        fsdbTreeCBDataVar in order to get the information of the returned signal. 
        After the function returns, all signals of the desired scope will all 
        be retrieved.
    Arguments
        var_start_log_uoff:
            Start logical unsigned offset of the signals belonging to the 
            desired scope.
        real_file_obj_idx:
            Specify the real file object index in virtual file. The information 
            about real_file_obj_idx can be obtained from the call back data
            (i.e. ffrTreeCBDataScope, ffrTreeCBDataUpScope and ffrTreeCBDataBeginTree).
    Return Value
        FSDB_RC_SUCCESS if successful; 
        FSDB_RC_FAILURE otherwise.
    */
    virtual bool_T 
        ffrHasDataTypeDef() = 0;
    /*
    Purpose
        Check if the fsdb file contains data type definition or not.
    Return Value
        TRUE if the fsdb file contains data type definition; 
        FALSE otherwise.
    */
    virtual fsdbRC 
        ffrReadDataTypeDefByBlkIdx(uint_T &blk_idx) = 0;
    /*
    Purpose
        Read the data type definitions in the fsdb file. In FSDB, all data type 
        definitions are stored in data type blocks. The argument blk_idx means 
        data type block index. It is usually zero, which means to read from the 
        first data type block to the last one. If it is not zero, say 1, it 
        means to read from data type block 1 to the last one. At the end of this
         call, argument blk_idx will be updated to indicate the biggest block 
         index it has read so far, which is a useful information when the fsdb 
         file is still growing.
    Arguments
        blk_idx :When calling the API, application needs to supply a 
        value(usually 0) to indicate which data type block to begin to read data
         type definitions. After the API finishes, it contains the biggest index
          of data type block the fsdb reader has read so far.
    Return Value
        FSDB_RC_SUCCESS if successful; 
        FSDB_RC_FAILURE otherwise.
    */
    virtual fsdbRC 
        ffrReadDataTypeDefByBlkIdx2(fsdbTreeCBFunc tree_cb_func,
                   void *tree_client_data, uint_T &blk_idx) = 0;
    /*
    Purpose
        This function is similar to ffrObject::ffrReadDataTypeDefByBlkIdx with 
        the difference that caller could ask the function to use tree_cb_func 
        and tree_client_data instead of that of the fsdb reader object to handle
        data type definitions. This is useful when caller needs to handle data 
        type definitions in its own way but does not want to change the tree 
        function pointer and tree callback data of the fsdb reader object. Note 
        that the tree function pointer and tree callback data of the fsdb reader
        object remains intact after the function finishes.
    Arguments
        tree_cb_func
            Tree callback function pointer that will be used in this function.
        tree_client_data
            Tree client data that will be used in this function.
        blk_idx
            When calling the API, application needs to supply a value(usually 0)
             to indicate which data type block to begin to read data type 
             definitions. After the API finishes, it contains the biggest index 
             of data type block the fsdb reader has read so far.
    Return Value
        FSDB_RC_SUCCESS if successful; 
        FSDB_RC_FAILURE otherwise.
    */

    virtual fsdbRC 
        ffrGetVHDLIntegerXValue(int &ret_xprop_value) = 0;

    virtual bool_T 
        ffrEnableXADummyTopCB() = 0;
    
    virtual bool_T 
        ffrHasScopeTree() = 0;
        
    virtual fsdbRC 
        ffrReadScopeTree() = 0;
        
    virtual fsdbRC 
        ffrReadScopeTree2(fsdbTreeCBFunc tree_cb_func, 
                   void *tree_client_data) = 0;

    virtual void   
        ffrStopTreeTraverse() = 0;
        
    virtual fsdbRC 
        ffrSync(bool_T &is_file_complete) = 0;
    
    virtual fsdbRC 
        ffrSyncSinglet(bool_T &is_file_complete, bool_T &is_tlist_read, bool_T &is_slist_read, 
                          bool_T &is_var_updated,   bool_T &is_dt_updated) = 0;
                          
    virtual fsdbRC 
        ffrSyncHeader(bool_T &is_file_complete) = 0;
        
    virtual fsdbRC 
        ffrSyncSingletHeader(bool_T &is_file_complete) = 0;
            
    virtual fsdbRC 
        ffrEditShiftedXTag(fsdbXTag *time_shift) = 0;
    // --------------------------------------------------
    // end read scop/datatype/var API
    //

    //
    // create vc_trvs_handle
    // --------------------------------------------------
    virtual ffrVCTrvsHdl 
        ffrCreateVCTrvsHdl(fsdbVarIdcode var_idcode) = 0;
    /*
    Purpose
        Creates a traverse handle associated with the signal which is specified 
        by its idcode. Application can traverse the value changes of a signal 
        via its traverse handle. If the value changes of a signal need to be 
        traversed in different ways at the same time, application could 
        associate multiple traverse handles with the signal.
    Arguments
        var_idcode : The idcode of the signal that is to be traversed.
    Return Value
        Non-null if successful; null otherwise.
    */


#define ffrCreateVCTraverseHandle ffrCreateVCTrvsHdl

    virtual ffrTimeBasedVCTrvsHdl 
        ffrCreateTimeBasedVCTrvsHdl(uint_T num,
                                fsdbVarIdcode sig_arr[]) = 0;
    /*
    Purpose
        Creates a traverse handle associated with the signals which are 
        specified in sig_arr[]. The number of signals in sig_arr[] is specified 
        by num. Application can traverse the value changes of these signal in a 
        time-based fashion via this traverse handle. See Traversing Value 
        Changes in Time-based fashion section for detail.
    Arguments
        num
            The number of signals in sig_arr[].
        sig_arr
            The idcodes of the signals that are to be traversed.
    Return Value
        Non-null if successful; null otherwise.
    */
    
    // --------------------------------------------------
    // end create vc_trvs_hdl API

    //
    // file time information API
    // --------------------------------------------------
    virtual fsdbRC          
        ffrGetMinFsdbTag64(fsdbTag64 *tag64) = 0;
    /*
    Purpose
        Get the minimum simulation time of the fsdb file in fsdbTag64 format. 
        fsdbTag64 is defined in fsdbShr.h header file. The API usually applies 
        to fsdb files of digital simulation results like Verilog/VHDL.
    Arguments
        tag64
            The minimum simulation time of the fsdb file in fsdbTag64 format.
    Return Value
        FSDB_RC_SUCCESS if successful; 
        FSDB_RC_FAILURE otherwise.
    */
    virtual fsdbRC          
        ffrGetMaxFsdbTag64(fsdbTag64 *tag64) = 0;
    /*
    Purpose
        Get the maxmum simulation time of the fsdb file in fsdbTag64 format. 
        fsdbTag64 is defined in fsdbShr.h header file. The API usually applies 
        to fsdb files of digital simulation results like Verilog/VHDL.
    Arguments
        tag64
            The maxmum simulation time of the fsdb file in fsdbTag64 format.
    Return Value
        FSDB_RC_SUCCESS if successful; 
        FSDB_RC_FAILURE otherwise.
    */
    virtual fsdbRC          
        ffrGetMinFsdbTagFloat(fsdbTagFloat *tag_float) = 0;

    virtual fsdbRC          
        ffrGetMaxFsdbTagFloat(fsdbTagFloat *tag_float) = 0;
        
    virtual fsdbRC          
        ffrGetMinFsdbTagDouble(fsdbTagDouble *tag_double) = 0;

    virtual fsdbRC          
        ffrGetMaxFsdbTagDouble(fsdbTagDouble *tag_double) = 0;
                    
    virtual fsdbXTagType    
        ffrGetXTagType(void) const = 0;
    static fsdbRC   ffrExtractScaleUnit(str_T scale_unit, uint_T &digit, char *&unit);
    
    virtual bool_T          
        ffrHasNewTree() = 0;
    // --------------------------------------------------
    // end file time info API
    //

    //
    // var information API
    // --------------------------------------------------
    virtual uint_T  
        ffrGetVCCount(fsdbVarIdcode var_idcode) = 0;
    /*
    Purpose
        Return the number of the loaded value changes of the signal specified 
        by var_idcode. Note that only the value changes that are loaded 
        into memory are counted. Whether ffrObject::ffrSetViewWindow is called 
        or not also affects the result of this API. This API only supports 
        FSDB file version 2.x and afterward.
    Arguments
        var_idcode : The idcode of the signal.
    Return Value
        The number of the loaded value changes of the signal. It returns 0 if
        1. The signal does not have any value change.
        2. var_idcode is invalid.
        3. The fsdb file is created by a FSDB file version < 2.0
    */
    virtual bool_T  
        ffrHasGlitch(fsdbVarIdcode idcode) = 0;
    
    virtual void
        ffrSetShowGlitchStartXtag(bool_T is_set, fsdbXTag start_xtag) = 0;

    virtual void
        ffrSetShowGlitchEndXtag(bool_T is_set, fsdbXTag end_xtag) = 0;
    /*
    Purpose
        Checks if the loaded value changes of the signal specified by idcode 
        contain any glitch. Note that it's the signal's "loaded value changes" 
        instead of "value changes in the whole file" that are checked. Whether 
        ffrObject::ffrSetViewWindow is called or not affects the result of this 
        API.
    Arguments
        var_idcode : The idcode of the signal to be tested.
    Return Value
        TRUE if the loaded value changes of the signal contain glitches or the 
        FSDB file is created by a Debussy older than 5.0; FALSE otherwise.
    */
    virtual fsdbRC  
        ffrGetVarInfoByVarIdcode(fsdbVarIdcode var_idcode,
                         ffrVarInfo **ret_var_info) = 0;
    // --------------------------------------------------
    // end var info API
    //

    //
    // Transaction API, only used for fsdb file 2.x version
    // --------------------------------------------------
    virtual fsdbRC  
        ffrGetTransInfo(fsdbTransId trans_id, ffrTransInfo *&info) = 0;
    /*
    Purpose
        Gets the values of the transaction specified by trans_id. The result 
        will be filled in info by fsdb reader. See ffrVCIterOne::ffrGetTransInfo
        for detail.
    Arguments
        trans_id : The id of the transaction.
        info : An ffrTransInfo that will hold the result returned by fsdb 
            reader. See ffrTransInfo in Transaction Traversal section for detail.
    Return Value
        FSDB_RC_SUCCESS on success; 
        FSDB_RC_FAILURE otherwise.
    */
    virtual fsdbRC  
        ffrGetRelationName(fsdbRelationHdl relation_hdl, str_T &name) = 0;
    /*
    Purpose
        Return the name of the relation specified by relation_hdl.
    Arguments
        relation_hdl : Handle of the relation
        name : Name of the relation fsdb reader returns
    Return Value
        FSDB_RC_SUCCESS if successful; 
        FSDB_RC_RELATION_NOT_FOUND if relation_hdl is invalid.
    */

    virtual ffrAttrElem*    
        ffrGetAttrByHdl(fsdbAttrHdl hdl) = 0;
    /*
    Purpose
        Return attribute datatype detail information by attribute_hdl.
    Arguments
        fsdbAttrHdl : Handle of the attribute
    Return Value
        Non-null if successful; null otherwise.
    */


    // --------------------------------------------------
    // end Transaction API
    //

    //
    // warning message call back function API
    // --------------------------------------------------
    static void ffrRegisterWarnCBFunc(fsdbMsgCBFunc cb);
    static void ffrRegisterErrorCBFunc(fsdbMsgCBFunc cb);
    static void ffrRegisterInfoCBFunc(fsdbMsgCBFunc cb);

    // those are used for fsdb file 2.3 version and afterward
    static void ffrWarnSuppress(bool_T if_suppress);
    static void ffrInfoSuppress(bool_T if_suppress);
    // --------------------------------------------------
    // end warning message call back fucniotn API
    //

    //
    // view window set, only used for fsdb file 2.x version
    // --------------------------------------------------
    virtual fsdbRC  
        ffrSetViewWindow(fsdbXTag *start_xtag, fsdbXTag *close_xtag) = 0;

    virtual fsdbRC  
        ffrResetViewWindow(fsdbXTag *start_xtag, fsdbXTag *close_xtag) = 0;
    
    virtual bool_T  
        ffrIsViewWindowSet() const = 0;
    // --------------------------------------------------
    // end view window set API
    //

    //
    // computed VC / Middle Var API, only used for fsdb file 2.x version
    // --------------------------------------------------
    virtual fsdbRC  
        ffrInsertComputedVC(fsdbVarIdcode var_idcode,
                                    fsdbXTag *time, byte_T *vc) = 0;

    virtual bool_T  
        ffrHasBeenComputed(fsdbVarIdcode var_idcode) = 0;

    virtual bool_T  
        ffrIsComputedVar(fsdbVarIdcode var_idcode) = 0;

    virtual bool_T  
        ffrIsMiddleOrComputedVar(fsdbVarIdcode var_idcode) = 0;

    virtual bool_T  
        ffrIsMiddleVar(fsdbVarIdcode var_idcode) = 0;

    // those are used for fsdb file 2.3 version and afterward
    virtual fsdbRC  
        ffrInsertComputedPropVC(fsdbVarIdcode var_idcode,
                fsdbXTag *begin_time, fsdbXTag *end_time,
                str_T result_str) = 0;

    virtual fsdbRC  
        ffrInsertComputedTransVC(fsdbVarIdcode var_idcode,
                fsdbXTag *begin_time, fsdbXTag *end_time,
                str_T label, fsdbAttrHdlVal attr_arr[],
                ushort_T attr_count, fsdbTransStatus status) = 0;

    virtual fsdbRC  
        ffrInsertComputedMDAVC(fsdbVarIdcode var_idcode,
                            fsdbXTag *time, int address, byte_T *val) = 0;

    virtual fsdbRC  
        ffrEnableOEMTimeBasedComputedVCScheme() = 0;
    // --------------------------------------------------
    // end computed VC / Middle Var API
    //

    //
    // EVcd API, only used for fsdb file 2.x version
    // --------------------------------------------------
    virtual bool_T          
        ffrIsEvcd() = 0;
    static fsdbRC   ffrGetEvcdPortStrength(byte_T *evcd_bit_val,
                    fsdbBitType *ret_port_val,
                    fsdbStrengthType *ret_s0, fsdbStrengthType *ret_s1);
    // --------------------------------------------------
    // end EVcd API
    //

    //
    // Support Info API, only used for fsdb file 2.x version
    // --------------------------------------------------
    virtual bool_T  
        ffrHasSupportInfo() = 0;

    virtual fsdbRC  
        ffrGetSupportInfo(fsdbSupInfo &support_info) = 0;
    // --------------------------------------------------
    // end Support Info API

    //
    // DumperEnd API, only used for fsdb file 2.x version
    // --------------------------------------------------
    virtual str_T   
        ffrGetDumperEnvVarVal(str_T target_env_var, str_T dump_env) = 0;

    virtual fsdbRC  
        ffrGetAllDumperEnvVarVal(str_T env_var) = 0;
    // --------------------------------------------------
    // end DumperEnd API

    //
    // LoopMarker API, only used for fsdb file 2.x version
    // --------------------------------------------------
    virtual bool_T 
        ffrHasLoopMarkerVar() = 0;

    virtual fsdbRC 
        ffrGetLoopMarkerArray(uint_T &count,
                fsdbVarIdcode *&idcode_array) = 0;
    // --------------------------------------------------
    // end LoopMarker API
    //

    //
    // TriggerPoint API, only used for fsdb file 2.x version
    // --------------------------------------------------
    virtual bool_T  
        ffrHasTriggerPointVar() = 0;

    virtual fsdbRC  
        ffrGetTriggerPointVarIdcode(fsdbVarIdcode &idcode) = 0;
    // --------------------------------------------------
    // end TriggerPoint API
    //

    //
    // DumpOff API, only used for fsdb file 2.x version
    // --------------------------------------------------
    virtual bool_T  
        ffrHasDumpOffRange() const = 0;

    virtual fsdbRC  
        ffrGetDumpOffRange(uint_T &count,
                        fsdbDumpOffRange *&range) = 0;
    // --------------------------------------------------
    // end DumpOff API
    //

    //
    // OEM API, only used for fsdb file 2.3 version and afterward
    // --------------------------------------------------
    static ffrOEMFuncTbl*   ffrGetOEMFuncTbl();
    static void             ffrSetOEMActiveObj(ffrObject *obj);
    static ffrObject*       ffrGetOEMActiveObj();

    // For reason code
    virtual fsdbRC                  
        ffrInsertOneStringOnTheFly(str_T str, uint_T &ret_str_idx) = 0;

    virtual fsdbRC                  
        ffrInsertReasonCodeDescription(uint_T idx_reasoncode, uint_T idx_description) = 0;

    virtual fsdbRC                  
        ffrInsertReasonCodeByIdcode(fsdbVarIdcode idcode, 
                                int bit_width, uint_T* reason_code_arr) = 0;
    // --------------------------------------------------
    // end OEM API
    //

    //
    // Property statistic, only used for fsdb file 2.3 version and afterward
    // --------------------------------------------------
    virtual fsdbRC  
        ffrGetPropVarStatInfo(fsdbVarIdcode var_idcode,
                  ffrPropVarStatInfo &ret_info) = 0;
            
    virtual void    
        ffrSetLoadNavDBOnlyOnce(bool_T flag) = 0;
    // --------------------------------------------------
    // end Property API
    //

    //*************************************
    // belows are not suggest use API
    //**************************************
    //
    // file API
    // --------------------------------------------------
    static fsdbRC   ffrCheckFile(str_T file_name);
    static fsdbRC   ffrCheckFSDB(str_T fsdb_fname);

    static str_T    ffrGetIdent();
    // The ident used to get machine and compiler information.

    static void     ffrInitialize();
    static void     ffrSetExitOnMallocFail(bool_T flag);
    // --------------------------------------------------
    // end file API
    //
    
    //
    // must open file before any implementation
    // --------------------------------------------------
    static ffrObject*   ffrOpen(str_T fname,
                fsdbTreeCBFunc tree_cb_func,
                void *tree_client_data,
                ffrOpenOption opt = FFR_OPEN_OPT_DFT,
                ffrCmprsFileArg *file_arg = NULL);

    static ffrObject*   ffrOpen2(str_T fname,
                fsdbTreeCBFunc tree_cb_func,
                void *tree_client_data,
                ffrOpenOption opt = FFR_OPEN_OPT_DFT,
                ffrCmprsFileArg *file_arg = NULL,
                bool_T isSharedObj = 0);
    // --------------------------------------------------
    // end opend file API
    //

    // -------------------------------------------------
    //  Vtrl File API
    //
    
    virtual bool_T 
        ffrIsVrtlFile(str_T fd_name) = 0;

    virtual fsdbVrtlFileType    
        ffrGetVrtlFileType() = 0;   
    // ------------------------------------------------
    // end Vtrl API
    //
    // Multi sweep, only can use in version 1.6
    // --------------------------------------------------
    virtual  fsdbRC  
        ffrRetrieveSweepConditionBySweepNum (
                            ushort_T    sweep_num,
                            ushort_T    &ret_condition_count,
                            fsdbMultiSweepCondition  *&ret_sweep_conditions ) = 0;
    // --------------------------------------------------
    // end multi sweep API
    //

    // Get latest supported FSDB version
    // --------------------------------------------------
     virtual str_T  
        ffrGetLatestFSDBVersion() = 0;
    // --------------------------------------------------
    // end  get latest supported FSDB version API
    //
    
    virtual fsdbRC  
        ffrNoChangeEpicVarName() = 0;

    virtual fsdbRC  
        ffrFreeNavDB() = 0;
    
    /*
    purpose:
        Free all the navdb used by those signals which are in signal list. 
    
    return
        FSDB_RC_FAILURE on failure
        FSDB_RC_SUCCESS on success
    */
    
    virtual short  
        ffrGetMTThreadCount() = 0; /* STAR9000843962*/
    
    /*
    purpose:
        Get the thread count after ffrOpen3().
    return
        >1 on multi-thread
        =1 on single-thread
        0  on failure
    */

    static const char*
    ffrGetVersion();
    /*
    purpose:
        To Get the library version
    */

    //
    // Get reason code API
    // --------------------------------------------------

    virtual bool_T  
        ffrHasReasonCode() = 0;
    virtual fsdbRC  
	ffrGetReasonCodeByIdcode(fsdbVarIdcode idcode, int &num, 
            str_T* &reason_code, str_T* &description, bool_T need_description, uint_T** idx_arr = NULL) = 0;

    /*
    Purpose
        Gets the reason code by idcode.
    Arguments
        idcode      : The id of the var.
        num         : If the var don't have reason code, it will get 0.
                      Else, the bit width of this var.
        reason_code : It will save string of array 
                      If the var is vector and one of bit don't have reason code, 
                      array of that bit will be NULL.

    Return Value
        FSDB_RC_SUCCESS on success; 
        FSDB_RC_FAILURE otherwise.
    */
    virtual fsdbRC  
        ffrGetReasonCodesDescription(int &num, str_T* &reason_code, str_T* &description) = 0;
    /*
    Purpose
        Gets all reason code.
    Arguments
        num         : Total reason code number.
        reason_code : It will save as string of array 
        description : It will save as string of array.

    Return Value
        FSDB_RC_SUCCESS on success; 
        FSDB_RC_FAILURE otherwise.
    */
    // --------------------------------------------------
    // end reason code API
    //
    
    virtual bool_T  
        ffrIsSignalBased() = 0;
    /*
    Purpose 
        Check this FSDB is signal based or not by check 
        data20->flush_hdr.max_xtag_has_been_reset
    Return Value
        TRUE : this FSDB is signal based
        FALSE : this FSDB is not signal based
    */

    //
    // ---------- End of APIs for 3rd Parties ----------
    //
    
    virtual fsdbRC  
        ffrIsEmptyVarLogUOff(fsdbLUOff *var_start_log_uoff, ushort_T real_file_obj_idx, bool_T &is_empty) = 0;
    /*
    Purpose
        Check the var_start_log_uoff is empty or not for specified idx of virtual file.
    Argument
        var_start_log_uoff: var start log uoff.
        real_file_obj_idx:  idx of virtual file.
        is_empty: this var_start_log_uoff is empty or not.
    Return Value
        FSDB_RC_SUCCESS on success.
        FSDB_RC_FAILURE on otherwise.
    */

    virtual fsdbRC  
        ffrIsEmptyVarLogUOff(fsdbLUOff *var_start_log_uoff, bool_T &is_empty) = 0;
    /*
    Purpose
        Check the var_start_log_uoff is empty or not.
    Argument
        var_start_log_uoff: var start log uoff.
        is_empty: this var_start_log_uoff is empty or not.
    Return Value
        FSDB_RC_SUCCESS on success.
        FSDB_RC_FAILURE on otherwise.
    */

    static bool_T       ffrCheckVersionCompatible(str_T fsdb_fname);
    /*
    Purpose
        Check this FSDB is compatible of FSDB reader
    Return Value
        TRUE : compatible
        FALSE : not compatible
    */
    
    virtual bool_T       
        ffrOver2GIdcode() const = 0;
    /*
    Purpose
        Check the number of idcode in this FSDB over 2G or not
    Return Value
        TRUE : over 2G
        FALSE : not over 2G
    */

    static  void        ffrCopyRight();
    /*
    Purpose
        Showing FSDB reader copyright message. 
    Return Value
        void
    */

#ifdef FSDB_SUPPRESS_COMPILER_WARNING
#include "sysMachine.h"
#endif

#ifdef WIN32
#ifndef NOVAS_FSDB

    // visual c++ compiler put the funtcions which get overloading in the same place in vft
    // this behavior cause FSDB Reader's hidden header mechanishm not work.
    // as a WA for windows, I put the overloading functions here to make external user can use ffrObject correctlly.
    // Please do not call any fucntion here
   
    virtual fsdbRC
        ffrSetViewWindow(fsdbXTag *start_xtag, fsdbXTag *close_xtag,
        bool_T supress_warning) = 0;

    virtual fsdbRC
        ffrSetViewWindow(fsdbXTag *start_xtag, fsdbXTag *close_xtag,
        bool_T supress_warning, bool_T is_merge_with_prev_vw) = 0;

    virtual fsdbRC
        ffrResetViewWindow(fsdbXTag *start_xtag, fsdbXTag *close_xtag,
        bool_T supress_warning) = 0;

    virtual fsdbRC
        ffrInsertComputedVC(fsdbVarIdcode var_idcode,
        fsdbXTag *time, byte_T *vc, ushort_T &vc_obj_idx) = 0;

    virtual fsdbRC
        ffrInsertComputedPropVC(fsdbVarIdcode var_idcode,
        fsdbXTag *begin_time, fsdbXTag *end_time,
        str_T result_str, ushort_T &vc_obj_idx) = 0;

    virtual fsdbRC
        ffrInsertComputedTransVC(fsdbVarIdcode var_idcode,
        fsdbXTag *begin_time, fsdbXTag *end_time,
        str_T label, fsdbAttrHdlVal attr_arr[],
        ushort_T attr_count, fsdbTransStatus status, ushort_T &vc_obj_idx) = 0;

    virtual bool_T
        ffrHasGlitch(void) = 0; // SPSqa34766

    virtual bool_T
        ffrHasGlitch(ffrVarPtr var_ptr) = 0;

    virtual fsdbRC
        ffrGetRelationName(uint_T relation_count,
        fsdbRelationHdl relation_hdl_arr[],
        str_T *&relation_name_arr) = 0;

    virtual fsdbRC
        ffrReadScopeVarTree(ushort_T real_obj_idx) = 0;

    virtual fsdbRC
        ffrReadScopeVarTree2(fsdbTreeCBFunc tree_cb_func,
            void *tree_client_data,
            ushort_T real_obj_idx) = 0;

    virtual fsdbRC
        ffrUnloadSignals(ffrVarPtr var_ptr) = 0;

    virtual fsdbRC
        ffrUnloadSignals(ffrPKVarPtr var_ptr) = 0;

    virtual fsdbRC
        ffrResetSignalList(bool_T bResetSecVar) = 0;
#endif
#endif

#ifdef NOVAS_FSDB
#include "ffrAPI_Int.h"
#include "ffrAPI_R.h"
#include "ffrObjectAPI.h"
#endif

#ifdef FSDB_INTERNAL_HEADER
#include "ffrAPI_Int.h"
#endif

};

#ifdef NOVAS_FSDB
#include "ffrRealAPI.h"
#endif

#endif /* ffrAPI_h */
