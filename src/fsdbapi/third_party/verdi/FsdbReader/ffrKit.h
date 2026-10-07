/* *****************************************************************************
// [ffrKit.h]
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
// You are not permitted to use this FSDB Reader under other agreements, including (without limitation) your Synopsys
// end user software license and maintenance agreement or evaluation agreement.
// ****************************************************************************/
#ifndef ffrKit_h
#define ffrKit_h

#include "fsdbShr.h"

enum ffrOpenOption {
    FFR_OPEN_OPT_DFT	        = 0x00, 
    FFR_OPEN_OPT_X_VC_ONLY      = 0x01,
    FFR_OPEN_OPT_SWITCH         = 0x02,
    FFR_OPEN_OPT_SPLIT          = 0x03, 
    FFR_OPEN_OPT_STITCH         = 0x04, 
    FFR_OPEN_OPT_STREAMLINED    = 0x05
};
typedef enum ffrOpenOption	ffrOpenOption; 

typedef struct ffrMinMaxAnalogValRec	ffrMinMaxAnalogValRec;
struct ffrMinMaxAnalogValRec {
    double      min;    
    double      max;    
};  

typedef class IntervalTree ffrThreadHis;

#ifdef NOVAS_FSDB
#include "ffrKitDS.h"
#include "interval_tree.h"
#include "ffrVCCtnr.h"
#include "kit_util.h"
#include "ffSynchronizer.h"
#endif

typedef struct ffrData		ffrData;
typedef struct ffrData20 	ffrData20;

typedef void (*ffrOverwrittenCBFunc)(str_T overwritten_fname);

/*
** Property Variable's Value (Property Value Change)
*/
struct ffrPropVarValue {
    fsdbXTag	begin_xtag;	// begin time
    fsdbXTag	end_xtag;	// end time
    str_T	result;		// result: success, failure, ...
    fsdbSeqNum  seq_num;	// sequence number
};
typedef struct ffrPropVarValue	ffrPropVarValue;

struct propSourceInfo {
    str_T   fname_line_no;
};
typedef struct propSourceInfo propSourceInfo;

struct ffrPropVarValue2 {
    union xtag_pos {
        fsdbXTag       begin_xtag;     // begin time
        propSourceInfo sourceInfo;
    } info;        
    fsdbXTag    end_xtag;       // end time
    str_T       result;         // result: success, failure, ...
    fsdbSeqNum  seq_num;        // sequence number
    byte_T      val_state;      // value defined in fsdbPropValState
    byte_T      val_source;     // value defined in fsdbPropValSource
    bool_T      is_begin_xtag;  // define whether info union is begin xtag or sourceInfo
    byte_T      unused[1];

    str_T failure_detail;       // failure_detail for failure or no-match 
};
typedef struct ffrPropVarValue2  ffrPropVarValue2;

/*
** Transaction
*/
struct ffrTransInfo {
    fsdbTransId     trans_id;
    ushort_T        attr_count;
    ushort_T        relation_count;
    byte_T          status;
    byte_T          pad[3];
    str_T           label;
    fsdbXTag        begin_time;
    fsdbXTag        end_time;
    ushort_T        *attr_hdl_arr;
    byte_T          **attr_val_arr;
    fsdbRelationHdl *relation_hdl_arr;
    fsdbTransId     *related_trans_arr;
//  below is for internal use only
    uint_T          begin_dc;
    uint_T          begin_gid;
    uint_T          end_dc;
    uint_T          end_gid;
};
typedef struct ffrTransInfo ffrTransInfo;

//
// SPSqa34630 transaction recording
//
typedef struct ffrAttrElem  ffrAttrElem;
struct ffrAttrElem {
    ffrAttrElem         *collision;
    ffrAttrElem         *next;
    str_T               name;
    fsdbAttrDataType    data_type;
    fsdbAttrHdl         hdl;
    int                 larridx;
    int                 rarridx;
    uint_T              byte_count;
    bool_T              hidden;
    char                pad[1];
    ushort_T            meta_info;
    fsdbAttrRadix       radix;
    byte_T              *extra_inf;
};

//  
// Used in data20->attr_elem_hs to store enum-specific info so that the 
// info could be queried with ffrGetAttrByHdl().
//  
struct ffrAttrElem_ExtraInf_Enum {
    ushort_T    literal_count;
    uint_T      val_len;
    byte_T      val_type;       // Type is fsdbEnumValueType
    str_T       *literal_arr;
    byte_T      **val_arr;
};
typedef struct ffrAttrElem_ExtraInf_Enum ffrAttrElem_ExtraInf_Enum;


/*
** MDA(Multiple Dimensional Array)
*/
struct ffrMDAVarValue {
    int     address;
    byte_T  *val;
};
typedef struct ffrMDAVarValue ffrMDAVarValue;

/*
** List & Asslociate List
*/
struct ffrListVarValue {
    int     elem_count; 
    byte_T  **elem_val_arr; 
    byte_T  **key_val_arr;  // only effective when is_associate_array = TRUE
};
typedef struct ffrListVarValue ffrListVarValue;

/*
** Instance 
*/
struct ffrInstVarValue {
    byte_T  **attr_val_arr; 
};
typedef struct ffrInstVarValue ffrInstVarValue;

/*
** Muilt Sweep
*/
struct ffrSweepVarExtraInf {
    ushort_T    condition_count;
    byte_T      pad[2];
    fsdbMultiSweepCondition *condition_array;
    str_T       sweep_var_name; // only for Novas Internal use.
};
typedef struct ffrSweepVarExtraInf     ffrSweepVarExtraInf;

#define FSDB_STAT_PROFILE 0

class IJoinedFile;
class DecompressManager;

#ifdef WIN32
#if WIN32 && !USE_LIB_FFR
#if NFFR_EXPORTS
class __declspec(dllexport) ffrDisk
#else // NFFR_EXPORTS
class __declspec(dllimport) ffrDisk
#endif // NFFR_EXPORTS
#else
class ffrDisk 
#endif //WIN32 && !USE_LIB_FFR
#else // WIN32 
class ffrDisk 
#endif // WIN32 
{

#ifdef NOVAS_FSDB
#include "ffrDiskAPI.h"
#endif

  protected:
    int			magic_num;
    int			byte_order;
    ffrData		*data;
    ffrData20		*data20;
    class ffrSynchronizer     *synchronizer;

    ffrOverwrittenCBFunc overwritten_cb_func;

    bool_T              is_virtual_fsdb;
    IJoinedFile*		m_pIJoinedFile;
    IJoinedFile*		m_pMergedFile;

    ffrDisk(str_T file_name, bool_T is_child_thread, IJoinedFile* pIJoinedFile, bool_T is_rtd, bool_T is_file_in_zwd, bool_T is_ignore_vpdgate);
    ffrDisk();
    ~ffrDisk();
};

#endif /* ffrKit_h */
