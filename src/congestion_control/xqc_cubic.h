/**
 * @copyright Copyright (c) 2022, Alibaba Group Holding Limited
 */

#ifndef _XQC_CUBIC_H_INCLUDED_
#define _XQC_CUBIC_H_INCLUDED_

#include <xquic/xquic_typedef.h>
#include <xquic/xquic.h>
#include "src/transport/xqc_send_ctl.h"
#include "src/transport/xqc_packet_out.h"
#include "src/congestion_control/xqc_ml_model.h"

typedef struct {
    uint32_t        min_cwnd;
    uint64_t        init_cwnd;          /* initial window size in MSS */
    uint64_t        cwnd;               /* current window size in bytes */
    uint64_t        tcp_cwnd;           /* cwnd calculated according to Reno's algorithm */
    uint64_t        tcp_cwnd_cnt;       /* Linear increase counter */
    uint64_t        last_max_cwnd;      /* last max window size */
    uint64_t        ssthresh;           /* slow start threshold */
    uint64_t        bic_origin_point;   /* Wmax origin point */
    uint64_t        bic_K;              /* time period from W growth to Wmax */
    xqc_usec_t      epoch_start;        /* congestion switchover moment, in microseconds */
    xqc_usec_t      min_rtt;
    xqc_usec_t      congestion_recovery_start_time;
    
    /* ML-assisted fields - using modular interface */
    xqc_send_ctl_t         *send_ctl;
    xqc_ml_model_t         *ml_model;   /* ML model handle */
    float                   ml_window[XQC_ML_WINDOW_SIZE][XQC_ML_NUM_FEATURES];
    int                     ml_window_idx;
    int                     ml_sample_count;
    float                   ml_state_probs[XQC_ML_NUM_STATES];
    xqc_usec_t              ml_last_rtt;
    uint8_t                 use_ml;     /* enable ML-assisted loss discrimination */
    
} xqc_cubic_t;

extern const xqc_cong_ctrl_callback_t xqc_cubic_cb;

#endif /* _XQC_CUBIC_H_INCLUDED_ */
