/**
 * @copyright Copyright (c) 2024, Alibaba Group Holding Limited
 */

#ifndef _XQC_ML_CC_H_INCLUDED_
#define _XQC_ML_CC_H_INCLUDED_

#include <xquic/xquic.h>
#include <xquic/xquic_typedef.h>
#include "src/transport/xqc_send_ctl.h"

#define XQC_ML_CC_WINDOW_SIZE           10
#define XQC_ML_CC_NUM_FEATURES          18
#define XQC_ML_CC_MSS                   (XQC_MSS)
#define XQC_ML_CC_INIT_WIN              (32 * XQC_ML_CC_MSS)
#define XQC_ML_CC_MIN_CWND              (4 * XQC_ML_CC_MSS)

#define XQC_ML_CC_QUEUE_THRESHOLD_LOSS  70.0f
#define XQC_ML_CC_QUEUE_THRESHOLD_K     80.0f
#define XQC_ML_CC_RECOVERY_RTT          3

typedef struct xqc_ml_cc_s {
    xqc_send_ctl_t         *send_ctl;
    xqc_log_t              *log;

    float                   cwnd_bytes;
    float                   init_cwnd_bytes;
    float                   min_cwnd_bytes;

    float                   history_window[XQC_ML_CC_WINDOW_SIZE][XQC_ML_CC_NUM_FEATURES];
    int                     window_idx;
    int                     sample_count;

    float                   scaler_mean[XQC_ML_CC_NUM_FEATURES];
    float                   scaler_scale[XQC_ML_CC_NUM_FEATURES];

    void                   *onnx_session;
    void                   *onnx_api;

    int                     use_ml_prediction;
    float                   last_prediction;

    xqc_usec_t             congestion_recovery_start_time;
    xqc_usec_t             last_rtt;
    float                   loss_reduction_factor;

} xqc_ml_cc_t;

extern const xqc_cong_ctrl_callback_t xqc_ml_cc_cb;

void xqc_ml_cc_feed_features(void *cong, xqc_usec_t ack_recv_time,
    xqc_usec_t adjusted_rtt, double short_loss_rate, double long_loss_rate,
    xqc_usec_t response_interval, uint64_t cwnd, unsigned pkt_in_fly);

#endif
