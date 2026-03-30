/**
 * @copyright Copyright (c) 2024, Alibaba Group Holding Limited
 */

#ifndef _XQC_ML_CC_H_INCLUDED_
#define _XQC_ML_CC_H_INCLUDED_

#include <xquic/xquic.h>
#include <xquic/xquic_typedef.h>
#include "src/transport/xqc_send_ctl.h"

#define XQC_ML_CC_WINDOW_SIZE             30
#define XQC_ML_CC_NUM_FEATURES            15
#define XQC_ML_CC_MSS                     (XQC_MSS)
#define XQC_ML_CC_INIT_WIN                (32 * XQC_ML_CC_MSS)
#define XQC_ML_CC_MIN_CWND                (20 * XQC_ML_CC_MSS)

#define XQC_ML_CC_RECOVERY_RTT            3

#define XQC_ML_CC_QU_CONSECUTIVE_THRESHOLD 10
#define XQC_ML_CC_NH_THRESHOLD            0.3f
#define XQC_ML_CC_NH_FREEZE_DURATION      100000
#define XQC_ML_CC_NH_LOSS_HIGH            50.0
#define XQC_ML_CC_NH_LOSS_LOW             10.0
#define XQC_ML_CC_NH_FREEZE_CWND_FACTOR   0.7f

#define XQC_ML_CC_UT_CWND_GAIN            0.6f
#define XQC_ML_CC_QU_CWND_DECREASE        0.95f
#define XQC_ML_CC_EB_CWND_DECREASE        0.75f

#define XQC_ML_CC_QU_QUEUE_K              0.40f
#define XQC_ML_CC_QU_QUEUE_DEADZONE       0.05f
#define XQC_ML_CC_QU_QUEUE_EMA_ALPHA      0.25f
#define XQC_ML_CC_QU_GROWTH_BASE          1.10f
#define XQC_ML_CC_QU_DRAIN_BASE           1.30f
#define XQC_ML_CC_QU_HOLD_GAIN            0.05f
#define XQC_ML_CC_QU_GAMMA_UP             1.5f
#define XQC_ML_CC_QU_GAMMA_DOWN           1.2f
#define XQC_ML_CC_QU_CONFIDENCE_THRESHOLD 0.50f

#define XQC_ML_CC_EB_THRESHOLD            0.60f
#define XQC_ML_CC_MAX_CONSECUTIVE_EB      1

typedef enum {
    XQC_ML_CC_STATE_UT = 0,
    XQC_ML_CC_STATE_QU = 1,
    XQC_ML_CC_STATE_EB = 2,
    XQC_ML_CC_STATE_NH = 3,
    XQC_ML_CC_STATE_NONE = 4
} xqc_ml_cc_state_t;

typedef struct xqc_ml_cc_s {
    xqc_send_ctl_t         *send_ctl;

    float                   cwnd_bytes;
    float                   init_cwnd_bytes;
    float                   min_cwnd_bytes;

    float                   history_window[XQC_ML_CC_WINDOW_SIZE][XQC_ML_CC_NUM_FEATURES];
    int                     window_idx;
    int                     sample_count;

    void                   *onnx_session;
    void                   *onnx_api;
    void                   *queue_onnx_session;

    int                     use_ml_prediction;
    float                   last_prediction;

    xqc_usec_t             congestion_recovery_start_time;
    xqc_usec_t             last_rtt;
    float                   loss_reduction_factor;

    xqc_ml_cc_state_t       last_state;
    float                   state_probs[4];
    int                     qu_consecutive_count;
    xqc_bool_t              is_frozen;
    xqc_usec_t              freeze_start_time;
    float                   frozen_cwnd;
    xqc_bool_t              loss_spike_during_freeze;
    float                   queue_depth_raw;
    float                   queue_depth_ema;
    float                   queue_threshold_k;
    int                     queue_model_enabled;
    int                     queue_model_ready;
    int                     state_model_ready;

    int                     eb_consecutive_count;

} xqc_ml_cc_t;

extern const float xqc_ml_cc_scaler_mean[XQC_ML_CC_NUM_FEATURES];
extern const float xqc_ml_cc_scaler_scale[XQC_ML_CC_NUM_FEATURES];
extern const float xqc_ml_cc_qu_scaler_mean[XQC_ML_CC_NUM_FEATURES];
extern const float xqc_ml_cc_qu_scaler_scale[XQC_ML_CC_NUM_FEATURES];

extern const xqc_cong_ctrl_callback_t xqc_ml_cc_cb;

void xqc_ml_cc_feed_features(void *cong, xqc_usec_t ack_recv_time,
    xqc_usec_t adjusted_rtt, double short_loss_rate, unsigned short_lost_cnt,
    unsigned short_send_cnt, double long_loss_rate, unsigned long_lost_cnt,
    unsigned long_send_cnt, xqc_usec_t response_interval, uint64_t cwnd,
    unsigned pkt_in_fly, uint64_t total_acked_bytes);

void xqc_ml_cc_handle_frozen_state(xqc_ml_cc_t *ml_cc, xqc_usec_t now);

#endif
