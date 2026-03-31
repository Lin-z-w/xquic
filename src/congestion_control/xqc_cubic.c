/**
 * @copyright Copyright (c) 2022, Alibaba Group Holding Limited
 * 
 * CUBIC based on https://tools.ietf.org/html/rfc8312
 * Enhanced with ML-based loss discrimination (Cubic-ML)
 */

#include "src/congestion_control/xqc_cubic.h"
#include "src/common/xqc_config.h"
#include "src/common/xqc_time.h"
#include "src/congestion_control/xqc_ml_model.h"
#include <math.h>
#include <string.h>

/* ML feature extraction helpers */
static float
xqc_cubic_clamp_float(float value, float min_value, float max_value)
{
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

static void
xqc_cubic_build_ml_features(xqc_cubic_t *cubic, float *features,
    xqc_usec_t adjusted_rtt, double short_loss_rate, unsigned short_lost_cnt,
    unsigned short_send_cnt, double long_loss_rate, unsigned long_lost_cnt,
    unsigned long_send_cnt, xqc_usec_t response_interval, uint64_t cwnd,
    unsigned pkt_in_fly)
{
    float prev_rtt = 0.0f;
    float prev_cwnd = 0.0f;
    float eps = 1e-8f;
    int prev_idx;

    if (cubic->ml_sample_count > 0) {
        prev_idx = (cubic->ml_window_idx - 1 + XQC_ML_WINDOW_SIZE) % XQC_ML_WINDOW_SIZE;
        prev_rtt = cubic->ml_window[prev_idx][0];
        prev_cwnd = cubic->ml_window[prev_idx][8];
    }

    features[0] = (float)adjusted_rtt;
    features[1] = (float)short_loss_rate;
    features[2] = (float)short_lost_cnt;
    features[3] = (float)short_send_cnt;
    features[4] = (float)long_loss_rate;
    features[5] = (float)long_lost_cnt;
    features[6] = (float)long_send_cnt;
    features[7] = (float)response_interval;
    features[8] = (float)cwnd;
    features[9] = (float)pkt_in_fly;
    features[10] = features[0] - prev_rtt;
    features[11] = features[8] / (prev_cwnd + eps);
    features[11] = xqc_cubic_clamp_float(features[11], 0.1f, 10.0f);
    features[12] = features[9] / (features[8] + eps);
    features[12] = xqc_cubic_clamp_float(features[12], 0.0f, 2.0f);
    features[13] = features[3] / (features[7] / 1000.0f + eps);
    features[13] = xqc_cubic_clamp_float(features[13], 0.0f, 10000.0f);
    features[14] = features[1] - features[4];
    features[14] = xqc_cubic_clamp_float(features[14], -100.0f, 100.0f);
}

static void
xqc_cubic_update_ml_window(xqc_cubic_t *cubic, float *features)
{
    int i;
    float *row = cubic->ml_window[cubic->ml_window_idx];
    for (i = 0; i < XQC_ML_NUM_FEATURES; i++) {
        row[i] = features[i];
    }
    cubic->ml_window_idx = (cubic->ml_window_idx + 1) % XQC_ML_WINDOW_SIZE;
    cubic->ml_sample_count++;
}

/* Run ML inference using the modular interface */
static xqc_ml_state_t
xqc_cubic_run_ml_inference(xqc_cubic_t *cubic)
{
    xqc_ml_output_t output;
    xqc_int_t ret;
    int start_idx, t;
    float window[XQC_ML_WINDOW_SIZE][XQC_ML_NUM_FEATURES];

    if (cubic->ml_sample_count < XQC_ML_WINDOW_SIZE || cubic->ml_model == NULL) {
        /* Not enough samples or model not ready - use default */
        return XQC_ML_STATE_UT;
    }

    /* Reorder window to chronological order for inference */
    start_idx = (cubic->ml_window_idx - XQC_ML_WINDOW_SIZE + XQC_ML_WINDOW_SIZE) % XQC_ML_WINDOW_SIZE;
    for (t = 0; t < XQC_ML_WINDOW_SIZE; t++) {
        int src_idx = (start_idx + t) % XQC_ML_WINDOW_SIZE;
        memcpy(window[t], cubic->ml_window[src_idx], sizeof(float) * XQC_ML_NUM_FEATURES);
    }

    ret = xqc_ml_model_infer(cubic->ml_model, window, &output, 
                             cubic->send_ctl ? cubic->send_ctl->ctl_conn->log : NULL);
    
    if (ret != XQC_OK || !output.valid) {
        return XQC_ML_STATE_UT;  /* Fallback to conservative behavior */
    }

    /* Copy probabilities for debugging */
    memcpy(cubic->ml_state_probs, output.state_probs, sizeof(cubic->ml_state_probs));

    return output.predicted_state;
}

#define XQC_CUBIC_FAST_CONVERGENCE  1
#define XQC_CUBIC_MSS               XQC_MSS
#define XQC_CUBIC_BETA              718     /* 718/1024=0.7 */
#define XQC_CUBIC_BETA_SCALE        1024
#define XQC_CUBIC_C                 410     /* 410/1024=0.4 */
#define XQC_CUBE_SCALE              40u     /* 2^40=1024 * 1024^3 */
#define XQC_CUBIC_TIME_SCALE        10u
#define XQC_CUBIC_MAX_SSTHRESH      0xFFFFFFFF

#define XQC_CUBIC_MIN_WIN           (4 * XQC_CUBIC_MSS)
#define XQC_CUBIC_MAX_MIN_WIN       (16 * XQC_CUBIC_MSS)
#define XQC_CUBIC_MAX_INIT_WIN      (100 * XQC_CUBIC_MSS)
#define XQC_CUBIC_INIT_WIN          (32 * XQC_CUBIC_MSS)

const static uint64_t xqc_cube_factor =
    (1ull << XQC_CUBE_SCALE) / XQC_CUBIC_C / XQC_CUBIC_MSS;

/*
 * Compute congestion window to use.
 * W_cubic(t) = C*(t-K)^3 + W_max (Eq. 1)
 * K = cubic_root(W_max*(1-beta_cubic)/C) (Eq. 2)
 */
static void
xqc_cubic_update(void *cong_ctl, uint32_t acked_bytes, xqc_usec_t now)
{
    xqc_cubic_t    *cubic = (xqc_cubic_t *)(cong_ctl);
    uint64_t        t;
    uint64_t        offs;
    uint64_t        delta, bic_target;

    /* First ACK after a loss event. */
    if (cubic->epoch_start == 0) {
        cubic->epoch_start = now;

        if (cubic->cwnd >= cubic->last_max_cwnd) {
            cubic->bic_K = 0;
            cubic->bic_origin_point = cubic->cwnd;
        } else {
            cubic->bic_K = cbrt(xqc_cube_factor * (cubic->last_max_cwnd - cubic->cwnd));
            cubic->bic_origin_point = cubic->last_max_cwnd;
        }
    }

    t = (now + cubic->min_rtt - cubic->epoch_start) << XQC_CUBIC_TIME_SCALE;
    t /= XQC_MICROS_PER_SECOND;

    if (t < cubic->bic_K) {
        offs = cubic->bic_K - t;
    } else {
        offs = t - cubic->bic_K;
    }

    delta = (XQC_CUBIC_C * offs * offs * offs) >> XQC_CUBE_SCALE;
    delta *= XQC_CUBIC_MSS;

    if (t < cubic->bic_K) {
        bic_target = cubic->bic_origin_point - delta;
    } else {
        bic_target = cubic->bic_origin_point + delta;
    }

    bic_target = xqc_min(bic_target, cubic->cwnd + acked_bytes / 2);
    bic_target = xqc_max(cubic->tcp_cwnd, bic_target);

    if (bic_target == 0) {
        bic_target = cubic->init_cwnd;
    }

    cubic->cwnd = bic_target;
}

static int
xqc_cubic_in_congestion_recovery(void *cong_ctl, xqc_usec_t sent_time)
{
    xqc_cubic_t *cubic = (xqc_cubic_t*)(cong_ctl);
    return sent_time <= cubic->congestion_recovery_start_time;
}

size_t
xqc_cubic_size()
{
    return sizeof(xqc_cubic_t);
}

static void
xqc_cubic_init(void *cong_ctl, xqc_send_ctl_t *ctl_ctx, xqc_cc_params_t cc_params)
{
    xqc_cubic_t *cubic = (xqc_cubic_t *)(cong_ctl);
    xqc_ml_model_config_t ml_config;

    cubic->init_cwnd = XQC_CUBIC_INIT_WIN;
    cubic->min_cwnd = XQC_CUBIC_MIN_WIN;

    if (cc_params.customize_on) {
        cc_params.min_cwnd *= XQC_CUBIC_MSS;
        cc_params.init_cwnd *= XQC_CUBIC_MSS;
        cubic->init_cwnd =
                cc_params.init_cwnd >= XQC_CUBIC_MIN_WIN && cc_params.init_cwnd <= XQC_CUBIC_MAX_INIT_WIN ?
                cc_params.init_cwnd : XQC_CUBIC_INIT_WIN;
        cubic->min_cwnd = 
                cc_params.min_cwnd >= XQC_CUBIC_MIN_WIN && cc_params.min_cwnd <= XQC_CUBIC_MAX_MIN_WIN ?
                cc_params.min_cwnd : XQC_CUBIC_MIN_WIN;
    }

    cubic->epoch_start = 0;
    cubic->cwnd = cubic->init_cwnd;
    cubic->tcp_cwnd = cubic->init_cwnd;
    cubic->tcp_cwnd_cnt = 0;
    cubic->last_max_cwnd = cubic->init_cwnd;
    cubic->ssthresh = XQC_CUBIC_MAX_SSTHRESH;
    cubic->congestion_recovery_start_time = 0;
    
    /* Initialize ML fields */
    cubic->send_ctl = ctl_ctx;
    cubic->ml_window_idx = 0;
    cubic->ml_sample_count = 0;
    cubic->ml_last_rtt = 0;
    cubic->ml_model = NULL;
    cubic->use_ml = 0;
    memset(cubic->ml_window, 0, sizeof(cubic->ml_window));
    memset(cubic->ml_state_probs, 0, sizeof(cubic->ml_state_probs));

    /* Initialize NH freeze state */
    cubic->is_frozen = XQC_FALSE;
    cubic->freeze_start_time = 0;
    cubic->frozen_cwnd = 0;
    cubic->loss_spike_during_freeze = XQC_FALSE;

    /* Load ML model for intelligent loss discrimination if enabled */
    if (cc_params.customize_on && cc_params.cubic_use_ml) {
        cubic->use_ml = 1;
        memset(&ml_config, 0, sizeof(ml_config));
        ml_config.model_path = XQC_ONNX_MODEL_PATH;
        ml_config.type = XQC_ML_MODEL_STATE;
        ml_config.use_scaler = 1;
        ml_config.scaler_mean = xqc_ml_scaler_mean;
        ml_config.scaler_scale = xqc_ml_scaler_scale;

        cubic->ml_model = xqc_ml_model_load(&ml_config,
            ctl_ctx ? ctl_ctx->ctl_conn->log : NULL);
    }
}


/* Check and handle NH freeze state exit conditions */
static void
xqc_cubic_check_freeze_state(xqc_cubic_t *cubic, xqc_usec_t now)
{
    xqc_usec_t elapsed;
    double short_loss_rate = 0.0;
    xqc_send_ctl_t *ctl = cubic->send_ctl;

    if (!cubic->is_frozen) {
        return;
    }

    elapsed = now - cubic->freeze_start_time;

    if (ctl && ctl->ctl_recent_send_count[0] > 0) {
        short_loss_rate = (double)ctl->ctl_recent_lost_count[0]
            / ctl->ctl_recent_send_count[0] * 100.0;
    }

    /* 100ms observation window logic */
    if (elapsed < XQC_CUBIC_ML_NH_FREEZE_DURATION) {
        if (short_loss_rate > XQC_CUBIC_ML_NH_LOSS_HIGH) {
            cubic->loss_spike_during_freeze = XQC_TRUE;
        }
    } else {
        if (!cubic->loss_spike_during_freeze) {
            /* elapsed >= 100ms and no loss spike - exit freeze */
            cubic->is_frozen = XQC_FALSE;
            xqc_log(cubic->send_ctl ? cubic->send_ctl->ctl_conn->log : NULL,
                    XQC_LOG_DEBUG,
                    "|cubic|freeze_exit|elapsed:%ui|no_loss_spike|",
                    (uint32_t)elapsed);
        } else if (short_loss_rate < XQC_CUBIC_ML_NH_LOSS_LOW) {
            /* loss spike was true but now below threshold - exit freeze */
            cubic->is_frozen = XQC_FALSE;
            cubic->loss_spike_during_freeze = XQC_FALSE;
            xqc_log(cubic->send_ctl ? cubic->send_ctl->ctl_conn->log : NULL,
                    XQC_LOG_DEBUG,
                    "|cubic|freeze_exit|elapsed:%ui|loss_below_threshold|",
                    (uint32_t)elapsed);
        }
    }
}

static void
xqc_cubic_on_lost(void *cong_ctl, xqc_usec_t lost_sent_time)
{
    xqc_cubic_t *cubic = (xqc_cubic_t*)(cong_ctl);

    cubic->tcp_cwnd_cnt = 0;

    if (xqc_cubic_in_congestion_recovery(cong_ctl, lost_sent_time)) {
        return;
    }

    /* In NH freeze state: don't call model, don't change cwnd on loss */
    if (cubic->use_ml && cubic->is_frozen) {
        return;
    }

    /* ML-assisted loss discrimination:
     * Decrease cwnd when state is QU (Queue) or EB (Exceeded Bandwidth)
     * UT/NH states: don't reduce on loss
     */
    if (cubic->use_ml && cubic->ml_model != NULL) {
        xqc_ml_state_t ml_state = xqc_cubic_run_ml_inference(cubic);
        int should_decrease = (ml_state == XQC_ML_STATE_QU || ml_state == XQC_ML_STATE_EB);

        xqc_log(cubic->send_ctl ? cubic->send_ctl->ctl_conn->log : NULL,
                XQC_LOG_REPORT,
                "|cubic|state_update|state:%s|prob_ut:%.3f|prob_qu:%.3f|prob_eb:%.3f|prob_nh:%.3f|cwnd:%ui|samples:%d|action:%s|",
                xqc_ml_state_to_str(ml_state),
                cubic->ml_state_probs[0],
                cubic->ml_state_probs[1],
                cubic->ml_state_probs[2],
                cubic->ml_state_probs[3],
                (uint32_t)cubic->cwnd,
                cubic->ml_sample_count,
                should_decrease ? "decrease" : "skip_decrease");

        /* Decrease for QU or EB state */
        if (!should_decrease) {
            return;
        }
    }

    cubic->congestion_recovery_start_time = xqc_monotonic_timestamp();
    cubic->epoch_start = 0;

    if (XQC_CUBIC_FAST_CONVERGENCE && cubic->cwnd < cubic->last_max_cwnd) {
        cubic->last_max_cwnd = cubic->cwnd * (XQC_CUBIC_BETA_SCALE + XQC_CUBIC_BETA) / (2 * XQC_CUBIC_BETA_SCALE);
    } else {
        cubic->last_max_cwnd = cubic->cwnd;
    }

    cubic->cwnd = cubic->cwnd * XQC_CUBIC_BETA / XQC_CUBIC_BETA_SCALE;
    cubic->cwnd = xqc_max(cubic->cwnd, cubic->min_cwnd);
    cubic->tcp_cwnd = cubic->cwnd;
    cubic->ssthresh = cubic->cwnd;
}


static void
xqc_cubic_on_ack(void *cong_ctl, xqc_packet_out_t *po, xqc_usec_t now)
{
    xqc_cubic_t *cubic = (xqc_cubic_t *)(cong_ctl);
    xqc_usec_t  sent_time = po->po_sent_time;
    uint32_t    acked_bytes = po->po_used_size;
    xqc_usec_t  rtt = now - sent_time;

    /* ML-assisted Cubic logic */
    if (cubic->use_ml && cubic->send_ctl != NULL) {
        /* Check freeze state exit conditions first */
        xqc_cubic_check_freeze_state(cubic, now);

        /* If frozen, don't call model, keep cwnd unchanged */
        if (cubic->is_frozen) {
            cubic->cwnd = cubic->frozen_cwnd;
            xqc_log(cubic->send_ctl->ctl_conn->log, XQC_LOG_DEBUG,
                    "|cubic|frozen|cwnd:%ui|", (uint32_t)cubic->cwnd);
            return;
        }

        /* Not frozen: collect features and run inference */
        float features[XQC_ML_NUM_FEATURES];
        xqc_send_ctl_t *ctl = cubic->send_ctl;
        double short_loss_rate = 0.0;
        double long_loss_rate = 0.0;
        unsigned short_lost_cnt = 0;
        unsigned short_send_cnt = 0;
        unsigned long_lost_cnt = 0;
        unsigned long_send_cnt = 0;

        if (ctl->ctl_recent_send_count[0] > 0) {
            short_lost_cnt = ctl->ctl_recent_lost_count[0];
            short_send_cnt = ctl->ctl_recent_send_count[0];
            short_loss_rate = (double)short_lost_cnt / short_send_cnt * 100.0;
        }
        if (ctl->ctl_recent_send_count[0] + ctl->ctl_recent_send_count[1] > 0) {
            long_lost_cnt = ctl->ctl_recent_lost_count[0] + ctl->ctl_recent_lost_count[1];
            long_send_cnt = ctl->ctl_recent_send_count[0] + ctl->ctl_recent_send_count[1];
            long_loss_rate = (double)long_lost_cnt / long_send_cnt * 100.0;
        }

        xqc_cubic_build_ml_features(cubic, features,
            rtt,
            short_loss_rate,
            short_lost_cnt,
            short_send_cnt,
            long_loss_rate,
            long_lost_cnt,
            long_send_cnt,
            0,  /* response_interval - simplified */
            cubic->cwnd,
            ctl->ctl_bytes_in_flight / XQC_MSS
        );

        xqc_cubic_update_ml_window(cubic, features);
        cubic->ml_last_rtt = rtt;

        /* Check for NH state and enter freeze if needed */
        if (cubic->ml_model != NULL && cubic->ml_sample_count >= XQC_ML_WINDOW_SIZE) {
            xqc_ml_state_t ml_state = xqc_cubic_run_ml_inference(cubic);

            /* Enter NH freeze state if prob[NH] > 0.3 */
            if (cubic->ml_state_probs[XQC_ML_STATE_NH] > XQC_CUBIC_ML_NH_THRESHOLD) {
                cubic->frozen_cwnd = cubic->cwnd * XQC_CUBIC_ML_NH_FREEZE_CWND_FACTOR;
                cubic->cwnd = cubic->frozen_cwnd;
                cubic->is_frozen = XQC_TRUE;
                cubic->freeze_start_time = now;
                cubic->loss_spike_during_freeze = XQC_FALSE;

                xqc_log(cubic->send_ctl->ctl_conn->log, XQC_LOG_REPORT,
                        "|cubic|freeze_enter|prob_nh:%.3f|cwnd:%ui|frozen_cwnd:%ui|",
                        cubic->ml_state_probs[XQC_ML_STATE_NH],
                        (uint32_t)(cubic->cwnd / XQC_CUBIC_ML_NH_FREEZE_CWND_FACTOR),
                        (uint32_t)cubic->frozen_cwnd);
                return;
            }
        }
    }

    if (cubic->min_rtt == 0 || rtt < cubic->min_rtt) {
        cubic->min_rtt = rtt;
    }

    if (xqc_cubic_in_congestion_recovery(cong_ctl, po->po_sent_time)) {
        return;
    }

    if (cubic->cwnd < cubic->ssthresh) {
        cubic->tcp_cwnd += acked_bytes;
        cubic->cwnd += acked_bytes;
    } else {
        cubic->tcp_cwnd_cnt += acked_bytes;
        if (cubic->tcp_cwnd_cnt >= cubic->tcp_cwnd) {
            cubic->tcp_cwnd += XQC_CUBIC_MSS;
            cubic->tcp_cwnd_cnt = 0;
        }
        xqc_cubic_update(cong_ctl, acked_bytes, now);
    }
}

uint64_t
xqc_cubic_get_cwnd(void *cong_ctl)
{
    xqc_cubic_t *cubic = (xqc_cubic_t *)(cong_ctl);
    return cubic->cwnd;
}

void
xqc_cubic_reset_cwnd(void *cong_ctl)
{
    xqc_cubic_t *cubic = (xqc_cubic_t *)(cong_ctl);
    cubic->epoch_start = 0;
    cubic->cwnd = cubic->min_cwnd;
    cubic->tcp_cwnd = cubic->min_cwnd;
    cubic->last_max_cwnd = cubic->min_cwnd;
}

int32_t
xqc_cubic_in_slow_start(void *cong_ctl)
{
    xqc_cubic_t *cubic = (xqc_cubic_t *)(cong_ctl);
    return cubic->cwnd < cubic->ssthresh ? 1 : 0;
}

void
xqc_cubic_restart_from_idle(void *cong_ctl, uint64_t arg)
{
    return;
}

static int
xqc_cubic_in_recovery(void *cong_ctl)
{
    return 0;
}

const xqc_cong_ctrl_callback_t xqc_cubic_cb = {
    .xqc_cong_ctl_size              = xqc_cubic_size,
    .xqc_cong_ctl_init              = xqc_cubic_init,
    .xqc_cong_ctl_on_lost           = xqc_cubic_on_lost,
    .xqc_cong_ctl_on_ack            = xqc_cubic_on_ack,
    .xqc_cong_ctl_get_cwnd          = xqc_cubic_get_cwnd,
    .xqc_cong_ctl_reset_cwnd        = xqc_cubic_reset_cwnd,
    .xqc_cong_ctl_in_slow_start     = xqc_cubic_in_slow_start,
    .xqc_cong_ctl_restart_from_idle = xqc_cubic_restart_from_idle,
    .xqc_cong_ctl_in_recovery       = xqc_cubic_in_recovery,
};
