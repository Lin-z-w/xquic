/**
 * @copyright Copyright (c) 2024, Alibaba Group Holding Limited
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <xquic/xquic.h>
#include "src/common/xqc_config.h"
#include "src/common/xqc_time.h"
#include "src/congestion_control/xqc_ml_cc.h"
#include "src/transport/xqc_send_ctl.h"

#ifdef XQC_ENABLE_ONNX
#include <onnxruntime_c_api.h>
#endif

#define XQC_ML_CC_USEC2SEC  1000000.0
#define XQC_ML_CC_MAX_CWND  (1024 * 1024 * 1024)

const float xqc_ml_cc_scaler_mean[XQC_ML_CC_NUM_FEATURES] = {
    8.96128915e+04f, 6.79938659e+00f, 1.30212317e+02f, 1.61393684e+03f,
    6.74786114e+00f, 3.83006478e+02f, 4.73249606e+03f, 3.85382679e+03f,
    1.62961778e+06f, 6.50594380e+02f, 1.62115443e+00f, 1.00222519e+00f,
    5.18842275e-04f, 1.89891871e+03f, 5.15254543e-02f
};

const float xqc_ml_cc_scaler_scale[XQC_ML_CC_NUM_FEATURES] = {
    5.86602124e+04f, 1.58179157e+01f, 4.45019329e+02f, 1.69024452e+03f,
    1.32390146e+01f, 9.93498002e+02f, 3.69058068e+03f, 1.17108226e+04f,
    1.48773062e+06f, 5.11887834e+02f, 5.66852703e+03f, 7.30771491e-02f,
    2.20059070e-04f, 2.61691161e+03f, 9.01346935e+00f
};

static void
xqc_ml_cc_softmax(float *input, float *output, int length)
{
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        sum += expf(input[i] - max_val);
    }

    for (int i = 0; i < length; i++) {
        output[i] = expf(input[i] - max_val) / sum;
    }
}

static void
xqc_ml_cc_normalize_features(float *normalized, float *raw, int size)
{
    for (int i = 0; i < size; i++) {
        if (xqc_ml_cc_scaler_scale[i] > 1e-6f) {
            normalized[i] = (raw[i] - xqc_ml_cc_scaler_mean[i]) / xqc_ml_cc_scaler_scale[i];
        } else {
            normalized[i] = raw[i] - xqc_ml_cc_scaler_mean[i];
        }
    }
}

static void
xqc_ml_cc_extract_features(xqc_ml_cc_t *ml_cc, float *features)
{
    xqc_send_ctl_t *ctl = ml_cc->send_ctl;
    if (!ctl) {
        memset(features, 0, XQC_ML_CC_NUM_FEATURES * sizeof(float));
        return;
    }

    float adjusted_rtt = (float)ctl->ctl_srtt / 1000.0f;
    float cwnd_bytes = ml_cc->cwnd_bytes;
    float pkt_in_fly = (float)ctl->ctl_bytes_in_flight;

    float short_lost_cnt = (float)ctl->ctl_recent_lost_count[0];
    float short_send_cnt = (float)ctl->ctl_recent_send_count[0];
    float short_loss_rate = (short_send_cnt > 0) ? (short_lost_cnt / short_send_cnt * 100.0f) : 0.0f;

    float long_lost_cnt = (float)(ctl->ctl_recent_lost_count[0] + ctl->ctl_recent_lost_count[1]);
    float long_send_cnt = (float)(ctl->ctl_recent_send_count[0] + ctl->ctl_recent_send_count[1]);
    float long_loss_rate = (long_send_cnt > 0) ? (long_lost_cnt / long_send_cnt * 100.0f) : 0.0f;

    float response_interval = 0.0f;
    if (ctl->ctl_delivered_time > 0 && ctl->ctl_first_sent_time > 0) {
        response_interval = (float)(ctl->ctl_delivered_time - ctl->ctl_first_sent_time) / 1000.0f;
    }

    float prev_rtt = 0.0f, prev_cwnd = 0.0f;
    if (ml_cc->sample_count > 0) {
        int prev_idx = (ml_cc->window_idx - 1 + XQC_ML_CC_WINDOW_SIZE) % XQC_ML_CC_WINDOW_SIZE;
        prev_rtt = ml_cc->history_window[prev_idx][0];
        prev_cwnd = ml_cc->history_window[prev_idx][8];
    }

    float rtt_diff = adjusted_rtt - prev_rtt;

    float eps = 1e-8f;
    float cwnd_change = cwnd_bytes / (prev_cwnd + eps);
    if (cwnd_change < 0.1f) cwnd_change = 0.1f;
    if (cwnd_change > 10.0f) cwnd_change = 10.0f;

    float pkt_ratio = pkt_in_fly / (cwnd_bytes + eps);
    if (pkt_ratio < 0.0f) pkt_ratio = 0.0f;
    if (pkt_ratio > 2.0f) pkt_ratio = 2.0f;

    float throughput_est = short_send_cnt / (response_interval / 1000.0f + eps);
    if (throughput_est < 0.0f) throughput_est = 0.0f;
    if (throughput_est > 10000.0f) throughput_est = 10000.0f;

    float loss_rate_diff = short_loss_rate - long_loss_rate;
    if (loss_rate_diff < -100.0f) loss_rate_diff = -100.0f;
    if (loss_rate_diff > 100.0f) loss_rate_diff = 100.0f;

    features[0] = adjusted_rtt;
    features[1] = short_loss_rate;
    features[2] = short_lost_cnt;
    features[3] = short_send_cnt;
    features[4] = long_loss_rate;
    features[5] = long_lost_cnt;
    features[6] = long_send_cnt;
    features[7] = response_interval;
    features[8] = cwnd_bytes;
    features[9] = pkt_in_fly;
    features[10] = rtt_diff;
    features[11] = cwnd_change;
    features[12] = pkt_ratio;
    features[13] = throughput_est;
    features[14] = loss_rate_diff;
}

static void
xqc_ml_cc_update_window(xqc_ml_cc_t *ml_cc, float *features)
{
    float *row = ml_cc->history_window[ml_cc->window_idx];
    for (int i = 0; i < XQC_ML_CC_NUM_FEATURES; i++) {
        row[i] = features[i];
    }
    ml_cc->window_idx = (ml_cc->window_idx + 1) % XQC_ML_CC_WINDOW_SIZE;
    ml_cc->sample_count++;
}

#ifdef XQC_ENABLE_ONNX
static void
xqc_ml_cc_run_inference(xqc_ml_cc_t *ml_cc)
{
    if (!ml_cc->onnx_session || !ml_cc->onnx_api) {
        ml_cc->state_probs[0] = 0.25f;
        ml_cc->state_probs[1] = 0.25f;
        ml_cc->state_probs[2] = 0.25f;
        ml_cc->state_probs[3] = 0.25f;
        return;
    }

    if (ml_cc->sample_count < XQC_ML_CC_WINDOW_SIZE) {
        ml_cc->state_probs[0] = 0.25f;
        ml_cc->state_probs[1] = 0.25f;
        ml_cc->state_probs[2] = 0.25f;
        ml_cc->state_probs[3] = 0.25f;
        return;
    }

    float input[1][XQC_ML_CC_WINDOW_SIZE][XQC_ML_CC_NUM_FEATURES];
    int start_idx = (ml_cc->window_idx - XQC_ML_CC_WINDOW_SIZE + XQC_ML_CC_WINDOW_SIZE) % XQC_ML_CC_WINDOW_SIZE;
    for (int t = 0; t < XQC_ML_CC_WINDOW_SIZE; t++) {
        int src_idx = (start_idx + t) % XQC_ML_CC_WINDOW_SIZE;
        for (int f = 0; f < XQC_ML_CC_NUM_FEATURES; f++) {
            input[0][t][f] = ml_cc->history_window[src_idx][f];
        }
    }

    const OrtApi *api = (const OrtApi *)ml_cc->onnx_api;
    OrtSession *session = (OrtSession *)ml_cc->onnx_session;

    int64_t input_shape[] = {1, XQC_ML_CC_WINDOW_SIZE, XQC_ML_CC_NUM_FEATURES};
    OrtMemoryInfo *mem_info = NULL;
    OrtStatus *status = api->CreateMemoryInfo("Cpu", OrtArenaAllocator, 0, OrtMemTypeDefault, &mem_info);
    if (status != NULL) {
        api->ReleaseStatus(status);
        return;
    }

    OrtValue *input_tensor = NULL;
    status = api->CreateTensorWithDataAsOrtValue(
        mem_info,
        input[0],
        sizeof(input),
        input_shape,
        3,
        ONNX_TYPE_TENSOR,
        &input_tensor
    );
    api->ReleaseMemoryInfo(mem_info);

    if (status != NULL) {
        api->ReleaseStatus(status);
        return;
    }

    const char *input_names[] = {"input"};
    const char *output_names[] = {"output"};
    OrtValue *output_tensor = NULL;

    status = api->Run(
        session,
        NULL,
        input_names,
        (const OrtValue *[]){input_tensor},
        1,
        output_names,
        1,
        &output_tensor
    );

    api->ReleaseValue(input_tensor);

    if (status != NULL) {
        api->ReleaseStatus(status);
        return;
    }

    float *output_data = NULL;
    status = api->GetTensorMutableData(output_tensor, (void **)&output_data);
    if (status == NULL && output_data != NULL) {
        float logits[4] = {output_data[0], output_data[1], output_data[2], output_data[3]};
        xqc_ml_cc_softmax(logits, ml_cc->state_probs, 4);
    } else {
        ml_cc->state_probs[0] = 0.25f;
        ml_cc->state_probs[1] = 0.25f;
        ml_cc->state_probs[2] = 0.25f;
        ml_cc->state_probs[3] = 0.25f;
    }

    api->ReleaseValue(output_tensor);
}
#else
static void
xqc_ml_cc_run_inference(xqc_ml_cc_t *ml_cc)
{
    ml_cc->state_probs[0] = 0.25f;
    ml_cc->state_probs[1] = 0.25f;
    ml_cc->state_probs[2] = 0.25f;
    ml_cc->state_probs[3] = 0.25f;
}
#endif

static xqc_ml_cc_state_t
xqc_ml_cc_determine_state(xqc_ml_cc_t *ml_cc)
{
    if (ml_cc->state_probs[3] > XQC_ML_CC_NH_THRESHOLD) {
        return XQC_ML_CC_STATE_NH;
    }

    float max_prob = ml_cc->state_probs[0];
    int max_idx = 0;
    for (int i = 1; i < 4; i++) {
        if (ml_cc->state_probs[i] > max_prob) {
            max_prob = ml_cc->state_probs[i];
            max_idx = i;
        }
    }

    return (xqc_ml_cc_state_t)max_idx;
}

void
xqc_ml_cc_feed_features(void *cong, xqc_usec_t ack_recv_time,
    xqc_usec_t adjusted_rtt, double short_loss_rate, unsigned short_lost_cnt,
    unsigned short_send_cnt, double long_loss_rate, unsigned long_lost_cnt,
    unsigned long_send_cnt, xqc_usec_t response_interval, uint64_t cwnd,
    unsigned pkt_in_fly)
{
    xqc_ml_cc_t *ml_cc = (xqc_ml_cc_t *)cong;
    if (!ml_cc) {
        return;
    }

    float features[XQC_ML_CC_NUM_FEATURES];
    memset(features, 0, sizeof(features));

    features[0] = (float)adjusted_rtt / 1000.0f;
    features[1] = (float)short_loss_rate;
    features[2] = (float)short_lost_cnt;
    features[3] = (float)short_send_cnt;
    features[4] = (float)long_loss_rate;
    features[5] = (float)long_lost_cnt;
    features[6] = (float)long_send_cnt;
    features[7] = (float)response_interval / 1000.0f;
    features[8] = (float)cwnd;
    features[9] = (float)pkt_in_fly;

    float prev_rtt = 0.0f, prev_cwnd = 0.0f;
    if (ml_cc->sample_count > 0) {
        int prev_idx = (ml_cc->window_idx - 1 + XQC_ML_CC_WINDOW_SIZE) % XQC_ML_CC_WINDOW_SIZE;
        prev_rtt = ml_cc->history_window[prev_idx][0];
        prev_cwnd = ml_cc->history_window[prev_idx][8];
    }

    features[10] = features[0] - prev_rtt;

    float eps = 1e-8f;
    features[11] = features[8] / (prev_cwnd + eps);
    if (features[11] < 0.1f) features[11] = 0.1f;
    if (features[11] > 10.0f) features[11] = 10.0f;

    features[12] = features[9] / (features[8] + eps);
    if (features[12] < 0.0f) features[12] = 0.0f;
    if (features[12] > 2.0f) features[12] = 2.0f;

    features[13] = features[3] / (features[7] / 1000.0f + eps);
    if (features[13] < 0.0f) features[13] = 0.0f;
    if (features[13] > 10000.0f) features[13] = 10000.0f;

    features[14] = features[1] - features[4];
    if (features[14] < -100.0f) features[14] = -100.0f;
    if (features[14] > 100.0f) features[14] = 100.0f;

    xqc_ml_cc_update_window(ml_cc, features);

    if (!ml_cc->is_frozen) {
        xqc_ml_cc_run_inference(ml_cc);
        ml_cc->last_state = xqc_ml_cc_determine_state(ml_cc);
    }

    ml_cc->last_rtt = adjusted_rtt;
}

static size_t
xqc_ml_cc_size(void)
{
    return sizeof(xqc_ml_cc_t);
}

static void
xqc_ml_cc_init(void *cong, xqc_send_ctl_t *ctl_ctx, xqc_cc_params_t cc_params)
{
    xqc_ml_cc_t *ml_cc = (xqc_ml_cc_t *)cong;
    ml_cc->send_ctl = ctl_ctx;
    ml_cc->log = ctl_ctx->ctl_conn->log;

    ml_cc->init_cwnd_bytes = XQC_ML_CC_INIT_WIN;
    ml_cc->min_cwnd_bytes = XQC_ML_CC_MIN_CWND;

    if (cc_params.customize_on) {
        if (cc_params.init_cwnd > 0) {
            ml_cc->init_cwnd_bytes = cc_params.init_cwnd * XQC_ML_CC_MSS;
        }
    }

    ml_cc->cwnd_bytes = ml_cc->init_cwnd_bytes;
    ml_cc->window_idx = 0;
    ml_cc->sample_count = 0;
    ml_cc->last_prediction = 50.0f;
    ml_cc->use_ml_prediction = 1;

    ml_cc->congestion_recovery_start_time = 0;
    ml_cc->last_rtt = 0;
    ml_cc->loss_reduction_factor = 0.0f;

    ml_cc->last_state = XQC_ML_CC_STATE_NONE;
    ml_cc->qu_consecutive_count = 0;
    ml_cc->is_frozen = XQC_FALSE;
    ml_cc->freeze_start_time = 0;
    ml_cc->last_state = XQC_ML_CC_STATE_NONE;
    ml_cc->frozen_cwnd = ml_cc->cwnd_bytes;
    ml_cc->loss_spike_during_freeze = XQC_FALSE;

    memset(ml_cc->history_window, 0, sizeof(ml_cc->history_window));
    memset(ml_cc->state_probs, 0, sizeof(ml_cc->state_probs));

#ifdef XQC_ENABLE_ONNX
    ml_cc->onnx_api = (void *)OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ml_cc->onnx_session = NULL;

    if (ml_cc->onnx_api) {
        const OrtApi *api = (const OrtApi *)ml_cc->onnx_api;
        OrtEnv *env = NULL;
        OrtStatus *status = api->CreateEnv(
            ORT_LOGGING_LEVEL_WARNING, "xqc_ml_cc", &env);

        if (status == NULL && env) {
            OrtSession *session = NULL;
            status = api->CreateSession(
                env,
                XQC_ONNX_MODEL_PATH,
                NULL,
                &session);

            if (status == NULL && session != NULL) {
                ml_cc->onnx_session = (void *)session;
            }
            api->ReleaseEnv(env);
        }
    }
#else
    ml_cc->onnx_session = NULL;
    ml_cc->onnx_api = NULL;
#endif

    xqc_log(ml_cc->log, XQC_LOG_DEBUG,
            "|ml_cc|initialized|cwnd:%u|onnx:%d|qu_threshold:%d|",
            (uint32_t)ml_cc->cwnd_bytes,
            ml_cc->onnx_session != NULL ? 1 : 0,
            XQC_ML_CC_QU_CONSECUTIVE_THRESHOLD);
}

static void
xqc_ml_cc_on_ack(void *cong, xqc_packet_out_t *po, xqc_usec_t now)
{
    xqc_ml_cc_t *ml_cc = (xqc_ml_cc_t *)cong;

    if (ml_cc->is_frozen) {
        xqc_usec_t elapsed = now - ml_cc->freeze_start_time;
        double short_loss_rate = 0.0;

        xqc_send_ctl_t *ctl = ml_cc->send_ctl;
        if (ctl && ctl->ctl_recent_send_count[0] > 0) {
            short_loss_rate = (double)ctl->ctl_recent_lost_count[0] / ctl->ctl_recent_send_count[0] * 100.0;
        }

        if (elapsed < XQC_ML_CC_NH_FREEZE_DURATION) {
            if (short_loss_rate > XQC_ML_CC_NH_LOSS_HIGH) {
                ml_cc->loss_spike_during_freeze = XQC_TRUE;
            }
        } else {
            if (!ml_cc->loss_spike_during_freeze) {
                ml_cc->is_frozen = XQC_FALSE;
                xqc_log(ml_cc->log, XQC_LOG_DEBUG,
                        "|ml_cc|frozen_exit|elapsed:%ui|no_loss_spike|",
                        (uint32_t)elapsed);
            } else if (short_loss_rate < XQC_ML_CC_NH_LOSS_LOW) {
                ml_cc->is_frozen = XQC_FALSE;
                ml_cc->loss_spike_during_freeze = XQC_FALSE;
                xqc_log(ml_cc->log, XQC_LOG_DEBUG,
                        "|ml_cc|frozen_exit|elapsed:%ui|loss_below_threshold|",
                        (uint32_t)elapsed);
            }
        }

        if (ml_cc->is_frozen) {
            ml_cc->cwnd_bytes = ml_cc->frozen_cwnd;
            xqc_log(ml_cc->log, XQC_LOG_DEBUG,
                    "|ml_cc|frozen|elapsed:%ui|loss:%.2f|spike:%d|cwnd:%u|",
                    (uint32_t)elapsed, short_loss_rate,
                    ml_cc->loss_spike_during_freeze ? 1 : 0,
                    (uint32_t)ml_cc->cwnd_bytes);
            return;
        }
    }

    xqc_ml_cc_state_t state = ml_cc->last_state;

    if (state == XQC_ML_CC_STATE_NH) {
        ml_cc->frozen_cwnd = ml_cc->cwnd_bytes * XQC_ML_CC_NH_FREEZE_CWND_FACTOR;
        ml_cc->cwnd_bytes = ml_cc->frozen_cwnd;
        ml_cc->is_frozen = XQC_TRUE;
        ml_cc->freeze_start_time = now;
        ml_cc->loss_spike_during_freeze = XQC_FALSE;

        xqc_log(ml_cc->log, XQC_LOG_DEBUG,
                "|ml_cc|nh_enter|frozen_cwnd:%u|prob_nh:%.3f|",
                (uint32_t)ml_cc->frozen_cwnd, ml_cc->state_probs[3]);
        return;
    }

    if (state == XQC_ML_CC_STATE_UT) {
        ml_cc->qu_consecutive_count = 0;
        float ack_bytes = 0.0f;
        if (po) {
            ack_bytes = (float)po->po_used_size;
        }
        ml_cc->cwnd_bytes += ack_bytes * XQC_ML_CC_UT_CWND_GAIN;

        xqc_log(ml_cc->log, XQC_LOG_DEBUG,
                "|ml_cc|ut_state|gain:%.2f|cwnd:%u|",
                XQC_ML_CC_UT_CWND_GAIN, (uint32_t)ml_cc->cwnd_bytes);

    } else if (state == XQC_ML_CC_STATE_QU) {
        ml_cc->qu_consecutive_count++;

        if (ml_cc->qu_consecutive_count >= XQC_ML_CC_QU_CONSECUTIVE_THRESHOLD) {
            ml_cc->cwnd_bytes *= XQC_ML_CC_QU_CWND_DECREASE;
            if (ml_cc->cwnd_bytes < ml_cc->min_cwnd_bytes) {
                ml_cc->cwnd_bytes = ml_cc->min_cwnd_bytes;
            }
            ml_cc->qu_consecutive_count = 0;

            xqc_log(ml_cc->log, XQC_LOG_DEBUG,
                    "|ml_cc|qu_state|decrease|cwnd:%u|qu_count:%d|",
                    (uint32_t)ml_cc->cwnd_bytes, ml_cc->qu_consecutive_count);
        }

    } else if (state == XQC_ML_CC_STATE_EB) {
        ml_cc->qu_consecutive_count = 0;
        ml_cc->cwnd_bytes *= XQC_ML_CC_EB_CWND_DECREASE;
        if (ml_cc->cwnd_bytes < ml_cc->min_cwnd_bytes) {
            ml_cc->cwnd_bytes = ml_cc->min_cwnd_bytes;
        }

        xqc_log(ml_cc->log, XQC_LOG_DEBUG,
                "|ml_cc|eb_state|decrease|cwnd:%u|",
                (uint32_t)ml_cc->cwnd_bytes);
    }

    if (ml_cc->cwnd_bytes > XQC_ML_CC_MAX_CWND) {
        ml_cc->cwnd_bytes = XQC_ML_CC_MAX_CWND;
    }
}

static void
xqc_ml_cc_on_lost(void *cong, xqc_usec_t lost_sent_time)
{
}

static uint64_t
xqc_ml_cc_get_cwnd(void *cong)
{
    xqc_ml_cc_t *ml_cc = (xqc_ml_cc_t *)cong;
    return (uint64_t)ml_cc->cwnd_bytes;
}

static void
xqc_ml_cc_reset_cwnd(void *cong)
{
    xqc_ml_cc_t *ml_cc = (xqc_ml_cc_t *)cong;
    ml_cc->cwnd_bytes = ml_cc->init_cwnd_bytes;
}

static int
xqc_ml_cc_in_slow_start(void *cong)
{
    xqc_ml_cc_t *ml_cc = (xqc_ml_cc_t *)cong;
    return (ml_cc->sample_count < XQC_ML_CC_WINDOW_SIZE) ? 1 : 0;
}

static int
xqc_ml_cc_in_recovery(void *cong)
{
    return 0;
}

static void
xqc_ml_cc_restart_from_idle(void *cong, uint64_t arg)
{
    xqc_ml_cc_t *ml_cc = (xqc_ml_cc_t *)cong;
    ml_cc->cwnd_bytes = xqc_max(ml_cc->cwnd_bytes, ml_cc->init_cwnd_bytes);
}

const xqc_cong_ctrl_callback_t xqc_ml_cc_cb = {
    .xqc_cong_ctl_size            = xqc_ml_cc_size,
    .xqc_cong_ctl_init            = xqc_ml_cc_init,
    .xqc_cong_ctl_on_lost         = xqc_ml_cc_on_lost,
    .xqc_cong_ctl_on_ack          = xqc_ml_cc_on_ack,
    .xqc_cong_ctl_get_cwnd        = xqc_ml_cc_get_cwnd,
    .xqc_cong_ctl_reset_cwnd      = xqc_ml_cc_reset_cwnd,
    .xqc_cong_ctl_in_slow_start   = xqc_ml_cc_in_slow_start,
    .xqc_cong_ctl_in_recovery     = xqc_ml_cc_in_recovery,
    .xqc_cong_ctl_restart_from_idle = xqc_ml_cc_restart_from_idle,
    .xqc_cong_ctl_info_cb         = NULL,
};