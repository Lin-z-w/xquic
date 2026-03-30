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
#define XQC_ML_CC_EPSILON   1e-6f

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

const float xqc_ml_cc_qu_scaler_mean[XQC_ML_CC_NUM_FEATURES] = {
    8.96128915e+04f, 6.79938659e+00f, 1.30212317e+02f, 1.61393684e+03f,
    6.74786114e+00f, 3.83006478e+02f, 4.73249606e+03f, 3.85382679e+03f,
    1.62961778e+06f, 6.50594380e+02f, 1.62115443e+00f, 1.00222519e+00f,
    5.18842275e-04f, 1.89891871e+03f, 5.15254543e-02f
};

const float xqc_ml_cc_qu_scaler_scale[XQC_ML_CC_NUM_FEATURES] = {
    5.86602124e+04f, 1.58179157e+01f, 4.45019329e+02f, 1.69024452e+03f,
    1.32390146e+01f, 9.93498002e+02f, 3.69058068e+03f, 1.17108226e+04f,
    1.48773062e+06f, 5.11887834e+02f, 5.66852703e+03f, 7.30771491e-02f,
    2.20059070e-04f, 2.61691161e+03f, 9.01346935e+00f
};

static float
xqc_ml_cc_clamp_float(float value, float min_value, float max_value)
{
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static xqc_log_t *
xqc_ml_cc_get_log(xqc_ml_cc_t *ml_cc)
{
    if (ml_cc == NULL || ml_cc->send_ctl == NULL || ml_cc->send_ctl->ctl_conn == NULL) {
        return NULL;
    }

    return ml_cc->send_ctl->ctl_conn->log;
}

#ifdef XQC_ENABLE_ONNX
static OrtEnv *xqc_ml_cc_shared_ort_env = NULL;

static void
xqc_ml_cc_log_onnx_failure(xqc_ml_cc_t *ml_cc, const char *model_tag,
    const char *phase, const char *detail, const char *output_name)
{
    xqc_log_t *log = xqc_ml_cc_get_log(ml_cc);

    if (log == NULL) {
        return;
    }

    xqc_log(log, XQC_LOG_REPORT,
            "|ml_cc|onnx_fail|model:%s|phase:%s|detail:%s|output:%s|samples:%d|"
            "window:%d|state_ready:%d|queue_ready:%d|",
            model_tag != NULL ? model_tag : "unknown",
            phase != NULL ? phase : "unknown",
            detail != NULL ? detail : "unknown",
            output_name != NULL ? output_name : "null",
            ml_cc != NULL ? ml_cc->sample_count : -1, XQC_ML_CC_WINDOW_SIZE,
            ml_cc != NULL ? ml_cc->state_model_ready : 0,
            ml_cc != NULL ? ml_cc->queue_model_ready : 0);
}

static void
xqc_ml_cc_log_ort_status(xqc_ml_cc_t *ml_cc, const OrtApi *api,
    OrtStatus *status, const char *model_tag, const char *phase,
    const char *output_name, const char *model_path)
{
    const char *msg = "unknown";
    xqc_log_t *log = xqc_ml_cc_get_log(ml_cc);

    if (log == NULL) {
        return;
    }

    if (api != NULL && status != NULL) {
        msg = api->GetErrorMessage(status);
    }

    xqc_log(log, XQC_LOG_REPORT,
            "|ml_cc|onnx_fail|model:%s|phase:%s|path:%s|output:%s|samples:%d|"
            "window:%d|msg:%s|",
            model_tag != NULL ? model_tag : "unknown",
            phase != NULL ? phase : "unknown",
            model_path != NULL ? model_path : "null",
            output_name != NULL ? output_name : "null",
            ml_cc != NULL ? ml_cc->sample_count : -1, XQC_ML_CC_WINDOW_SIZE,
            msg != NULL ? msg : "unknown");
}

static OrtEnv *
xqc_ml_cc_get_shared_env(xqc_ml_cc_t *ml_cc, const OrtApi *api,
    const char *model_tag, const char *model_path)
{
    OrtStatus *status = NULL;

    if (api == NULL) {
        xqc_ml_cc_log_onnx_failure(ml_cc, model_tag, "create_env",
            "null_onnx_api", NULL);
        return NULL;
    }

    if (xqc_ml_cc_shared_ort_env != NULL) {
        return xqc_ml_cc_shared_ort_env;
    }

    status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "xqc_ml_cc",
        &xqc_ml_cc_shared_ort_env);
    if (status != NULL) {
        xqc_ml_cc_log_ort_status(ml_cc, api, status, model_tag,
            "create_env", NULL, model_path);
        api->ReleaseStatus(status);
        xqc_ml_cc_shared_ort_env = NULL;
        return NULL;
    }

    return xqc_ml_cc_shared_ort_env;
}
#endif

static const char *
xqc_ml_cc_state_to_str(xqc_ml_cc_state_t state)
{
    switch (state) {
    case XQC_ML_CC_STATE_UT:
        return "UT";
    case XQC_ML_CC_STATE_QU:
        return "QU";
    case XQC_ML_CC_STATE_EB:
        return "EB";
    case XQC_ML_CC_STATE_NH:
        return "NH";
    case XQC_ML_CC_STATE_NONE:
    default:
        return "NONE";
    }
}

static void
xqc_ml_cc_softmax(float *input, float *output, int length)
{
    float max_val = input[0];
    float sum = 0.0f;
    int i;

    for (i = 1; i < length; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    for (i = 0; i < length; i++) {
        sum += expf(input[i] - max_val);
    }

    if (sum <= XQC_ML_CC_EPSILON) {
        for (i = 0; i < length; i++) {
            output[i] = 1.0f / length;
        }
        return;
    }

    for (i = 0; i < length; i++) {
        output[i] = expf(input[i] - max_val) / sum;
    }
}

static void
xqc_ml_cc_set_uniform_probs(xqc_ml_cc_t *ml_cc)
{
    int i;
    for (i = 0; i < 4; i++) {
        ml_cc->state_probs[i] = 0.25f;
    }
}

static void
xqc_ml_cc_normalize_feature_row(float *normalized, const float *raw,
    const float *mean, const float *scale, int size)
{
    int i;
    for (i = 0; i < size; i++) {
        if (scale[i] > XQC_ML_CC_EPSILON) {
            normalized[i] = (raw[i] - mean[i]) / scale[i];
        } else {
            normalized[i] = raw[i] - mean[i];
        }
    }
}

static void
xqc_ml_cc_fill_input_window(xqc_ml_cc_t *ml_cc,
    float input[1][XQC_ML_CC_WINDOW_SIZE][XQC_ML_CC_NUM_FEATURES],
    const float *mean, const float *scale)
{
    int start_idx, t, f;

    start_idx = (ml_cc->window_idx - XQC_ML_CC_WINDOW_SIZE + XQC_ML_CC_WINDOW_SIZE)
        % XQC_ML_CC_WINDOW_SIZE;

    for (t = 0; t < XQC_ML_CC_WINDOW_SIZE; t++) {
        int src_idx = (start_idx + t) % XQC_ML_CC_WINDOW_SIZE;
        xqc_ml_cc_normalize_feature_row(input[0][t], ml_cc->history_window[src_idx],
            mean, scale, XQC_ML_CC_NUM_FEATURES);
        for (f = 0; f < XQC_ML_CC_NUM_FEATURES; f++) {
            if (!isfinite(input[0][t][f])) {
                input[0][t][f] = 0.0f;
            }
        }
    }
}

static void
xqc_ml_cc_build_features(xqc_ml_cc_t *ml_cc, float *features,
    xqc_usec_t adjusted_rtt, double short_loss_rate, unsigned short_lost_cnt,
    unsigned short_send_cnt, double long_loss_rate, unsigned long_lost_cnt,
    unsigned long_send_cnt, xqc_usec_t response_interval, uint64_t cwnd,
    unsigned pkt_in_fly)
{
    float prev_rtt = 0.0f;
    float prev_cwnd = 0.0f;
    float eps = 1e-8f;

    if (ml_cc->sample_count > 0) {
        int prev_idx = (ml_cc->window_idx - 1 + XQC_ML_CC_WINDOW_SIZE) % XQC_ML_CC_WINDOW_SIZE;
        prev_rtt = ml_cc->history_window[prev_idx][0];
        prev_cwnd = ml_cc->history_window[prev_idx][8];
    }

    /*
     * state_predict 的 derived_v1_15 特征和配套 scaler 使用 RTT/响应间隔
     * 的原始微秒量级；这里只保留原始值，避免和部署包统计量失配。
     */
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
    features[11] = xqc_ml_cc_clamp_float(features[11], 0.1f, 10.0f);

    features[12] = features[9] / (features[8] + eps);
    features[12] = xqc_ml_cc_clamp_float(features[12], 0.0f, 2.0f);

    features[13] = features[3] / (features[7] / 1000.0f + eps);
    features[13] = xqc_ml_cc_clamp_float(features[13], 0.0f, 10000.0f);

    features[14] = features[1] - features[4];
    features[14] = xqc_ml_cc_clamp_float(features[14], -100.0f, 100.0f);
}

static void
xqc_ml_cc_update_window(xqc_ml_cc_t *ml_cc, float *features)
{
    int i;
    float *row = ml_cc->history_window[ml_cc->window_idx];

    for (i = 0; i < XQC_ML_CC_NUM_FEATURES; i++) {
        row[i] = features[i];
    }

    ml_cc->window_idx = (ml_cc->window_idx + 1) % XQC_ML_CC_WINDOW_SIZE;
    ml_cc->sample_count++;
}

static float
xqc_ml_cc_get_acked_bytes(xqc_packet_out_t *po)
{
    if (po && po->po_used_size > 0) {
        return (float)po->po_used_size;
    }
    return (float)XQC_ML_CC_MSS;
}

static void
xqc_ml_cc_clamp_cwnd(xqc_ml_cc_t *ml_cc)
{
    if (ml_cc->cwnd_bytes < ml_cc->min_cwnd_bytes) {
        ml_cc->cwnd_bytes = ml_cc->min_cwnd_bytes;
    }
    if (ml_cc->cwnd_bytes > XQC_ML_CC_MAX_CWND) {
        ml_cc->cwnd_bytes = XQC_ML_CC_MAX_CWND;
    }
}

static xqc_ml_cc_state_t
xqc_ml_cc_determine_state(xqc_ml_cc_t *ml_cc)
{
    float max_prob;
    int max_idx;
    int i;

    if (ml_cc->state_probs[3] > XQC_ML_CC_NH_THRESHOLD) {
        return XQC_ML_CC_STATE_NH;
    }

    max_prob = ml_cc->state_probs[0];
    max_idx = 0;
    for (i = 1; i < 4; i++) {
        if (ml_cc->state_probs[i] > max_prob) {
            max_prob = ml_cc->state_probs[i];
            max_idx = i;
        }
    }

    return (xqc_ml_cc_state_t)max_idx;
}

#ifdef XQC_ENABLE_ONNX
static OrtSession *
xqc_ml_cc_create_session(xqc_ml_cc_t *ml_cc, const OrtApi *api,
    const char *model_tag, const char *model_path)
{
    OrtEnv *env = NULL;
    OrtSessionOptions *options = NULL;
    OrtSession *session = NULL;
    OrtStatus *status = NULL;

    if (api == NULL || model_path == NULL) {
        xqc_ml_cc_log_onnx_failure(ml_cc, model_tag, "create_session",
            "invalid_api_or_model_path", NULL);
        return NULL;
    }

    env = xqc_ml_cc_get_shared_env(ml_cc, api, model_tag, model_path);
    if (env == NULL) {
        goto end;
    }

    status = api->CreateSessionOptions(&options);
    if (status != NULL) {
        xqc_ml_cc_log_ort_status(ml_cc, api, status, model_tag,
            "create_session_options", NULL, model_path);
        goto end;
    }

    status = api->CreateSession(env, model_path, options, &session);
    if (status != NULL) {
        xqc_ml_cc_log_ort_status(ml_cc, api, status, model_tag,
            "create_session", NULL, model_path);
    }

end:
    if (status != NULL) {
        api->ReleaseStatus(status);
        session = NULL;
    }
    if (options != NULL) {
        api->ReleaseSessionOptions(options);
    }
    return session;
}

static xqc_bool_t
xqc_ml_cc_run_ort_inference(xqc_ml_cc_t *ml_cc, void *session_ptr,
    const char *model_tag, const char *output_name, const float *mean,
    const float *scale,
    float *output, size_t output_len)
{
    const OrtApi *api;
    OrtSession *session;
    OrtMemoryInfo *mem_info = NULL;
    OrtValue *input_tensor = NULL;
    OrtValue *output_tensor = NULL;
    OrtStatus *status = NULL;
    const char *input_names[] = {"input"};
    const char *output_names[1];
    const OrtValue *input_values[1];
    float input[1][XQC_ML_CC_WINDOW_SIZE][XQC_ML_CC_NUM_FEATURES];
    float *output_data = NULL;
    int64_t input_shape[] = {1, XQC_ML_CC_WINDOW_SIZE, XQC_ML_CC_NUM_FEATURES};
    size_t i;
    xqc_bool_t ret = XQC_FALSE;

    if (ml_cc->sample_count < XQC_ML_CC_WINDOW_SIZE) {
        return XQC_FALSE;
    }

    if (session_ptr == NULL) {
        xqc_ml_cc_log_onnx_failure(ml_cc, model_tag, "precheck",
            "null_session", output_name);
        return XQC_FALSE;
    }

    if (ml_cc->onnx_api == NULL) {
        xqc_ml_cc_log_onnx_failure(ml_cc, model_tag, "precheck",
            "null_onnx_api", output_name);
        return XQC_FALSE;
    }

    if (output == NULL || output_name == NULL) {
        xqc_ml_cc_log_onnx_failure(ml_cc, model_tag, "precheck",
            "invalid_output_buffer", output_name);
        return XQC_FALSE;
    }

    api = (const OrtApi *)ml_cc->onnx_api;
    session = (OrtSession *)session_ptr;

    xqc_ml_cc_fill_input_window(ml_cc, input, mean, scale);

    status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
    if (status != NULL) {
        xqc_ml_cc_log_ort_status(ml_cc, api, status, model_tag,
            "create_cpu_memory_info", output_name, NULL);
        goto end;
    }

    status = api->CreateTensorWithDataAsOrtValue(mem_info, input, sizeof(input),
        input_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    if (status != NULL) {
        xqc_ml_cc_log_ort_status(ml_cc, api, status, model_tag,
            "create_input_tensor", output_name, NULL);
        goto end;
    }

    output_names[0] = output_name;
    input_values[0] = input_tensor;
    status = api->Run(session, NULL, input_names, input_values, 1,
        output_names, 1, &output_tensor);
    if (status != NULL) {
        xqc_ml_cc_log_ort_status(ml_cc, api, status, model_tag,
            "run", output_name, NULL);
        goto end;
    }

    status = api->GetTensorMutableData(output_tensor, (void **)&output_data);
    if (status != NULL) {
        xqc_ml_cc_log_ort_status(ml_cc, api, status, model_tag,
            "get_tensor_data", output_name, NULL);
        goto end;
    }

    if (output_data == NULL) {
        xqc_ml_cc_log_onnx_failure(ml_cc, model_tag, "get_tensor_data",
            "null_output_data", output_name);
        goto end;
    }

    for (i = 0; i < output_len; i++) {
        output[i] = output_data[i];
        if (!isfinite(output[i])) {
            xqc_ml_cc_log_onnx_failure(ml_cc, model_tag, "validate_output",
                "nonfinite_output", output_name);
            goto end;
        }
    }

    ret = XQC_TRUE;

end:
    if (status != NULL) {
        api->ReleaseStatus(status);
    }
    if (output_tensor != NULL) {
        api->ReleaseValue(output_tensor);
    }
    if (input_tensor != NULL) {
        api->ReleaseValue(input_tensor);
    }
    if (mem_info != NULL) {
        api->ReleaseMemoryInfo(mem_info);
    }
    return ret;
}

static void
xqc_ml_cc_run_state_inference(xqc_ml_cc_t *ml_cc)
{
    float logits[4];

    if (!xqc_ml_cc_run_ort_inference(ml_cc, ml_cc->onnx_session, "state",
            "output", xqc_ml_cc_scaler_mean, xqc_ml_cc_scaler_scale, logits,
            4))
    {
        xqc_ml_cc_set_uniform_probs(ml_cc);
        return;
    }

    xqc_ml_cc_softmax(logits, ml_cc->state_probs, 4);
}

static xqc_bool_t
xqc_ml_cc_run_qu_queue_inference(xqc_ml_cc_t *ml_cc, float *queue_depth)
{
    float raw_output = 0.0f;

    if (!xqc_ml_cc_run_ort_inference(ml_cc, ml_cc->queue_onnx_session,
            "queue", "queue_depth_percentage", xqc_ml_cc_qu_scaler_mean,
            xqc_ml_cc_qu_scaler_scale, &raw_output, 1))
    {
        return XQC_FALSE;
    }

    *queue_depth = xqc_ml_cc_clamp_float(raw_output, 0.0f, 1.0f);
    return XQC_TRUE;
}
#else
static void
xqc_ml_cc_run_state_inference(xqc_ml_cc_t *ml_cc)
{
    xqc_ml_cc_set_uniform_probs(ml_cc);
}

static xqc_bool_t
xqc_ml_cc_run_qu_queue_inference(xqc_ml_cc_t *ml_cc, float *queue_depth)
{
    (void)ml_cc;
    (void)queue_depth;
    return XQC_FALSE;
}
#endif

static void
xqc_ml_cc_apply_qu_fallback(xqc_ml_cc_t *ml_cc)
{
    ml_cc->qu_consecutive_count++;

    if (ml_cc->qu_consecutive_count >= XQC_ML_CC_QU_CONSECUTIVE_THRESHOLD) {
        ml_cc->cwnd_bytes *= XQC_ML_CC_QU_CWND_DECREASE;
        ml_cc->qu_consecutive_count = 0;
        xqc_ml_cc_clamp_cwnd(ml_cc);

        xqc_log(xqc_ml_cc_get_log(ml_cc), XQC_LOG_DEBUG,
                "|ml_cc|qu_fallback|cwnd:%u|threshold:%d|",
                (uint32_t)ml_cc->cwnd_bytes,
                XQC_ML_CC_QU_CONSECUTIVE_THRESHOLD);
    }
}

static float
xqc_ml_cc_calc_qu_growth_strength(xqc_ml_cc_t *ml_cc)
{
    float low_boundary = ml_cc->queue_threshold_k - XQC_ML_CC_QU_QUEUE_DEADZONE;
    float dist_up;

    if (ml_cc->queue_depth_ema >= low_boundary) {
        return 0.0f;
    }

    low_boundary = xqc_max(low_boundary, XQC_ML_CC_EPSILON);
    dist_up = (low_boundary - ml_cc->queue_depth_ema) / low_boundary;
    dist_up = xqc_ml_cc_clamp_float(dist_up, 0.0f, 1.0f);

    return powf(dist_up, XQC_ML_CC_QU_GAMMA_UP);
}

static float
xqc_ml_cc_calc_qu_drain_strength(xqc_ml_cc_t *ml_cc)
{
    float high_boundary = ml_cc->queue_threshold_k + XQC_ML_CC_QU_QUEUE_DEADZONE;
    float denom;
    float dist_down;

    if (ml_cc->queue_depth_ema <= high_boundary) {
        return 0.0f;
    }

    denom = xqc_max(1.0f - high_boundary, XQC_ML_CC_EPSILON);
    dist_down = (ml_cc->queue_depth_ema - high_boundary) / denom;
    dist_down = xqc_ml_cc_clamp_float(dist_down, 0.0f, 1.0f);

    return powf(dist_down, XQC_ML_CC_QU_GAMMA_DOWN);
}

static void
xqc_ml_cc_apply_qu_fine_grained_control(xqc_ml_cc_t *ml_cc, xqc_packet_out_t *po)
{
    float acked = xqc_ml_cc_get_acked_bytes(po);
    float grow_strength = 0.0f;
    float drain_strength = 0.0f;
    float cwnd_before = ml_cc->cwnd_bytes;
    const char *mode = "hold";

    ml_cc->qu_consecutive_count = 0;

    if (ml_cc->queue_depth_ema < ml_cc->queue_threshold_k - XQC_ML_CC_QU_QUEUE_DEADZONE) {
        grow_strength = xqc_ml_cc_calc_qu_growth_strength(ml_cc);
        ml_cc->cwnd_bytes += acked * XQC_ML_CC_QU_GROWTH_BASE * grow_strength;
        mode = "grow";

    } else if (ml_cc->queue_depth_ema > ml_cc->queue_threshold_k + XQC_ML_CC_QU_QUEUE_DEADZONE) {
        drain_strength = xqc_ml_cc_calc_qu_drain_strength(ml_cc);
        ml_cc->cwnd_bytes -= acked * XQC_ML_CC_QU_DRAIN_BASE * drain_strength;
        mode = "drain";

    } else {
        ml_cc->cwnd_bytes += acked * XQC_ML_CC_QU_HOLD_GAIN;
    }

    xqc_ml_cc_clamp_cwnd(ml_cc);

    xqc_log(xqc_ml_cc_get_log(ml_cc), XQC_LOG_DEBUG,
            "|ml_cc|qu_fine|mode:%s|prob_qu:%.3f|queue_raw:%.3f|queue_ema:%.3f|"
            "queue_k:%.3f|grow:%.3f|drain:%.3f|acked:%.1f|cwnd_before:%u|cwnd_after:%u|",
            mode, ml_cc->state_probs[XQC_ML_CC_STATE_QU], ml_cc->queue_depth_raw,
            ml_cc->queue_depth_ema, ml_cc->queue_threshold_k, grow_strength,
            drain_strength, acked, (uint32_t)cwnd_before, (uint32_t)ml_cc->cwnd_bytes);
}

void
xqc_ml_cc_feed_features(void *cong, xqc_usec_t ack_recv_time,
    xqc_usec_t adjusted_rtt, double short_loss_rate, unsigned short_lost_cnt,
    unsigned short_send_cnt, double long_loss_rate, unsigned long_lost_cnt,
    unsigned long_send_cnt, xqc_usec_t response_interval, uint64_t cwnd,
    unsigned pkt_in_fly)
{
    xqc_ml_cc_t *ml_cc = (xqc_ml_cc_t *)cong;
    float features[XQC_ML_CC_NUM_FEATURES];

    (void)ack_recv_time;

    if (ml_cc == NULL) {
        return;
    }

    memset(features, 0, sizeof(features));
    xqc_ml_cc_build_features(ml_cc, features, adjusted_rtt, short_loss_rate,
        short_lost_cnt, short_send_cnt, long_loss_rate, long_lost_cnt,
        long_send_cnt, response_interval, cwnd, pkt_in_fly);

    xqc_ml_cc_update_window(ml_cc, features);

    if (!ml_cc->is_frozen) {
        xqc_ml_cc_state_t prev_state = ml_cc->last_state;
        xqc_ml_cc_run_state_inference(ml_cc);
        ml_cc->last_state = xqc_ml_cc_determine_state(ml_cc);
        xqc_log(xqc_ml_cc_get_log(ml_cc), XQC_LOG_DEBUG,
                "|ml_cc|state_update|prev_state:%s|state:%s|"
                "prob_ut:%.3f|prob_qu:%.3f|prob_eb:%.3f|prob_nh:%.3f|"
                "adjusted_rtt:%ui|short_loss:%.2f|long_loss:%.2f|cwnd:%ui|pkt_in_fly:%ud|samples:%d|",
                xqc_ml_cc_state_to_str(prev_state),
                xqc_ml_cc_state_to_str(ml_cc->last_state),
                ml_cc->state_probs[XQC_ML_CC_STATE_UT],
                ml_cc->state_probs[XQC_ML_CC_STATE_QU],
                ml_cc->state_probs[XQC_ML_CC_STATE_EB],
                ml_cc->state_probs[XQC_ML_CC_STATE_NH],
                adjusted_rtt, short_loss_rate, long_loss_rate, (uint32_t)cwnd,
                pkt_in_fly, ml_cc->sample_count);
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
    ml_cc->init_cwnd_bytes = XQC_ML_CC_INIT_WIN;
    ml_cc->min_cwnd_bytes = XQC_ML_CC_MIN_CWND;

    if (cc_params.customize_on) {
        if (cc_params.init_cwnd > 0) {
            ml_cc->init_cwnd_bytes = cc_params.init_cwnd * XQC_ML_CC_MSS;
        }
        if (cc_params.min_cwnd > 0) {
            ml_cc->min_cwnd_bytes = cc_params.min_cwnd * XQC_ML_CC_MSS;
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
    ml_cc->frozen_cwnd = ml_cc->cwnd_bytes;
    ml_cc->loss_spike_during_freeze = XQC_FALSE;
    ml_cc->queue_depth_raw = 0.0f;
    ml_cc->queue_depth_ema = 0.0f;
    ml_cc->queue_threshold_k = XQC_ML_CC_QU_QUEUE_K;
    ml_cc->queue_model_enabled = 1;
    ml_cc->queue_model_ready = 0;
    ml_cc->state_model_ready = 0;

    memset(ml_cc->history_window, 0, sizeof(ml_cc->history_window));
    memset(ml_cc->state_probs, 0, sizeof(ml_cc->state_probs));

#ifdef XQC_ENABLE_ONNX
    ml_cc->onnx_api = (void *)OrtGetApiBase()->GetApi(ORT_API_VERSION);
    ml_cc->onnx_session = NULL;
    ml_cc->queue_onnx_session = NULL;

    if (ml_cc->onnx_api != NULL) {
        const OrtApi *api = (const OrtApi *)ml_cc->onnx_api;
        ml_cc->onnx_session = (void *)xqc_ml_cc_create_session(ml_cc, api,
            "state", XQC_ONNX_MODEL_PATH);
#ifdef XQC_ONNX_QUEUE_MODEL_PATH
        ml_cc->queue_onnx_session = (void *)xqc_ml_cc_create_session(ml_cc,
            api, "queue", XQC_ONNX_QUEUE_MODEL_PATH);
#endif
    } else {
        xqc_ml_cc_log_onnx_failure(ml_cc, "state", "init",
            "onnx_api_unavailable", NULL);
    }

    ml_cc->state_model_ready = ml_cc->onnx_session != NULL ? 1 : 0;
    ml_cc->queue_model_ready = ml_cc->queue_onnx_session != NULL ? 1 : 0;

    if (!ml_cc->state_model_ready) {
        xqc_ml_cc_log_onnx_failure(ml_cc, "state", "init",
            "state_model_session_unavailable", "output");
    }

    if (!ml_cc->queue_model_ready) {
        xqc_ml_cc_log_onnx_failure(ml_cc, "queue", "init",
            "queue_model_session_unavailable", "queue_depth_percentage");
    }
#else
    ml_cc->onnx_api = NULL;
    ml_cc->onnx_session = NULL;
    ml_cc->queue_onnx_session = NULL;
    ml_cc->state_model_ready = 0;
    ml_cc->queue_model_ready = 0;
#endif

    xqc_log(xqc_ml_cc_get_log(ml_cc), XQC_LOG_DEBUG,
            "|ml_cc|initialized|cwnd:%u|state_onnx:%d|queue_onnx:%d|queue_k:%.2f|",
            (uint32_t)ml_cc->cwnd_bytes,
            ml_cc->state_model_ready,
            ml_cc->queue_model_ready,
            ml_cc->queue_threshold_k);
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
            short_loss_rate = (double)ctl->ctl_recent_lost_count[0]
                / ctl->ctl_recent_send_count[0] * 100.0;
        }

        if (elapsed < XQC_ML_CC_NH_FREEZE_DURATION) {
            if (short_loss_rate > XQC_ML_CC_NH_LOSS_HIGH) {
                ml_cc->loss_spike_during_freeze = XQC_TRUE;
            }
        } else {
            if (!ml_cc->loss_spike_during_freeze) {
                ml_cc->is_frozen = XQC_FALSE;
                xqc_log(xqc_ml_cc_get_log(ml_cc), XQC_LOG_DEBUG,
                        "|ml_cc|frozen_exit|elapsed:%ui|no_loss_spike|",
                        (uint32_t)elapsed);
            } else if (short_loss_rate < XQC_ML_CC_NH_LOSS_LOW) {
                ml_cc->is_frozen = XQC_FALSE;
                ml_cc->loss_spike_during_freeze = XQC_FALSE;
                xqc_log(xqc_ml_cc_get_log(ml_cc), XQC_LOG_DEBUG,
                        "|ml_cc|frozen_exit|elapsed:%ui|loss_below_threshold|",
                        (uint32_t)elapsed);
            }
        }

        if (ml_cc->is_frozen) {
            ml_cc->cwnd_bytes = ml_cc->frozen_cwnd;
            xqc_log(xqc_ml_cc_get_log(ml_cc), XQC_LOG_DEBUG,
                    "|ml_cc|frozen|elapsed:%ui|loss:%.2f|spike:%d|cwnd:%u|",
                    (uint32_t)elapsed, short_loss_rate,
                    ml_cc->loss_spike_during_freeze ? 1 : 0,
                    (uint32_t)ml_cc->cwnd_bytes);
            return;
        }
    }

    if (ml_cc->last_state == XQC_ML_CC_STATE_NH) {
        ml_cc->frozen_cwnd = ml_cc->cwnd_bytes * XQC_ML_CC_NH_FREEZE_CWND_FACTOR;
        ml_cc->cwnd_bytes = ml_cc->frozen_cwnd;
        ml_cc->is_frozen = XQC_TRUE;
        ml_cc->freeze_start_time = now;
        ml_cc->loss_spike_during_freeze = XQC_FALSE;

        xqc_log(xqc_ml_cc_get_log(ml_cc), XQC_LOG_DEBUG,
                "|ml_cc|nh_enter|frozen_cwnd:%u|prob_nh:%.3f|",
                (uint32_t)ml_cc->frozen_cwnd, ml_cc->state_probs[3]);
        return;
    }

    if (ml_cc->last_state == XQC_ML_CC_STATE_UT) {
        float acked = xqc_ml_cc_get_acked_bytes(po);
        ml_cc->qu_consecutive_count = 0;
        ml_cc->cwnd_bytes += acked * XQC_ML_CC_UT_CWND_GAIN;
        xqc_ml_cc_clamp_cwnd(ml_cc);

        xqc_log(xqc_ml_cc_get_log(ml_cc), XQC_LOG_DEBUG,
                "|ml_cc|ut_state|gain:%.2f|acked:%.1f|cwnd:%u|",
                XQC_ML_CC_UT_CWND_GAIN, acked, (uint32_t)ml_cc->cwnd_bytes);
        return;
    }

    if (ml_cc->last_state == XQC_ML_CC_STATE_QU) {
        float queue_depth = 0.0f;

        if (ml_cc->queue_model_enabled
            && ml_cc->queue_model_ready
            && ml_cc->state_probs[XQC_ML_CC_STATE_QU] >= XQC_ML_CC_QU_CONFIDENCE_THRESHOLD
            && xqc_ml_cc_run_qu_queue_inference(ml_cc, &queue_depth))
        {
            float prev_queue_ema = ml_cc->queue_depth_ema;
            ml_cc->queue_depth_raw = queue_depth;
            ml_cc->queue_depth_ema = xqc_ml_cc_clamp_float(
                (1.0f - XQC_ML_CC_QU_QUEUE_EMA_ALPHA) * ml_cc->queue_depth_ema
                + XQC_ML_CC_QU_QUEUE_EMA_ALPHA * queue_depth,
                0.0f, 1.0f);
            xqc_log(xqc_ml_cc_get_log(ml_cc), XQC_LOG_DEBUG,
                    "|ml_cc|queue_update|state:%s|prob_qu:%.3f|queue_raw:%.3f|"
                    "queue_ema_prev:%.3f|queue_ema:%.3f|queue_k:%.3f|",
                    xqc_ml_cc_state_to_str(ml_cc->last_state),
                    ml_cc->state_probs[XQC_ML_CC_STATE_QU],
                    ml_cc->queue_depth_raw, prev_queue_ema,
                    ml_cc->queue_depth_ema, ml_cc->queue_threshold_k);
            xqc_ml_cc_apply_qu_fine_grained_control(ml_cc, po);
        } else {
            xqc_ml_cc_apply_qu_fallback(ml_cc);
        }

        return;
    }

    if (ml_cc->last_state == XQC_ML_CC_STATE_EB) {
        ml_cc->qu_consecutive_count = 0;
        ml_cc->cwnd_bytes *= XQC_ML_CC_EB_CWND_DECREASE;
        xqc_ml_cc_clamp_cwnd(ml_cc);

        xqc_log(xqc_ml_cc_get_log(ml_cc), XQC_LOG_DEBUG,
                "|ml_cc|eb_state|decrease|cwnd:%u|",
                (uint32_t)ml_cc->cwnd_bytes);
        return;
    }
}

static void
xqc_ml_cc_on_lost(void *cong, xqc_usec_t lost_sent_time)
{
    (void)cong;
    (void)lost_sent_time;
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
    (void)cong;
    return 0;
}

static void
xqc_ml_cc_restart_from_idle(void *cong, uint64_t arg)
{
    xqc_ml_cc_t *ml_cc = (xqc_ml_cc_t *)cong;
    (void)arg;
    ml_cc->cwnd_bytes = xqc_max(ml_cc->cwnd_bytes, ml_cc->init_cwnd_bytes);
}

const xqc_cong_ctrl_callback_t xqc_ml_cc_cb = {
    .xqc_cong_ctl_size              = xqc_ml_cc_size,
    .xqc_cong_ctl_init              = xqc_ml_cc_init,
    .xqc_cong_ctl_on_lost           = xqc_ml_cc_on_lost,
    .xqc_cong_ctl_on_ack            = xqc_ml_cc_on_ack,
    .xqc_cong_ctl_get_cwnd          = xqc_ml_cc_get_cwnd,
    .xqc_cong_ctl_reset_cwnd        = xqc_ml_cc_reset_cwnd,
    .xqc_cong_ctl_in_slow_start     = xqc_ml_cc_in_slow_start,
    .xqc_cong_ctl_in_recovery       = xqc_ml_cc_in_recovery,
    .xqc_cong_ctl_restart_from_idle = xqc_ml_cc_restart_from_idle,
    .xqc_cong_ctl_info_cb           = NULL,
};
