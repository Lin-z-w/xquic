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

typedef struct {
    float                   adjusted_rtt;
    float                   short_loss_rate;
    float                   long_loss_rate;
    float                   response_interval;
    float                   cwnd;
    float                   pkt_in_fly;
    float                   rtt_delta;
    float                   cwnd_delta;
    float                   pkt_in_fly_delta;
    float                   cwnd_per_rtt;
    float                   pkt_per_rtt;
    float                   pkt_per_cwnd;
    float                   adjusted_rtt_roll_mean;
    float                   adjusted_rtt_roll_std;
    float                   cwnd_roll_mean;
    float                   cwnd_roll_std;
    float                   pkt_in_fly_roll_mean;
    float                   pkt_in_fly_roll_std;
} xqc_ml_features_t;

static int
xqc_ml_cc_load_npy(const char *filename, float *data, int expected_size)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return -1;
    }

    char header[256];
    if (fgets(header, sizeof(header), fp) == NULL) {
        fclose(fp);
        return -1;
    }

    int dims = 0;
    int shape[4] = {0};
    char *p = strstr(header, "(");
    if (p) {
        char *q = p + 1;
        while (*q && *q != ')' && dims < 4) {
            if (*q >= '0' && *q <= '9') {
                shape[dims] = shape[dims] * 10 + (*q - '0');
            } else if (*q == ',') {
                dims++;
            }
            q++;
        }
        if (*q == ')') dims++;
    }

    int total = 1;
    for (int i = 0; i < dims; i++) total *= shape[i];

    if (total != expected_size) {
        fclose(fp);
        return -1;
    }

    fread(data, sizeof(float), total, fp);
    fclose(fp);

    return 0;
}

static void
xqc_ml_cc_extract_features(xqc_ml_cc_t *ml_cc, xqc_ml_features_t *features)
{
    xqc_send_ctl_t *ctl = ml_cc->send_ctl;
    if (!ctl) {
        memset(features, 0, sizeof(xqc_ml_features_t));
        return;
    }

    float adjusted_rtt = (float)ctl->ctl_srtt / 1000.0f;
    float cwnd_bytes = (float)ml_cc->cwnd_bytes;
    float pkt_in_fly = (float)ctl->ctl_bytes_in_flight;
    float cwnd_pkt = cwnd_bytes / XQC_ML_CC_MSS;
    float pkt_in_fly_pkt = pkt_in_fly / XQC_ML_CC_MSS;

    float short_loss_rate = 0.0f;
    float long_loss_rate = 0.0f;
    if (ctl->ctl_send_count > 0) {
        short_loss_rate = (float)ctl->ctl_recent_lost_count[0] / 
                          xqc_max(1, ctl->ctl_recent_send_count[0]);
        long_loss_rate = (float)ctl->ctl_lost_count / 
                        xqc_max(1, ctl->ctl_send_count);
    }

    float response_interval = 0.0f;
    if (ctl->ctl_delivered_time > 0 && ctl->ctl_first_sent_time > 0) {
        response_interval = (float)(ctl->ctl_delivered_time - ctl->ctl_first_sent_time) / 1000.0f;
    }

    features->adjusted_rtt = adjusted_rtt;
    features->short_loss_rate = short_loss_rate;
    features->long_loss_rate = long_loss_rate;
    features->response_interval = response_interval;
    features->cwnd = cwnd_bytes;
    features->pkt_in_fly = pkt_in_fly;

    float prev_cwnd = 0.0f, prev_pkt_in_fly = 0.0f, prev_rtt = 0.0f;
    if (ml_cc->sample_count > 0) {
        int prev_idx = (ml_cc->window_idx - 1 + XQC_ML_CC_WINDOW_SIZE) % XQC_ML_CC_WINDOW_SIZE;
        prev_cwnd = ml_cc->history_window[prev_idx][4];
        prev_pkt_in_fly = ml_cc->history_window[prev_idx][5];
        prev_rtt = ml_cc->history_window[prev_idx][0];
    }
    features->rtt_delta = adjusted_rtt - prev_rtt;
    features->cwnd_delta = cwnd_bytes - prev_cwnd;
    features->pkt_in_fly_delta = pkt_in_fly - prev_pkt_in_fly;

    features->cwnd_per_rtt = (adjusted_rtt > 0) ? (cwnd_bytes / adjusted_rtt) : 0.0f;
    features->pkt_per_rtt = (adjusted_rtt > 0) ? (pkt_in_fly_pkt / adjusted_rtt) : 0.0f;
    features->pkt_per_cwnd = (cwnd_pkt > 0) ? (pkt_in_fly_pkt / cwnd_pkt) : 0.0f;

    float rtt_sum = 0.0f, rtt_sq_sum = 0.0f;
    float cwnd_sum = 0.0f, cwnd_sq_sum = 0.0f;
    float pkt_sum = 0.0f, pkt_sq_sum = 0.0f;
    int count = xqc_min(ml_cc->sample_count, XQC_ML_CC_WINDOW_SIZE);

    for (int i = 0; i < count; i++) {
        int idx = (ml_cc->window_idx - i - 1 + XQC_ML_CC_WINDOW_SIZE) % XQC_ML_CC_WINDOW_SIZE;
        rtt_sum += ml_cc->history_window[idx][0];
        rtt_sq_sum += ml_cc->history_window[idx][0] * ml_cc->history_window[idx][0];
        cwnd_sum += ml_cc->history_window[idx][4];
        cwnd_sq_sum += ml_cc->history_window[idx][4] * ml_cc->history_window[idx][4];
        pkt_sum += ml_cc->history_window[idx][5];
        pkt_sq_sum += ml_cc->history_window[idx][5] * ml_cc->history_window[idx][5];
    }

    float rtt_mean = (count > 0) ? (rtt_sum / count) : 0.0f;
    float cwnd_mean = (count > 0) ? (cwnd_sum / count) : 0.0f;
    float pkt_mean = (count > 0) ? (pkt_sum / count) : 0.0f;

    float rtt_var = (count > 1) ? ((rtt_sq_sum - rtt_sum * rtt_sum / count) / count) : 0.0f;
    float cwnd_var = (count > 1) ? ((cwnd_sq_sum - cwnd_sum * cwnd_sum / count) / count) : 0.0f;
    float pkt_var = (count > 1) ? ((pkt_sq_sum - pkt_sum * pkt_sum / count) / count) : 0.0f;

    features->adjusted_rtt_roll_mean = rtt_mean;
    features->adjusted_rtt_roll_std = sqrtf(xqc_max(0.0f, rtt_var));
    features->cwnd_roll_mean = cwnd_mean;
    features->cwnd_roll_std = sqrtf(xqc_max(0.0f, cwnd_var));
    features->pkt_in_fly_roll_mean = pkt_mean;
    features->pkt_in_fly_roll_std = sqrtf(xqc_max(0.0f, pkt_var));
}

static void
xqc_ml_cc_update_window(xqc_ml_cc_t *ml_cc, xqc_ml_features_t *features)
{
    float *row = ml_cc->history_window[ml_cc->window_idx];
    row[0] = features->adjusted_rtt;
    row[1] = features->short_loss_rate;
    row[2] = features->long_loss_rate;
    row[3] = features->response_interval;
    row[4] = features->cwnd;
    row[5] = features->pkt_in_fly;
    row[6] = features->rtt_delta;
    row[7] = features->cwnd_delta;
    row[8] = features->pkt_in_fly_delta;
    row[9] = features->cwnd_per_rtt;
    row[10] = features->pkt_per_rtt;
    row[11] = features->pkt_per_cwnd;
    row[12] = features->adjusted_rtt_roll_mean;
    row[13] = features->adjusted_rtt_roll_std;
    row[14] = features->cwnd_roll_mean;
    row[15] = features->cwnd_roll_std;
    row[16] = features->pkt_in_fly_roll_mean;
    row[17] = features->pkt_in_fly_roll_std;

    ml_cc->window_idx = (ml_cc->window_idx + 1) % XQC_ML_CC_WINDOW_SIZE;
    ml_cc->sample_count++;
}

static void
xqc_ml_cc_normalize_features(float *normalized, float *raw, float *mean, float *scale, int size)
{
    for (int i = 0; i < size; i++) {
        if (scale[i] > 1e-6f) {
            normalized[i] = (raw[i] - mean[i]) / scale[i];
        } else {
            normalized[i] = raw[i] - mean[i];
        }
    }
}

#ifdef XQC_ENABLE_ONNX
static float
xqc_ml_cc_run_inference(xqc_ml_cc_t *ml_cc)
{
    if (!ml_cc->onnx_session || !ml_cc->onnx_api) {
        return 50.0f;
    }

    xqc_ml_features_t features;
    xqc_ml_cc_extract_features(ml_cc, &features);

    float raw_features[XQC_ML_CC_NUM_FEATURES];
    raw_features[0] = features.adjusted_rtt;
    raw_features[1] = features.short_loss_rate;
    raw_features[2] = features.long_loss_rate;
    raw_features[3] = features.response_interval;
    raw_features[4] = features.cwnd;
    raw_features[5] = features.pkt_in_fly;
    raw_features[6] = features.rtt_delta;
    raw_features[7] = features.cwnd_delta;
    raw_features[8] = features.pkt_in_fly_delta;
    raw_features[9] = features.cwnd_per_rtt;
    raw_features[10] = features.pkt_per_rtt;
    raw_features[11] = features.pkt_per_cwnd;
    raw_features[12] = features.adjusted_rtt_roll_mean;
    raw_features[13] = features.adjusted_rtt_roll_std;
    raw_features[14] = features.cwnd_roll_mean;
    raw_features[15] = features.cwnd_roll_std;
    raw_features[16] = features.pkt_in_fly_roll_mean;
    raw_features[17] = features.pkt_in_fly_roll_std;

    xqc_ml_cc_update_window(ml_cc, &features);

    if (ml_cc->sample_count < XQC_ML_CC_WINDOW_SIZE) {
        return 50.0f;
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
        return 50.0f;
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
        return 50.0f;
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
        return 50.0f;
    }

    float *output_data = NULL;
    status = api->GetTensorMutableData(output_tensor, (void **)&output_data);
    float prediction = 50.0f;
    if (status == NULL && output_data != NULL) {
        prediction = output_data[0];
    }

    api->ReleaseValue(output_tensor);

    if (prediction < 0) prediction = 0;
    if (prediction > 100) prediction = 100;

    return prediction;
}
#else
static float
xqc_ml_cc_run_inference(xqc_ml_cc_t *ml_cc)
{
    xqc_ml_features_t features;
    xqc_ml_cc_extract_features(ml_cc, &features);
    xqc_ml_cc_update_window(ml_cc, &features);

    float base_prediction = 50.0f;
    float queue_factor = features.pkt_in_fly / xqc_max(1, features.cwnd);
    float loss_factor = 1.0f + features.short_loss_rate + features.long_loss_rate;
    float rtt_factor = features.adjusted_rtt / 100.0f;

    float prediction = base_prediction * queue_factor * loss_factor * rtt_factor;
    prediction = xqc_max(0.0f, xqc_min(100.0f, prediction));

    return prediction;
}
#endif

void
xqc_ml_cc_feed_features(void *cong, xqc_usec_t ack_recv_time,
    xqc_usec_t adjusted_rtt, double short_loss_rate, double long_loss_rate,
    xqc_usec_t response_interval, uint64_t cwnd, unsigned pkt_in_fly)
{
    xqc_ml_cc_t *ml_cc = (xqc_ml_cc_t *)cong;
    if (!ml_cc) {
        return;
    }

    xqc_ml_features_t features;
    memset(&features, 0, sizeof(features));

    features.adjusted_rtt = (float)adjusted_rtt / 1000.0f;
    features.short_loss_rate = (float)short_loss_rate;
    features.long_loss_rate = (float)long_loss_rate;
    features.response_interval = (float)response_interval / 1000.0f;
    features.cwnd = (float)cwnd;
    features.pkt_in_fly = (float)pkt_in_fly;

    float prev_cwnd = 0.0f, prev_pkt_in_fly = 0.0f, prev_rtt = 0.0f;
    if (ml_cc->sample_count > 0) {
        int prev_idx = (ml_cc->window_idx - 1 + XQC_ML_CC_WINDOW_SIZE) % XQC_ML_CC_WINDOW_SIZE;
        prev_cwnd = ml_cc->history_window[prev_idx][4];
        prev_pkt_in_fly = ml_cc->history_window[prev_idx][5];
        prev_rtt = ml_cc->history_window[prev_idx][0];
    }
    features.rtt_delta = features.adjusted_rtt - prev_rtt;
    features.cwnd_delta = features.cwnd - prev_cwnd;
    features.pkt_in_fly_delta = features.pkt_in_fly - prev_pkt_in_fly;

    float cwnd_pkt = features.cwnd / XQC_ML_CC_MSS;
    float pkt_in_fly_pkt = features.pkt_in_fly / XQC_ML_CC_MSS;
    features.cwnd_per_rtt = (features.adjusted_rtt > 0) ? (features.cwnd / features.adjusted_rtt) : 0.0f;
    features.pkt_per_rtt = (features.adjusted_rtt > 0) ? (pkt_in_fly_pkt / features.adjusted_rtt) : 0.0f;
    features.pkt_per_cwnd = (cwnd_pkt > 0) ? (pkt_in_fly_pkt / cwnd_pkt) : 0.0f;

    int count = xqc_min(ml_cc->sample_count, XQC_ML_CC_WINDOW_SIZE);
    float rtt_sum = 0.0f, rtt_sq_sum = 0.0f;
    float ccwnd_sum = 0.0f, ccwnd_sq_sum = 0.0f;
    float pkt_sum = 0.0f, pkt_sq_sum = 0.0f;

    for (int i = 0; i < count; i++) {
        int idx = (ml_cc->window_idx - i - 1 + XQC_ML_CC_WINDOW_SIZE) % XQC_ML_CC_WINDOW_SIZE;
        rtt_sum += ml_cc->history_window[idx][0];
        rtt_sq_sum += ml_cc->history_window[idx][0] * ml_cc->history_window[idx][0];
        ccwnd_sum += ml_cc->history_window[idx][4];
        ccwnd_sq_sum += ml_cc->history_window[idx][4] * ml_cc->history_window[idx][4];
        pkt_sum += ml_cc->history_window[idx][5];
        pkt_sq_sum += ml_cc->history_window[idx][5] * ml_cc->history_window[idx][5];
    }

    float rtt_mean = (count > 0) ? (rtt_sum / count) : 0.0f;
    float ccwnd_mean = (count > 0) ? (ccwnd_sum / count) : 0.0f;
    float pkt_mean = (count > 0) ? (pkt_sum / count) : 0.0f;

    float rtt_var = (count > 1) ? ((rtt_sq_sum - rtt_sum * rtt_sum / count) / count) : 0.0f;
    float ccwnd_var = (count > 1) ? ((ccwnd_sq_sum - ccwnd_sum * ccwnd_sum / count) / count) : 0.0f;
    float pkt_var = (count > 1) ? ((pkt_sq_sum - pkt_sum * pkt_sum / count) / count) : 0.0f;

    features.adjusted_rtt_roll_mean = rtt_mean;
    features.adjusted_rtt_roll_std = sqrtf(xqc_max(0.0f, rtt_var));
    features.cwnd_roll_mean = ccwnd_mean;
    features.cwnd_roll_std = sqrtf(xqc_max(0.0f, ccwnd_var));
    features.pkt_in_fly_roll_mean = pkt_mean;
    features.pkt_in_fly_roll_std = sqrtf(xqc_max(0.0f, pkt_var));

    xqc_ml_cc_update_window(ml_cc, &features);

    ml_cc->last_prediction = xqc_ml_cc_run_inference(ml_cc);
    ml_cc->last_rtt = adjusted_rtt;

    xqc_log(ml_cc->log, XQC_LOG_REPORT,
        "|ml_cc|feed_features|rtt:%.2f|loss:%.2f/%.2f|cwnd:%.0f|pkt_fly:%.0f|queue_pct:%.2f|recovery_elapsed:%ui|",
        features.adjusted_rtt, features.short_loss_rate, features.long_loss_rate,
        features.cwnd, features.pkt_in_fly, ml_cc->last_prediction,
        (ml_cc->congestion_recovery_start_time > 0) ? 
            (uint32_t)((xqc_monotonic_timestamp() - ml_cc->congestion_recovery_start_time) / 1000) : 0);
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

    memset(ml_cc->history_window, 0, sizeof(ml_cc->history_window));

    for (int i = 0; i < XQC_ML_CC_NUM_FEATURES; i++) {
        ml_cc->scaler_mean[i] = 0.0f;
        ml_cc->scaler_scale[i] = 1.0f;
    }

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

    if (ml_cc->onnx_session) {
        char model_path_mean[256];
        char model_path_scale[256];
        snprintf(model_path_mean, sizeof(model_path_mean), 
                 "%s/scaler_mean.npy", XQC_ONNX_MODEL_DIR);
        snprintf(model_path_scale, sizeof(model_path_scale), 
                 "%s/scaler_scale.npy", XQC_ONNX_MODEL_DIR);

        xqc_ml_cc_load_npy(model_path_mean, ml_cc->scaler_mean, XQC_ML_CC_NUM_FEATURES);
        xqc_ml_cc_load_npy(model_path_scale, ml_cc->scaler_scale, XQC_ML_CC_NUM_FEATURES);
    }
#else
    ml_cc->onnx_session = NULL;
    ml_cc->onnx_api = NULL;
#endif

    xqc_log(ml_cc->log, XQC_LOG_DEBUG,
            "|ml_cc|initialized|cwnd:%u|onnx:%d|threshold_loss:%.1f|threshold_k:%.1f|recovery_rtt:%d|",
            (uint32_t)ml_cc->cwnd_bytes,
            ml_cc->onnx_session != NULL ? 1 : 0,
            XQC_ML_CC_QUEUE_THRESHOLD_LOSS,
            XQC_ML_CC_QUEUE_THRESHOLD_K,
            XQC_ML_CC_RECOVERY_RTT);
}

static void
xqc_ml_cc_on_ack(void *cong, xqc_packet_out_t *po, xqc_usec_t now)
{
    xqc_ml_cc_t *ml_cc = (xqc_ml_cc_t *)cong;

    xqc_usec_t elapsed = 0;
    if (ml_cc->congestion_recovery_start_time > 0) {
        elapsed = now - ml_cc->congestion_recovery_start_time;
    }

    if (ml_cc->last_rtt > 0 && 
        elapsed < ml_cc->last_rtt * XQC_ML_CC_RECOVERY_RTT) {
        xqc_log(ml_cc->log, XQC_LOG_DEBUG,
                "|ml_cc|on_ack|in_recovery|elapsed:%ui|recovery_window:%ui|cwnd:%u|",
                (uint32_t)elapsed, 
                (uint32_t)(ml_cc->last_rtt * XQC_ML_CC_RECOVERY_RTT),
                (uint32_t)ml_cc->cwnd_bytes);
        return;
    }

    float queue_pct = ml_cc->last_prediction;
    float cwnd_gain = 1.0f;

    if (queue_pct <= XQC_ML_CC_QUEUE_THRESHOLD_LOSS) {
        cwnd_gain = 1.0f;

    } else if (queue_pct < XQC_ML_CC_QUEUE_THRESHOLD_K) {
        float t = (queue_pct - XQC_ML_CC_QUEUE_THRESHOLD_LOSS) / 
                  (XQC_ML_CC_QUEUE_THRESHOLD_K - XQC_ML_CC_QUEUE_THRESHOLD_LOSS);
        cwnd_gain = 1.1f - t * 0.1f;

    } else {
        float t = (queue_pct - XQC_ML_CC_QUEUE_THRESHOLD_K) / 
                  (100.0f - XQC_ML_CC_QUEUE_THRESHOLD_K);
        cwnd_gain = 1.0f - t * 0.5f;
        cwnd_gain = xqc_max(cwnd_gain, 0.5f);
    }

    float ack_bytes = 0.0f;
    if (po) {
        ack_bytes = (float)po->po_used_size;
    }

    float target_cwnd = ml_cc->cwnd_bytes + ack_bytes * cwnd_gain;
    target_cwnd = xqc_min(target_cwnd, (float)XQC_ML_CC_MAX_CWND);

    ml_cc->cwnd_bytes = target_cwnd;

    xqc_log(ml_cc->log, XQC_LOG_DEBUG,
            "|ml_cc|on_ack|queue_pct:%.2f|gain:%.3f|cwnd:%u|ack_bytes:%u|recovery:%d|",
            queue_pct, cwnd_gain, (uint32_t)ml_cc->cwnd_bytes, 
            (uint32_t)ack_bytes, 
            (ml_cc->congestion_recovery_start_time > 0 && 
             elapsed < ml_cc->last_rtt * XQC_ML_CC_RECOVERY_RTT) ? 1 : 0);
}

static void
xqc_ml_cc_on_lost(void *cong, xqc_usec_t lost_sent_time)
{
    xqc_ml_cc_t *ml_cc = (xqc_ml_cc_t *)cong;

    float queue_pct = ml_cc->last_prediction;

    if (queue_pct > XQC_ML_CC_QUEUE_THRESHOLD_LOSS) {
        float t = (queue_pct - XQC_ML_CC_QUEUE_THRESHOLD_LOSS) / 
                  (100.0f - XQC_ML_CC_QUEUE_THRESHOLD_LOSS);
        float reduction_factor = 0.7f + t * 0.25f;
        reduction_factor = xqc_min(reduction_factor, 0.95f);

        ml_cc->loss_reduction_factor = reduction_factor;
        ml_cc->cwnd_bytes = xqc_max(
            ml_cc->min_cwnd_bytes,
            ml_cc->cwnd_bytes * reduction_factor);

        ml_cc->congestion_recovery_start_time = xqc_monotonic_timestamp();

        xqc_log(ml_cc->log, XQC_LOG_DEBUG,
                "|ml_cc|on_lost|reduction:%.3f|cwnd:%u|queue_pct:%.2f|recovery_start|",
                reduction_factor, (uint32_t)ml_cc->cwnd_bytes, queue_pct);

    } else {
        xqc_log(ml_cc->log, XQC_LOG_DEBUG,
                "|ml_cc|on_lost|skip|cwnd:%u|queue_pct:%.2f|below_threshold|",
                (uint32_t)ml_cc->cwnd_bytes, queue_pct);
    }
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