/**
 * @copyright Copyright (c) 2024, Alibaba Group Holding Limited
 * @brief ML Model Implementation for Congestion Control
 */

#include <string.h>
#include <math.h>
#include "src/congestion_control/xqc_ml_model.h"
#include "src/common/xqc_malloc.h"
#include "src/common/xqc_log.h"

#ifdef XQC_ENABLE_ONNX
#include <onnxruntime_c_api.h>
#endif

/* Feature scaler constants - from sklearn StandardScaler on training data */
const float xqc_ml_scaler_mean[XQC_ML_NUM_FEATURES] = {
    8.96128915e+04f, 6.79938659e+00f, 1.30212317e+02f, 1.61393684e+03f,
    6.74786114e+00f, 3.83006478e+02f, 4.73249606e+03f, 3.85382679e+03f,
    1.62961778e+06f, 6.50594380e+02f, 1.62115443e+00f, 1.00222519e+00f,
    5.18842275e-04f, 1.89891871e+03f, 5.15254543e-02f
};

const float xqc_ml_scaler_scale[XQC_ML_NUM_FEATURES] = {
    5.86602124e+04f, 1.58179157e+01f, 4.45019329e+02f, 1.69024452e+03f,
    1.32390146e+01f, 9.93498002e+02f, 3.69058068e+03f, 1.17108226e+04f,
    1.48773062e+06f, 5.11887834e+02f, 5.66852703e+03f, 7.30771491e-02f,
    2.20059070e-04f, 2.61691161e+03f, 9.01346935e+00f
};

#ifdef XQC_ENABLE_ONNX

/* Shared ONNX Runtime environment (one per process) */
static OrtEnv *xqc_ml_ort_env = NULL;
static int xqc_ml_ort_ref_count = 0;

/* Model structure */
struct xqc_ml_model_s {
    OrtSession      *session;
    xqc_ml_model_type_t type;
    int              use_scaler;
    const float     *scaler_mean;
    const float     *scaler_scale;
    char            *input_name;
    char            *output_name;
};

static void
xqc_ml_log_onnx_error(xqc_log_t *log, const char *phase, const char *detail)
{
    if (log != NULL) {
        xqc_log(log, XQC_LOG_WARN, "|ml_model|onnx_error|phase:%s|detail:%s|", 
                phase ? phase : "unknown", detail ? detail : "unknown");
    }
}

xqc_int_t
xqc_ml_model_global_init(void)
{
    xqc_ml_ort_ref_count++;
    return XQC_OK;
}

void
xqc_ml_model_global_cleanup(void)
{
    xqc_ml_ort_ref_count--;
    if (xqc_ml_ort_ref_count <= 0 && xqc_ml_ort_env != NULL) {
        const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        if (api != NULL) {
            api->ReleaseEnv(xqc_ml_ort_env);
        }
        xqc_ml_ort_env = NULL;
        xqc_ml_ort_ref_count = 0;
    }
}

static OrtEnv *
xqc_ml_get_or_create_env(xqc_log_t *log)
{
    const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtStatus *status = NULL;

    if (api == NULL) {
        xqc_ml_log_onnx_error(log, "get_api", "null_api_base");
        return NULL;
    }

    if (xqc_ml_ort_env != NULL) {
        return xqc_ml_ort_env;
    }

    status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "xqc_ml_model", 
                            &xqc_ml_ort_env);
    if (status != NULL) {
        xqc_ml_log_onnx_error(log, "create_env", 
            api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return NULL;
    }

    return xqc_ml_ort_env;
}

xqc_ml_model_t *
xqc_ml_model_load(const xqc_ml_model_config_t *config, xqc_log_t *log)
{
    const OrtApi *api;
    OrtEnv *env;
    OrtSessionOptions *options = NULL;
    OrtSession *session = NULL;
    OrtStatus *status = NULL;
    xqc_ml_model_t *model = NULL;

    if (config == NULL || config->model_path == NULL) {
        xqc_ml_log_onnx_error(log, "load", "null_config");
        return NULL;
    }

    api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (api == NULL) {
        xqc_ml_log_onnx_error(log, "load", "null_api");
        return NULL;
    }

    env = xqc_ml_get_or_create_env(log);
    if (env == NULL) {
        return NULL;
    }

    status = api->CreateSessionOptions(&options);
    if (status != NULL) {
        xqc_ml_log_onnx_error(log, "session_options", 
            api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return NULL;
    }

    status = api->CreateSession(env, config->model_path, options, &session);
    api->ReleaseSessionOptions(options);

    if (status != NULL) {
        xqc_ml_log_onnx_error(log, "create_session", 
            api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        return NULL;
    }

    model = (xqc_ml_model_t *)xqc_malloc(sizeof(xqc_ml_model_t));
    if (model == NULL) {
        api->ReleaseSession(session);
        return NULL;
    }

    memset(model, 0, sizeof(xqc_ml_model_t));
    model->session = session;
    model->type = config->type;
    model->use_scaler = config->use_scaler;
    model->scaler_mean = config->scaler_mean;
    model->scaler_scale = config->scaler_scale;
    model->input_name = xqc_malloc(6);
    model->output_name = xqc_malloc(7);
    if (model->input_name) strcpy(model->input_name, "input");
    if (model->output_name) strcpy(model->output_name, "output");

    xqc_log(log, XQC_LOG_INFO, "|ml_model|loaded|path:%s|type:%d|", 
            config->model_path, config->type);

    return model;
}

void
xqc_ml_model_unload(xqc_ml_model_t *model)
{
    const OrtApi *api;

    if (model == NULL) {
        return;
    }

    api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (api != NULL && model->session != NULL) {
        api->ReleaseSession(model->session);
    }

    if (model->input_name) xqc_free(model->input_name);
    if (model->output_name) xqc_free(model->output_name);
    xqc_free(model);
}

xqc_int_t
xqc_ml_model_infer(xqc_ml_model_t *model,
    const float window[XQC_ML_WINDOW_SIZE][XQC_ML_NUM_FEATURES],
    xqc_ml_output_t *output, xqc_log_t *log)
{
    const OrtApi *api;
    OrtMemoryInfo *mem_info = NULL;
    OrtValue *input_tensor = NULL;
    OrtValue *output_tensor = NULL;
    OrtStatus *status = NULL;
    float input[XQC_ML_WINDOW_SIZE][XQC_ML_NUM_FEATURES];
    int64_t input_shape[] = {1, XQC_ML_WINDOW_SIZE, XQC_ML_NUM_FEATURES};
    float *output_data = NULL;
    float logits[XQC_ML_NUM_STATES];
    int t, f;
    xqc_int_t ret = -XQC_ERROR;
    const char *input_names[] = {"input"};
    const char *output_names[] = {"output"};

    if (model == NULL || model->session == NULL || output == NULL) {
        xqc_ml_log_onnx_error(log, "infer", "null_args");
        return -XQC_EPARAM;
    }
    
    api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (api == NULL) {
        return -XQC_ERROR;
    }

    /* Copy and optionally normalize input window */
    for (t = 0; t < XQC_ML_WINDOW_SIZE; t++) {
        for (f = 0; f < XQC_ML_NUM_FEATURES; f++) {
            input[t][f] = window[t][f];
            if (!isfinite(input[t][f])) {
                input[t][f] = 0.0f;
            }
            /* Apply scaler if configured */
            if (model->use_scaler && model->scaler_mean && model->scaler_scale) {
                input[t][f] = (input[t][f] - model->scaler_mean[f]) 
                              / model->scaler_scale[f];
            }
        }
    }

    status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, 
                                      &mem_info);
    if (status != NULL) goto cleanup;

    status = api->CreateTensorWithDataAsOrtValue(mem_info, input, sizeof(input),
        input_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    if (status != NULL) goto cleanup;

    status = api->Run(model->session, NULL, input_names, 
        (const OrtValue * const*)&input_tensor, 1,
        output_names, 1, &output_tensor);
    if (status != NULL) goto cleanup;

    status = api->GetTensorMutableData(output_tensor, (void **)&output_data);
    if (status != NULL || output_data == NULL) goto cleanup;

    /* Process output based on model type */
    if (model->type == XQC_ML_MODEL_STATE) {
        /* State classification: 4-class output */
        logits[0] = output_data[0];
        logits[1] = output_data[1];
        logits[2] = output_data[2];
        logits[3] = output_data[3];
        
        if (!isfinite(logits[0]) || !isfinite(logits[1]) || 
            !isfinite(logits[2]) || !isfinite(logits[3])) {
            goto cleanup;
        }

        xqc_ml_softmax(logits, output->state_probs, XQC_ML_NUM_STATES);
        
        /* Find argmax */
        output->predicted_state = XQC_ML_STATE_UT;
        float max_prob = output->state_probs[0];
        for (f = 1; f < XQC_ML_NUM_STATES; f++) {
            if (output->state_probs[f] > max_prob) {
                max_prob = output->state_probs[f];
                output->predicted_state = (xqc_ml_state_t)f;
            }
        }
        output->valid = 1;
    } else {
        /* Queue regression: single value output */
        output->queue_depth = output_data[0];
        if (!isfinite(output->queue_depth)) {
            output->queue_depth = 0.0f;
        }
        output->valid = 1;
    }

    ret = XQC_OK;

cleanup:
    if (status != NULL) {
        xqc_ml_log_onnx_error(log, "infer_run", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
    }
    if (output_tensor != NULL) api->ReleaseValue(output_tensor);
    if (input_tensor != NULL) api->ReleaseValue(input_tensor);
    if (mem_info != NULL) api->ReleaseMemoryInfo(mem_info);

    if (ret != XQC_OK) {
        memset(output, 0, sizeof(xqc_ml_output_t));
    }
    return ret;
}

#else /* !XQC_ENABLE_ONNX */

/* Stub implementations when ONNX is not available */

struct xqc_ml_model_s {
    int dummy;
};

xqc_int_t
xqc_ml_model_global_init(void)
{
    return XQC_OK;
}

void
xqc_ml_model_global_cleanup(void)
{
}

xqc_ml_model_t *
xqc_ml_model_load(const xqc_ml_model_config_t *config, xqc_log_t *log)
{
    (void)config;
    (void)log;
    return NULL;
}

void
xqc_ml_model_unload(xqc_ml_model_t *model)
{
    (void)model;
}

xqc_int_t
xqc_ml_model_infer(xqc_ml_model_t *model,
    const float window[XQC_ML_WINDOW_SIZE][XQC_ML_NUM_FEATURES],
    xqc_ml_output_t *output, xqc_log_t *log)
{
    (void)model;
    (void)window;
    (void)log;
    if (output) {
        memset(output, 0, sizeof(xqc_ml_output_t));
    }
    return -XQC_ERROR;
}

#endif /* XQC_ENABLE_ONNX */

const char *
xqc_ml_state_to_str(xqc_ml_state_t state)
{
    switch (state) {
    case XQC_ML_STATE_UT:
        return "UT";
    case XQC_ML_STATE_QU:
        return "QU";
    case XQC_ML_STATE_EB:
        return "EB";
    case XQC_ML_STATE_NH:
        return "NH";
    case XQC_ML_STATE_NONE:
    default:
        return "NONE";
    }
}

void
xqc_ml_apply_scaler(float features[XQC_ML_NUM_FEATURES],
    const float mean[XQC_ML_NUM_FEATURES],
    const float scale[XQC_ML_NUM_FEATURES])
{
    int i;
    for (i = 0; i < XQC_ML_NUM_FEATURES; i++) {
        if (scale[i] != 0.0f) {
            features[i] = (features[i] - mean[i]) / scale[i];
        }
    }
}

void
xqc_ml_softmax(float *input, float *output, int length)
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

    if (sum <= 1e-6f) {
        for (i = 0; i < length; i++) {
            output[i] = 1.0f / length;
        }
        return;
    }

    for (i = 0; i < length; i++) {
        output[i] = expf(input[i] - max_val) / sum;
    }
}
