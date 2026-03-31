/**
 * @copyright Copyright (c) 2024, Alibaba Group Holding Limited
 * @brief ML Model Interface for Congestion Control
 * 
 * Provides unified ONNX model inference for ML-based congestion control algorithms.
 * Supports state classification (UT/QU/EB/NH) and queue depth regression.
 */

#ifndef XQC_ML_MODEL_H
#define XQC_ML_MODEL_H

#include <xquic/xquic.h>
#include "src/common/xqc_config.h"

/* ML State Definitions */
typedef enum {
    XQC_ML_STATE_UT   = 0,   /* Underutilized */
    XQC_ML_STATE_QU   = 1,   /* Queue (congested) */
    XQC_ML_STATE_EB   = 2,   /* Exceeded Bandwidth */
    XQC_ML_STATE_NH   = 3,   /* Network Handover */
    XQC_ML_STATE_NONE = 4
} xqc_ml_state_t;

/* Number of state classes */
#define XQC_ML_NUM_STATES    4

/* Feature dimensions */
#define XQC_ML_WINDOW_SIZE   30
#define XQC_ML_NUM_FEATURES  15

/* Feature scaler constants (from training) */
extern const float xqc_ml_scaler_mean[XQC_ML_NUM_FEATURES];
extern const float xqc_ml_scaler_scale[XQC_ML_NUM_FEATURES];

/* ML Model Handle (opaque) */
typedef struct xqc_ml_model_s xqc_ml_model_t;

/**
 * @brief Model types supported
 */
typedef enum {
    XQC_ML_MODEL_STATE,      /* State classification: input -> [UT, QU, EB, NH] */
    XQC_ML_MODEL_QUEUE       /* Queue regression: input -> queue_depth */
} xqc_ml_model_type_t;

/**
 * @brief Inference output
 */
typedef struct {
    xqc_ml_state_t  predicted_state;    /* Argmax state */
    float           state_probs[XQC_ML_NUM_STATES];  /* Softmax probabilities */
    float           queue_depth;        /* Queue model output (if applicable) */
    int             valid;              /* 1 if inference successful */
} xqc_ml_output_t;

/**
 * @brief Model configuration
 */
typedef struct {
    const char     *model_path;         /* Path to .onnx file */
    xqc_ml_model_type_t type;           /* Model type */
    int             use_scaler;         /* Whether to apply feature normalization */
    const float    *scaler_mean;        /* Scaler mean (15 floats) or NULL */
    const float    *scaler_scale;       /* Scaler scale (15 floats) or NULL */
} xqc_ml_model_config_t;

/**
 * @brief Initialize ML model system (call once at startup)
 * @return XQC_OK on success, negative error code on failure
 */
xqc_int_t xqc_ml_model_global_init(void);

/**
 * @brief Cleanup ML model system (call at shutdown)
 */
void xqc_ml_model_global_cleanup(void);

/**
 * @brief Load an ONNX model
 * @param config Model configuration
 * @param log Logger for error reporting (can be NULL)
 * @return Model handle or NULL on error
 */
xqc_ml_model_t *xqc_ml_model_load(const xqc_ml_model_config_t *config, 
    xqc_log_t *log);

/**
 * @brief Unload a model and free resources
 * @param model Model handle
 */
void xqc_ml_model_unload(xqc_ml_model_t *model);

/**
 * @brief Run inference on a window of features
 * @param model Model handle
 * @param window Feature window [WINDOW_SIZE][NUM_FEATURES], chronological order
 * @param output Output structure to fill
 * @param log Logger for debugging (can be NULL)
 * @return XQC_OK on success, negative error code on failure
 * 
 * Note: window should contain WINDOW_SIZE samples in chronological order.
 * The function applies softmax to state classification outputs automatically.
 */
xqc_int_t xqc_ml_model_infer(xqc_ml_model_t *model, 
    const float window[XQC_ML_WINDOW_SIZE][XQC_ML_NUM_FEATURES],
    xqc_ml_output_t *output, xqc_log_t *log);

/**
 * @brief Get string representation of state
 * @param state State enum
 * @return String name (e.g., "UT", "QU")
 */
const char *xqc_ml_state_to_str(xqc_ml_state_t state);

/**
 * @brief Apply standard scaler normalization to features
 * @param features Input/output feature array (15 floats)
 * @param mean Scaler mean array (15 floats)
 * @param scale Scaler scale array (15 floats)
 */
void xqc_ml_apply_scaler(float features[XQC_ML_NUM_FEATURES],
    const float mean[XQC_ML_NUM_FEATURES],
    const float scale[XQC_ML_NUM_FEATURES]);

/**
 * @brief Softmax activation function
 * @param input Input logits array
 * @param output Output probabilities array
 * @param length Array length
 */
void xqc_ml_softmax(float *input, float *output, int length);

/**
 * @brief Clamp float value to range
 * @param value Input value
 * @param min_value Minimum
 * @param max_value Maximum
 * @return Clamped value
 */
static inline float
xqc_ml_clamp_float(float value, float min_value, float max_value)
{
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

#endif /* XQC_ML_MODEL_H */
