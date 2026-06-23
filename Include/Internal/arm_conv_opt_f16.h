/*
 * SPDX-FileCopyrightText: Copyright 2010-2026 Arm Limited and/or its affiliates
 * <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_conv_opt_f16.h
 * Description:  Float16 convolution specialization helpers
 *
 * $Date:        27 March 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_CONV_OPT_F16_H
#define ARM_CONV_OPT_F16_H

/* Internal specialization helpers (included from arm_convolve_f16.c). */

#include "Internal/arm_conv_opt_common.h"
#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnsupportfunctions.h"

#ifndef NN_DISABLE_SPECIALIZATION
typedef bool (*arm_conv_match_f16)(const cmsis_nn_context *ctx,
                                   const cmsis_nn_conv_params_f16 *params,
                                   const cmsis_nn_dims *input_dims,
                                   const float16_t *input_data,
                                   const cmsis_nn_dims *filter_dims,
                                   const float16_t *filter_data,
                                   const cmsis_nn_dims *bias_dims,
                                   const float16_t *bias_data,
                                   const cmsis_nn_dims *output_dims,
                                   float16_t *output_data);

typedef arm_cmsis_nn_status (*arm_conv_call_f16)(const cmsis_nn_context *ctx,
                                                 const cmsis_nn_conv_params_f16 *params,
                                                 const cmsis_nn_dims *input_dims,
                                                 const float16_t *input_data,
                                                 const cmsis_nn_dims *filter_dims,
                                                 const float16_t *filter_data,
                                                 const cmsis_nn_dims *bias_dims,
                                                 const float16_t *bias_data,
                                                 const cmsis_nn_dims *output_dims,
                                                 float16_t *output_data);

typedef struct
{
    arm_conv_match_f16 match;
    arm_conv_call_f16 call;
} arm_conv_spec_f16;

/* -----------------------------------------------------------------------
 * Table-driven convolution specialization.
 *
 * Two kernel signatures exist, determined by the dispatch pattern:
 *
 *   1D (row-by-row): filter_h == 1, each H row is an independent 1D conv.
 *       (row_ptr, in_c, in_w, kernel, bias, out_row, out_c, out_w)
 *
 *   2D (per-batch tile): kernel handles full H×W spatial tile internally.
 *       (slice_ptr, in_c, in_h, in_w, kernel, bias, out, out_c, out_h, out_w)
 *
 * Adding a new shape = one table line + the underlying kernel source file.
 * ----------------------------------------------------------------------- */

typedef void (*arm_conv_1d_kernel_f16)(const float16_t *,
                                       int32_t,
                                       int32_t,
                                       const float16_t *,
                                       const float16_t *,
                                       float16_t *,
                                       int32_t,
                                       int32_t);

typedef void (*arm_conv_2d_kernel_f16)(const float16_t *,
                                       int32_t,
                                       int32_t,
                                       int32_t,
                                       const float16_t *,
                                       const float16_t *,
                                       float16_t *,
                                       int32_t,
                                       int32_t,
                                       int32_t);

typedef struct
{
    int32_t filter_w;
    arm_conv_1d_kernel_f16 packed; /* packed-weight kernel */
    arm_conv_1d_kernel_f16 nhwc;   /* NHWC-weight kernel, NULL → packed-only */
} arm_conv_1d_spec_f16;

typedef struct
{
    int32_t filter_h;
    int32_t filter_w;
    arm_conv_2d_kernel_f16 packed; /* packed-weight kernel (always packed-only) */
} arm_conv_2d_spec_f16;

static const arm_conv_1d_spec_f16 arm_conv_1d_specs_f16[] = {
    {9, arm_nn_conv1d_k9_packed_f16, NULL},
    {7, arm_nn_conv1d_k7_packed_f16, NULL},
    {5, arm_nn_conv1d_k5_packed_f16, arm_nn_conv1d_k5_nhwc_f16},
    {3, arm_nn_conv1d_k3_packed_f16, arm_nn_conv1d_k3_nhwc_f16},
};

static const arm_conv_2d_spec_f16 arm_conv_2d_specs_f16[] = {
    {1, 2, arm_nn_conv1d_k2_packed_f16},
    {2, 2, arm_nn_conv2d_2x2_packed_f16},
    {2, 3, arm_nn_conv2d_2x3_packed_f16},
    {2, 5, arm_nn_conv2d_2x5_packed_f16},
};

/* --- 1D match: filter_h == 1, dispatch per spatial row --- */

static bool arm_conv_1d_spec_nhwc_f16_match(const cmsis_nn_context *ctx,
                                            const cmsis_nn_conv_params_f16 *params,
                                            const cmsis_nn_dims *input_dims,
                                            const float16_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const float16_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const float16_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            float16_t *output_data)
{
    (void)ctx;
    (void)input_data;
    (void)filter_data;
    (void)bias_dims;
    (void)bias_data;
    (void)output_data;

    if (filter_dims->h != 1 || params->stride.h != 1 || params->stride.w != 1 || params->padding.h != 0 ||
        params->padding.w != 0 || params->dilation.h != 1 || params->dilation.w != 1)
    {
        return false;
    }

    const int32_t batch = input_dims->n;
    if (batch < 1 || input_dims->h < 1 || output_dims->h != input_dims->h ||
        output_dims->w != input_dims->w - filter_dims->w + 1 || input_dims->w < filter_dims->w)
    {
        return false;
    }

    for (size_t i = 0; i < sizeof(arm_conv_1d_specs_f16) / sizeof(arm_conv_1d_specs_f16[0]); ++i)
    {
        if (filter_dims->w == arm_conv_1d_specs_f16[i].filter_w)
        {
            if (params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED && arm_conv_1d_specs_f16[i].packed)
            {
                return true;
            }
            if (params->weight_format != ARM_NN_WEIGHT_FORMAT_NT_N_PACKED && arm_conv_1d_specs_f16[i].nhwc)
            {
                return true;
            }
            return false;
        }
    }

    return false;
}

/* --- 1D call --- */

static arm_cmsis_nn_status arm_conv_1d_spec_nhwc_f16_call(const cmsis_nn_context *ctx,
                                                          const cmsis_nn_conv_params_f16 *params,
                                                          const cmsis_nn_dims *input_dims,
                                                          const float16_t *input_data,
                                                          const cmsis_nn_dims *filter_dims,
                                                          const float16_t *filter_data,
                                                          const cmsis_nn_dims *bias_dims,
                                                          const float16_t *bias_data,
                                                          const cmsis_nn_dims *output_dims,
                                                          float16_t *output_data)
{
    (void)ctx;
    (void)bias_dims;

    arm_conv_1d_kernel_f16 kernel_fn = NULL;

    for (size_t i = 0; i < sizeof(arm_conv_1d_specs_f16) / sizeof(arm_conv_1d_specs_f16[0]); ++i)
    {
        if (filter_dims->w == arm_conv_1d_specs_f16[i].filter_w)
        {
            kernel_fn = (params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED) ? arm_conv_1d_specs_f16[i].packed
                                                                                    : arm_conv_1d_specs_f16[i].nhwc;
            break;
        }
    }

    if (kernel_fn == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t batch = input_dims->n;
    const int32_t input_h = input_dims->h;
    const int32_t input_c = input_dims->c;
    const int32_t input_w = input_dims->w;
    const int32_t output_h = output_dims->h;
    const int32_t output_c = output_dims->c;
    const int32_t output_w = output_dims->w;

    for (int32_t b = 0; b < batch; ++b)
    {
        const float16_t *input_b = input_data + (size_t)b * input_h * input_w * input_c;
        float16_t *output_b = output_data + (size_t)b * output_h * output_w * output_c;

        for (int32_t y = 0; y < output_h; ++y)
        {
            const float16_t *input_row = input_b + (size_t)y * input_w * input_c;
            float16_t *output_row = output_b + (size_t)y * output_w * output_c;
            kernel_fn(input_row, input_c, input_w, filter_data, bias_data, output_row, output_c, output_w);
        }
    }

    const int32_t out_count = batch * output_h * output_w * output_c;
    arm_nn_vector_clamp_f16(output_data, out_count, params->activation.min, params->activation.max);

    return ARM_CMSIS_NN_SUCCESS;
}

/* --- 2D match: kernel handles full spatial tile per batch --- */

static bool arm_conv_2d_spec_nhwc_f16_match(const cmsis_nn_context *ctx,
                                            const cmsis_nn_conv_params_f16 *params,
                                            const cmsis_nn_dims *input_dims,
                                            const float16_t *input_data,
                                            const cmsis_nn_dims *filter_dims,
                                            const float16_t *filter_data,
                                            const cmsis_nn_dims *bias_dims,
                                            const float16_t *bias_data,
                                            const cmsis_nn_dims *output_dims,
                                            float16_t *output_data)
{
    (void)ctx;
    (void)input_data;
    (void)filter_data;
    (void)bias_dims;
    (void)bias_data;
    (void)output_data;

    if (params->weight_format != ARM_NN_WEIGHT_FORMAT_NT_N_PACKED || params->stride.h != 1 || params->stride.w != 1 ||
        params->padding.h != 0 || params->padding.w != 0 || params->dilation.h != 1 || params->dilation.w != 1)
    {
        return false;
    }

    const int32_t batch = input_dims->n;
    if (batch < 1 || batch != output_dims->n || filter_dims->c != input_dims->c || output_dims->c != filter_dims->n ||
        input_dims->h < filter_dims->h || input_dims->w < filter_dims->w ||
        output_dims->h != input_dims->h - filter_dims->h + 1 || output_dims->w != input_dims->w - filter_dims->w + 1)
    {
        return false;
    }

    for (size_t i = 0; i < sizeof(arm_conv_2d_specs_f16) / sizeof(arm_conv_2d_specs_f16[0]); ++i)
    {
        if (filter_dims->h == arm_conv_2d_specs_f16[i].filter_h && filter_dims->w == arm_conv_2d_specs_f16[i].filter_w)
        {
            return true;
        }
    }

    return false;
}

/* --- 2D call --- */

static arm_cmsis_nn_status arm_conv_2d_spec_nhwc_f16_call(const cmsis_nn_context *ctx,
                                                          const cmsis_nn_conv_params_f16 *params,
                                                          const cmsis_nn_dims *input_dims,
                                                          const float16_t *input_data,
                                                          const cmsis_nn_dims *filter_dims,
                                                          const float16_t *filter_data,
                                                          const cmsis_nn_dims *bias_dims,
                                                          const float16_t *bias_data,
                                                          const cmsis_nn_dims *output_dims,
                                                          float16_t *output_data)
{
    (void)ctx;
    (void)bias_dims;

    arm_conv_2d_kernel_f16 kernel_fn = NULL;

    for (size_t i = 0; i < sizeof(arm_conv_2d_specs_f16) / sizeof(arm_conv_2d_specs_f16[0]); ++i)
    {
        if (filter_dims->h == arm_conv_2d_specs_f16[i].filter_h && filter_dims->w == arm_conv_2d_specs_f16[i].filter_w)
        {
            kernel_fn = arm_conv_2d_specs_f16[i].packed;
            break;
        }
    }

    if (kernel_fn == NULL)
    {
        return ARM_CMSIS_NN_ARG_ERROR;
    }

    const int32_t batch = input_dims->n;
    const int32_t input_h = input_dims->h;
    const int32_t input_c = input_dims->c;
    const int32_t input_w = input_dims->w;
    const int32_t output_h = output_dims->h;
    const int32_t output_c = output_dims->c;
    const int32_t output_w = output_dims->w;

    for (int32_t b = 0; b < batch; ++b)
    {
        const float16_t *input_b = input_data + (size_t)b * input_h * input_w * input_c;
        float16_t *output_b = output_data + (size_t)b * output_h * output_w * output_c;
        kernel_fn(input_b, input_c, input_h, input_w, filter_data, bias_data, output_b, output_c, output_h, output_w);
    }

    const int32_t out_count = batch * output_h * output_w * output_c;
    arm_nn_vector_clamp_f16(output_data, out_count, params->activation.min, params->activation.max);

    return ARM_CMSIS_NN_SUCCESS;
}

/* --- Dispatch table wiring --- */

static const arm_conv_spec_f16 arm_conv_spec_nhwc_f16[] = {
    ARM_CONV_SPEC_ENTRY(arm_conv_1d_spec_nhwc_f16_match, arm_conv_1d_spec_nhwc_f16_call),
    ARM_CONV_SPEC_ENTRY(arm_conv_2d_spec_nhwc_f16_match, arm_conv_2d_spec_nhwc_f16_call),
};

__STATIC_INLINE bool arm_conv_spec_nhwc_f16_matches_any(const cmsis_nn_context *ctx,
                                                        const cmsis_nn_conv_params_f16 *params,
                                                        const cmsis_nn_dims *input_dims,
                                                        const cmsis_nn_dims *filter_dims,
                                                        const cmsis_nn_dims *output_dims)
{
    for (size_t i = 0; i < ARM_CONV_ARRAY_SIZE(arm_conv_spec_nhwc_f16); ++i)
    {
        if (arm_conv_spec_nhwc_f16[i].match(
                ctx, params, input_dims, NULL, filter_dims, NULL, NULL, NULL, output_dims, NULL))
        {
            return true;
        }
    }

    return false;
}
#endif

#endif /* ARM_CONV_OPT_F16_H */
