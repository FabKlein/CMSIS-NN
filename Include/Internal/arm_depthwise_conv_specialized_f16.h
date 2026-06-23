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
 * Title:        arm_depthwise_conv_specialized_f16.h
 * Description:  Float16 depthwise-convolution specialization helpers
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.1
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_DEPTHWISE_CONV_SPECIALIZED_F16_H
#define ARM_DEPTHWISE_CONV_SPECIALIZED_F16_H

/* Internal specialization helpers (included from arm_depthwise_conv_f16.c). */

#include "Internal/arm_depthwise_conv_select_f16.h"
#include "Internal/arm_depthwise_conv_specialized_common.h"
#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnsupportfunctions.h"

#ifndef NN_DISABLE_SPECIALIZATION
typedef arm_cmsis_nn_status (*arm_dw_call_f16)(const cmsis_nn_dw_conv_params_f16 *params,
                                               const cmsis_nn_dims *input_dims,
                                               const float16_t *input,
                                               const float16_t *kernel,
                                               const float16_t *bias,
                                               const cmsis_nn_dims *output_dims,
                                               float16_t *output);

/* Call adapters are used only after selection has checked their requirements. */
static arm_cmsis_nn_status arm_dw_spec_k9_1d_nhwc_f16_call(const cmsis_nn_dw_conv_params_f16 *params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const float16_t *input,
                                                           const float16_t *kernel,
                                                           const float16_t *bias,
                                                           const cmsis_nn_dims *output_dims,
                                                           float16_t *output)
{
    const int32_t batch = input_dims->n;
    const int32_t input_h = input_dims->h;
    const int32_t input_c = input_dims->c;
    const int32_t input_w = input_dims->w;
    const int32_t output_h = output_dims->h;
    const int32_t output_w = output_dims->w;
    const int32_t output_c = output_dims->c;
    const size_t input_batch_stride = (size_t)input_h * (size_t)input_w * (size_t)input_c;
    const size_t output_batch_stride = (size_t)output_h * (size_t)output_w * (size_t)output_c;

    /* A 1x9 depthwise convolution over NHWC data is independent for each H row. */
    for (int32_t b = 0; b < batch; ++b)
    {
        const float16_t *input_b = input + (size_t)b * input_batch_stride;
        float16_t *output_b = output + (size_t)b * output_batch_stride;

        for (int32_t y = 0; y < output_h; ++y)
        {
            const float16_t *input_row = input_b + (size_t)y * input_w * input_c;
            float16_t *output_row = output_b + (size_t)y * output_w * output_c;
            arm_nn_depthwise_conv1d_k9_nhwc_f16(
                input_row, input_c, input_w, params->ch_mult, kernel, bias, output_row, output_w);
        }
    }

    const int32_t out_count = output_dims->n * output_dims->c * output_dims->h * output_dims->w;
    arm_nn_vector_clamp_f16(output, out_count, params->activation.min, params->activation.max);

    return ARM_CMSIS_NN_SUCCESS;
}

static arm_cmsis_nn_status arm_dw_spec_k3_1d_nhwc_f16_call(const cmsis_nn_dw_conv_params_f16 *params,
                                                           const cmsis_nn_dims *input_dims,
                                                           const float16_t *input,
                                                           const float16_t *kernel,
                                                           const float16_t *bias,
                                                           const cmsis_nn_dims *output_dims,
                                                           float16_t *output)
{
    const int32_t batch = input_dims->n;
    const int32_t input_h = input_dims->h;
    const int32_t input_c = input_dims->c;
    const int32_t input_w = input_dims->w;
    const int32_t output_h = output_dims->h;
    const int32_t output_w = output_dims->w;
    const int32_t output_c = output_dims->c;
    const size_t input_batch_stride = (size_t)input_h * (size_t)input_w * (size_t)input_c;
    const size_t output_batch_stride = (size_t)output_h * (size_t)output_w * (size_t)output_c;

    /* A 1x3 depthwise convolution over NHWC data is independent for each H row. */
    for (int32_t b = 0; b < batch; ++b)
    {
        const float16_t *input_b = input + (size_t)b * input_batch_stride;
        float16_t *output_b = output + (size_t)b * output_batch_stride;

        for (int32_t y = 0; y < output_h; ++y)
        {
            const float16_t *input_row = input_b + (size_t)y * input_w * input_c;
            float16_t *output_row = output_b + (size_t)y * output_w * output_c;
            arm_nn_depthwise_conv1d_k3_nhwc_f16(input_row, input_c, input_w, kernel, bias, output_row, output_w);
        }
    }

    const int32_t out_count = output_dims->n * output_dims->c * output_dims->h * output_dims->w;
    arm_nn_vector_clamp_f16(output, out_count, params->activation.min, params->activation.max);

    return ARM_CMSIS_NN_SUCCESS;
}

static arm_cmsis_nn_status arm_dw_spec_2x5_nhwc_f16_call(const cmsis_nn_dw_conv_params_f16 *params,
                                                         const cmsis_nn_dims *input_dims,
                                                         const float16_t *input,
                                                         const float16_t *kernel,
                                                         const float16_t *bias,
                                                         const cmsis_nn_dims *output_dims,
                                                         float16_t *output)
{
    arm_nn_depthwise_conv2x5_nhwc_f16(input,
                                      input_dims->n,
                                      input_dims->c,
                                      input_dims->w,
                                      params->ch_mult,
                                      kernel,
                                      bias,
                                      output,
                                      output_dims->w,
                                      params->activation.min,
                                      params->activation.max);
    return ARM_CMSIS_NN_SUCCESS;
}

static arm_cmsis_nn_status arm_dw_spec_3x3_nhwc_f16_call(const cmsis_nn_dw_conv_params_f16 *params,
                                                         const cmsis_nn_dims *input_dims,
                                                         const float16_t *input,
                                                         const float16_t *kernel,
                                                         const float16_t *bias,
                                                         const cmsis_nn_dims *output_dims,
                                                         float16_t *output)
{
    arm_nn_depthwise_conv3x3_nhwc_f16(input,
                                      input_dims->n,
                                      input_dims->c,
                                      input_dims->h,
                                      input_dims->w,
                                      kernel,
                                      bias,
                                      output,
                                      params->stride.w,
                                      params->stride.h,
                                      params->padding.w,
                                      params->padding.h,
                                      output_dims->h,
                                      output_dims->w,
                                      params->activation.min,
                                      params->activation.max);
    return ARM_CMSIS_NN_SUCCESS;
}

/* Function pointers follow the same registry order as the selection metadata. */
static const arm_dw_call_f16 arm_dw_calls_nhwc_f16[] = {
    #define ARM_DW_SPEC(H, W, CALL, ...) CALL,
    #include "Internal/arm_depthwise_conv_specialized_registry_f16.h"
    #undef ARM_DW_SPEC
};

/* Public validation must precede selection. All registered kernels use KC weights. */
static inline arm_dw_call_f16 arm_dw_select_nhwc_f16(const cmsis_nn_dw_conv_params_f16 *params,
                                                     const cmsis_nn_dims *input_dims,
                                                     const cmsis_nn_dims *filter_dims,
                                                     const cmsis_nn_dims *output_dims,
                                                     arm_nn_dw_kernel_layout_f16 kernel_layout)
{
    size_t index;
    if (arm_depthwise_conv_select_specialized_f16(params, input_dims, filter_dims, output_dims, kernel_layout, &index))
    {
        return arm_dw_calls_nhwc_f16[index];
    }
    return NULL;
}
#endif

#endif /* ARM_DEPTHWISE_CONV_SPECIALIZED_F16_H */
