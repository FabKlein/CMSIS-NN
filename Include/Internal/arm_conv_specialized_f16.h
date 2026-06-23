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
 * Title:        arm_conv_specialized_f16.h
 * Description:  Float16 convolution specialization helpers
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_CONV_SPECIALIZED_F16_H
#define ARM_CONV_SPECIALIZED_F16_H

/* Internal specialization helpers (included from arm_convolve_f16.c). */

#include "Internal/arm_conv_select_f16.h"
#include "Internal/arm_nn_activation_flt.h"
#include "arm_nnsupportfunctions.h"

#ifndef NN_DISABLE_SPECIALIZATION

/* Execute one batch of a height-1 filter. The row kernel handles horizontal geometry. */
static inline void arm_conv_execute_1d_f16(arm_conv_1d_kernel_f16 kernel,
                                           const cmsis_nn_conv_params_f16 *params,
                                           const cmsis_nn_dims *input_dims,
                                           const float16_t *input,
                                           const float16_t *weights,
                                           const float16_t *bias,
                                           const cmsis_nn_dims *output_dims,
                                           float16_t *output)
{
    for (int32_t y = 0; y < output_dims->h; ++y)
    {
        const int32_t input_y = y * params->stride.h - params->padding.h;
        float16_t *output_row = output + (size_t)y * output_dims->w * output_dims->c;
        if (input_y >= 0 && input_y < input_dims->h)
        {
            const float16_t *input_row = input + (size_t)input_y * input_dims->w * input_dims->c;
            kernel(input_row, input_dims->c, input_dims->w, weights, bias, output_row, output_dims->c, output_dims->w);
        }
        else
        {
            /* A row wholly in vertical padding contributes only bias; clamping follows execution. */
            for (int32_t x = 0; x < output_dims->w; ++x)
            {
                for (int32_t c = 0; c < output_dims->c; ++c)
                {
                    output_row[x * output_dims->c + c] = bias ? bias[c] : 0;
                }
            }
        }
    }
}

/* Execute a successful selection without searching the registry again. */
static inline void arm_conv_execute_specialized_f16(const arm_conv_selection_f16 *selection,
                                                    const cmsis_nn_conv_params_f16 *params,
                                                    const cmsis_nn_dims *input_dims,
                                                    const float16_t *input_data,
                                                    const float16_t *filter_data,
                                                    const float16_t *bias_data,
                                                    const cmsis_nn_dims *output_dims,
                                                    float16_t *output_data)
{
    const arm_conv_kernels_f16 kernels = arm_conv_resolve_specialized_f16(selection);
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
        if (kernels.kernel_1d)
        {
            arm_conv_execute_1d_f16(
                kernels.kernel_1d, params, input_dims, input_b, filter_data, bias_data, output_dims, output_b);
        }
        else
        {
            kernels.kernel_2d(
                input_b, input_c, input_h, input_w, filter_data, bias_data, output_b, output_c, output_h, output_w);
        }
    }

    const int32_t out_count = batch * output_h * output_w * output_c;
    arm_nn_vector_clamp_f16(output_data, out_count, params->activation.min, params->activation.max);
}
#endif

#endif /* ARM_CONV_SPECIALIZED_F16_H */
