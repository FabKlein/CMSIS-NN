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
 * Title:        arm_depthwise_conv_specialized_common.h
 * Description:  Shared float depthwise-convolution specialization helpers
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.1
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_DEPTHWISE_CONV_SPECIALIZED_COMMON_H
#define ARM_DEPTHWISE_CONV_SPECIALIZED_COMMON_H

#include "Internal/arm_nn_compiler.h"
#include "Internal/arm_nn_tensor_utils.h"
#include "arm_nn_types.h"

#include <stdbool.h>
#include <stdint.h>

/*
 * These capabilities describe the support kernels, not the public API's accepted shapes.
 * ANY accepts any value that has passed public parameter validation.
 */
typedef enum
{
    ARM_DW_MULTIPLIER_ONE, /* Exactly one output channel per input channel. */
    ARM_DW_MULTIPLIER_ANY  /* Any positive multiplier accepted by public validation. */
} arm_dw_multiplier_support;

typedef enum
{
    ARM_DW_STRIDE_EXACT,
    ARM_DW_STRIDE_ANY
} arm_dw_stride_policy;

typedef enum
{
    ARM_DW_PADDING_EXACT,
    ARM_DW_PADDING_ANY
} arm_dw_padding_policy;

typedef struct
{
    int32_t filter_h;
    int32_t filter_w;
    arm_dw_multiplier_support multiplier;
    arm_dw_stride_policy stride_policy;
    cmsis_nn_tile stride; /* Required values only when stride_policy is ARM_DW_STRIDE_EXACT. */
    arm_dw_padding_policy padding_policy;
    cmsis_nn_tile padding; /* Required values only when padding_policy is ARM_DW_PADDING_EXACT. */
    cmsis_nn_tile dilation;
    bool require_input_height_equals_filter;
} arm_dw_requirements;

/*
 * Match a KC-layout specialization after public tensor/channel/index validation.
 * Validation guarantees positive strides/dilation, nonnegative padding, matching
 * batches/channels, and padded dimensions and effective filters that fit int32_t.
 * Policies describe implementation support; the public validator checks validity.
 * Each field is matched independently, then output geometry uses the requested dilation.
 */
static inline bool arm_dw_spec_matches(const arm_dw_requirements *spec,
                                       const cmsis_nn_dims *input_dims,
                                       const cmsis_nn_dims *filter_dims,
                                       const cmsis_nn_dims *output_dims,
                                       cmsis_nn_tile stride,
                                       cmsis_nn_tile padding,
                                       cmsis_nn_tile dilation,
                                       int32_t ch_mult)
{
    if (filter_dims->h != spec->filter_h || filter_dims->w != spec->filter_w ||
        (spec->multiplier == ARM_DW_MULTIPLIER_ONE && ch_mult != 1) || dilation.h != spec->dilation.h ||
        dilation.w != spec->dilation.w)
    {
        return false;
    }

    const bool stride_matches = spec->stride_policy == ARM_DW_STRIDE_ANY ||
        (spec->stride_policy == ARM_DW_STRIDE_EXACT && stride.h == spec->stride.h && stride.w == spec->stride.w);
    const bool padding_matches = spec->padding_policy == ARM_DW_PADDING_ANY ||
        (spec->padding_policy == ARM_DW_PADDING_EXACT && padding.h == spec->padding.h && padding.w == spec->padding.w);
    if (!stride_matches || !padding_matches)
    {
        return false;
    }
    if (spec->require_input_height_equals_filter && input_dims->h != filter_dims->h)
    {
        return false;
    }

    /* Public validation bounds these intermediate expressions, including dilation products. */
    const int32_t padded_h = input_dims->h + 2 * padding.h;
    const int32_t padded_w = input_dims->w + 2 * padding.w;
    const int32_t effective_h = (filter_dims->h - 1) * dilation.h + 1;
    const int32_t effective_w = (filter_dims->w - 1) * dilation.w + 1;
    return padded_h >= effective_h && padded_w >= effective_w &&
        output_dims->h == (padded_h - effective_h) / stride.h + 1 &&
        output_dims->w == (padded_w - effective_w) / stride.w + 1;
}

/* Reject geometry that cannot be evaluated by the kernels' signed 32-bit indexing. */
static inline int arm_depthwise_conv_axis_is_valid(int32_t input,
                                                   int32_t kernel,
                                                   int32_t output,
                                                   int32_t stride,
                                                   int32_t padding,
                                                   int32_t dilation)
{
    if (input <= 0 || kernel <= 0 || output <= 0 || stride <= 0 || padding < 0 || dilation <= 0)
    {
        return 0;
    }

    /* Widen before arithmetic: validation must not itself overflow. */
    const int64_t padded_input = (int64_t)input + 2 * (int64_t)padding;
    const int64_t last_origin = (int64_t)(output - 1) * stride;
    const int64_t kernel_span = (int64_t)(kernel - 1) * dilation;

    /*
     * Dilated clipping evaluates input - base + dilation before subtracting 1.
     * Bound that intermediate addition at the left/top edge, where base == -padding.
     * Unit dilation uses no rounded division and does not need this extra bound.
     */
    const bool clipping_fits = dilation == 1 || (int64_t)input + padding + dilation <= INT32_MAX;

    return padded_input <= INT32_MAX && last_origin <= INT32_MAX && kernel_span < INT32_MAX &&
        last_origin - padding + kernel_span <= INT32_MAX && clipping_fits;
}

/*
 * Shared dimension/channel/index checks for execution and scratch-size selection.
 * Callers check the parameter and dimension pointers before calling this helper.
 * Output extents need not imply symmetric padding for the general depthwise API;
 * specialization matching separately checks each kernel's exact output geometry.
 */
static inline bool arm_depthwise_conv_geometry_is_valid(const cmsis_nn_dims *input_dims,
                                                        const cmsis_nn_dims *filter_dims,
                                                        const cmsis_nn_dims *output_dims,
                                                        cmsis_nn_tile stride,
                                                        cmsis_nn_tile padding,
                                                        cmsis_nn_tile dilation,
                                                        int32_t ch_mult)
{
    return ch_mult > 0 && input_dims->n == output_dims->n && (int64_t)input_dims->c * ch_mult == output_dims->c &&
        arm_nn_tensor_size_is_valid(input_dims->n, input_dims->h, input_dims->w, input_dims->c) &&
        arm_nn_tensor_size_is_valid(output_dims->n, output_dims->h, output_dims->w, output_dims->c) &&
        arm_nn_tensor_size_is_valid(1, filter_dims->h, filter_dims->w, output_dims->c) &&
        arm_depthwise_conv_axis_is_valid(
               input_dims->h, filter_dims->h, output_dims->h, stride.h, padding.h, dilation.h) &&
        arm_depthwise_conv_axis_is_valid(
               input_dims->w, filter_dims->w, output_dims->w, stride.w, padding.w, dilation.w);
}

/* NHWC input index within one batch. Coordinates are zero-based; width/channel_count are tensor dimensions. */
static inline int32_t
arm_depthwise_conv_input_index_nhwc(int32_t x, int32_t y, int32_t channel_index, int32_t width, int32_t channel_count)
{
    return (y * width + x) * channel_count + channel_index;
}

/* NHWC output index within one batch. Coordinates are zero-based; width/channel_count are tensor dimensions. */
static inline int32_t
arm_depthwise_conv_output_index_nhwc(int32_t x, int32_t y, int32_t channel_index, int32_t width, int32_t channel_count)
{
    return (y * width + x) * channel_count + channel_index;
}

#endif /* ARM_DEPTHWISE_CONV_SPECIALIZED_COMMON_H */
