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
 * Title:        arm_conv_specialized_common.h
 * Description:  Shared float convolution specialization helpers
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_CONV_SPECIALIZED_COMMON_H
#define ARM_CONV_SPECIALIZED_COMMON_H

#include "Internal/arm_nn_tensor_utils.h"
#include "arm_nn_types_flt.h"

#define ARM_CONV_ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

/* Exact requirements for one specialized kernel; weight format is checked separately. */
typedef struct
{
    int32_t filter_h;
    int32_t filter_w;
    cmsis_nn_tile stride;
    cmsis_nn_tile padding;
    cmsis_nn_tile dilation;
} arm_conv_specialized_requirements;

/*
 * Buffer-size queries use these requirements and availability flags to select a kernel.
 * Function pointers are stored separately so calling a buffer-size query alone does
 * not cause the linker to include the convolution implementations in the executable.
 */
typedef struct
{
    arm_conv_specialized_requirements requirements;
    bool packed;
    bool standard;
} arm_conv_specialized_candidate;

/*
 * Check one output extent with widened arithmetic before using int32_t indexing.
 * Bounding the padded input and effective filter also bounds each output origin
 * and sampled coordinate when the output extent matches the convolution formula.
 */
static inline bool arm_conv_specialized_axis_is_valid(int32_t input,
                                                      int32_t filter,
                                                      int32_t output,
                                                      int32_t stride,
                                                      int32_t padding,
                                                      int32_t dilation)
{
    if (input <= 0 || filter <= 0 || output <= 0 || stride <= 0 || padding < 0 || dilation <= 0)
    {
        return false;
    }
    const int64_t padded_input = (int64_t)input + 2 * (int64_t)padding;
    const int64_t effective_filter = (int64_t)(filter - 1) * dilation + 1;
    if (padded_input > INT32_MAX || effective_filter > padded_input)
    {
        return false;
    }

    /* The bounded nonnegative numerator needs only a 32-bit division. */
    return output == (int32_t)(padded_input - effective_filter) / stride + 1;
}

/*
 * Check tensor consistency and output geometry for specialized NHWC convolution.
 * Stride, padding, and dilation are validated here without prescribing a kernel's
 * supported values. Exact requirements belong to that kernel's registry entry.
 *
 * A true result alone does not imply an available implementation. A false result
 * prevents specialized selection; other convolution paths may still accept the case.
 */
static inline bool arm_conv_specialized_nhwc_geometry_is_valid(const cmsis_nn_tile *stride,
                                                               const cmsis_nn_tile *padding,
                                                               const cmsis_nn_tile *dilation,
                                                               const cmsis_nn_dims *input_dims,
                                                               const cmsis_nn_dims *filter_dims,
                                                               const cmsis_nn_dims *output_dims)
{
    if (!stride || !padding || !dilation || !input_dims || !filter_dims || !output_dims)
    {
        return false;
    }

    /* Bound tensor products used for indexing, including batch offsets. */
    if (!arm_nn_tensor_size_is_valid(input_dims->n, input_dims->h, input_dims->w, input_dims->c) ||
        !arm_nn_tensor_size_is_valid(filter_dims->n, filter_dims->h, filter_dims->w, filter_dims->c) ||
        !arm_nn_tensor_size_is_valid(output_dims->n, output_dims->h, output_dims->w, output_dims->c))
    {
        return false;
    }

    /* Filter dimensions describe [Cout, Hk, Wk, Cin], including for packed weights. */
    if (input_dims->n != output_dims->n || filter_dims->c != input_dims->c || output_dims->c != filter_dims->n)
    {
        return false;
    }

    return arm_conv_specialized_axis_is_valid(
               input_dims->h, filter_dims->h, output_dims->h, stride->h, padding->h, dilation->h) &&
        arm_conv_specialized_axis_is_valid(
               input_dims->w, filter_dims->w, output_dims->w, stride->w, padding->w, dilation->w);
}

/* Compare against declared capabilities only; general geometry is checked above. */
static inline bool arm_conv_specialized_requirements_match(const arm_conv_specialized_requirements *spec,
                                                           const cmsis_nn_dims *filter_dims,
                                                           const cmsis_nn_tile *stride,
                                                           const cmsis_nn_tile *padding,
                                                           const cmsis_nn_tile *dilation)
{
    return filter_dims->h == spec->filter_h && filter_dims->w == spec->filter_w && stride->h == spec->stride.h &&
        stride->w == spec->stride.w && padding->h == spec->padding.h && padding->w == spec->padding.w &&
        dilation->h == spec->dilation.h && dilation->w == spec->dilation.w;
}

/*
 * Heuristics for selecting the NHWC packed-patch-matrix + GEMM float32 path.
 * Below these sizes the packing/setup overhead tends to outweigh the GEMM win.
 * MAX_TILE_ROWS bounds scratch usage and keeps the packed panel cache-friendly.
 */
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F32_MAX_TILE_ROWS (8)
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F32_MIN_K (16)
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F32_MIN_OC (8)
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F32_MIN_POS (8)

/*
 * Heuristics for selecting the NHWC packed-patch-matrix + GEMM float16 path.
 * These mirror the float32 thresholds and keep the packed tile large enough to amortize setup cost.
 */
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MAX_TILE_ROWS (8)
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MIN_K (16)
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MIN_OC (8)
#define ARM_NN_CONV_NHWC_PATCH_GEMM_F16_MIN_POS (8)

#endif /* ARM_CONV_SPECIALIZED_COMMON_H */
