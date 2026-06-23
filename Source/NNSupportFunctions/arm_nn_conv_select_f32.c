/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
 * Title:        arm_nn_conv_select_f32.c
 * Description:  Table-based F32 specialized convolution selection
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32 && !defined(NN_DISABLE_SPECIALIZATION)

    #include "Internal/arm_conv_specialized_common.h"
    #include "Internal/arm_conv_select_f32.h"

/* Function presence is folded to bool constants here; these tables contain
 * no function addresses. The execution tables use the same registry order.
 */
static const arm_conv_specialized_candidate arm_conv_1d_available_f32[] = {
    #define ARM_CONV_1D_SPEC(W, PACKED, STANDARD, ...)                                                                 \
        {{.filter_h = 1, .filter_w = W, __VA_ARGS__}, (PACKED) != NULL, (STANDARD) != NULL},
    #define ARM_CONV_2D_SPEC(H, W, PACKED, ...)
    #include "Internal/arm_conv_specialized_registry_f32.h"
    #undef ARM_CONV_2D_SPEC
    #undef ARM_CONV_1D_SPEC
};

static const arm_conv_specialized_candidate arm_conv_2d_available_f32[] = {
    #define ARM_CONV_1D_SPEC(W, PACKED, STANDARD, ...)
    #define ARM_CONV_2D_SPEC(H, W, PACKED, ...) {{.filter_h = H, .filter_w = W, __VA_ARGS__}, (PACKED) != NULL, false},
    #include "Internal/arm_conv_specialized_registry_f32.h"
    #undef ARM_CONV_2D_SPEC
    #undef ARM_CONV_1D_SPEC
};

bool arm_conv_select_specialized_f32(const cmsis_nn_conv_params_f32 *params,
                                     const cmsis_nn_dims *input_dims,
                                     const cmsis_nn_dims *filter_dims,
                                     const cmsis_nn_dims *output_dims,
                                     arm_conv_selection_f32 *selection)
{
    if (selection)
    {
        selection->index = 0;
        selection->is_1d = false;
        selection->packed = false;
    }
    if (!params ||
        (params->weight_format != ARM_NN_WEIGHT_FORMAT_STANDARD &&
         params->weight_format != ARM_NN_WEIGHT_FORMAT_NT_N_PACKED) ||
        !arm_conv_specialized_nhwc_geometry_is_valid(
            &params->stride, &params->padding, &params->dilation, input_dims, filter_dims, output_dims))
    {
        return false;
    }

    const bool packed = params->weight_format == ARM_NN_WEIGHT_FORMAT_NT_N_PACKED;
    const bool is_1d = filter_dims->h == 1;
    const arm_conv_specialized_candidate *candidates = is_1d ? arm_conv_1d_available_f32 : arm_conv_2d_available_f32;
    const size_t count =
        is_1d ? ARM_CONV_ARRAY_SIZE(arm_conv_1d_available_f32) : ARM_CONV_ARRAY_SIZE(arm_conv_2d_available_f32);
    for (size_t i = 0; i < count; ++i)
    {
        if ((packed ? candidates[i].packed : candidates[i].standard) &&
            arm_conv_specialized_requirements_match(
                &candidates[i].requirements, filter_dims, &params->stride, &params->padding, &params->dilation))
        {
            if (selection)
            {
                selection->index = i;
                selection->is_1d = is_1d;
                selection->packed = packed;
            }
            return true;
        }
    }
    return false;
}

#endif /* ARM_NN_ENABLE_F32 && !NN_DISABLE_SPECIALIZATION */
