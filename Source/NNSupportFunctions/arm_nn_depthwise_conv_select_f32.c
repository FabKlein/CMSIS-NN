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
 * Title:        arm_nn_depthwise_conv_select_f32.c
 * Description:  F32 depthwise specialization metadata and selection
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_depthwise_conv_select_f32.h"
#include "Internal/arm_depthwise_conv_specialized_common.h"

#if ARM_NN_ENABLE_F32 && !defined(NN_DISABLE_SPECIALIZATION)

/* The callback argument is deliberately unused so queries do not link kernel code. */
static const arm_dw_requirements arm_dw_requirements_nhwc_f32[] = {
    #define ARM_DW_SPEC(H, W, CALL, ...) {.filter_h = H, .filter_w = W, __VA_ARGS__},
    #include "Internal/arm_depthwise_conv_specialized_registry_f32.h"
    #undef ARM_DW_SPEC
};

bool arm_depthwise_conv_select_specialized_f32(const cmsis_nn_dw_conv_params_f32 *params,
                                               const cmsis_nn_dims *input_dims,
                                               const cmsis_nn_dims *filter_dims,
                                               const cmsis_nn_dims *output_dims,
                                               arm_nn_dw_kernel_layout_f32 kernel_layout,
                                               size_t *index)
{
    if (index)
    {
        *index = 0;
    }
    if (kernel_layout != ARM_NN_DW_KERNEL_KC)
    {
        return false;
    }
    for (size_t i = 0; i < sizeof(arm_dw_requirements_nhwc_f32) / sizeof(arm_dw_requirements_nhwc_f32[0]); ++i)
    {
        if (arm_dw_spec_matches(&arm_dw_requirements_nhwc_f32[i],
                                input_dims,
                                filter_dims,
                                output_dims,
                                params->stride,
                                params->padding,
                                params->dilation,
                                params->ch_mult))
        {
            if (index)
            {
                *index = i;
            }
            return true;
        }
    }
    return false;
}

#endif /* ARM_NN_ENABLE_F32 && !NN_DISABLE_SPECIALIZATION */
