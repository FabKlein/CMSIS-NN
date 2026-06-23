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
 * Title:        arm_nn_conv_resolve_f32.c
 * Description:  Typed F32 specialized convolution kernel resolution
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "Internal/arm_conv_select_f32.h"
#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32 && !defined(NN_DISABLE_SPECIALIZATION)

static const struct
{
    arm_conv_1d_kernel_f32 packed;
    arm_conv_1d_kernel_f32 standard;
} arm_conv_1d_specs_f32[] = {
    #define ARM_CONV_1D_SPEC(W, PACKED, STANDARD, ...) {PACKED, STANDARD},
    #define ARM_CONV_2D_SPEC(H, W, PACKED, ...)
    #include "Internal/arm_conv_specialized_registry_f32.h"
    #undef ARM_CONV_2D_SPEC
    #undef ARM_CONV_1D_SPEC
};

static const arm_conv_2d_kernel_f32 arm_conv_2d_specs_f32[] = {
    #define ARM_CONV_1D_SPEC(W, PACKED, STANDARD, ...)
    #define ARM_CONV_2D_SPEC(H, W, PACKED, ...) PACKED,
    #include "Internal/arm_conv_specialized_registry_f32.h"
    #undef ARM_CONV_2D_SPEC
    #undef ARM_CONV_1D_SPEC
};

arm_conv_kernels_f32 arm_conv_resolve_specialized_f32(const arm_conv_selection_f32 *selection)
{
    arm_conv_kernels_f32 kernels = {NULL, NULL};
    if (selection->is_1d)
    {
        kernels.kernel_1d = selection->packed ? arm_conv_1d_specs_f32[selection->index].packed
                                              : arm_conv_1d_specs_f32[selection->index].standard;
    }
    else
    {
        kernels.kernel_2d = arm_conv_2d_specs_f32[selection->index];
    }
    return kernels;
}

#endif /* ARM_NN_ENABLE_F32 && !NN_DISABLE_SPECIALIZATION */
