/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_conv2d_2x2_packed_f32.c
 * Description:  Support: NHWC specialized convolution kernel size 2x2 for packed f32 weights
 *
 * $Date:        9 September 2026
 * $Revision:    V.1.0.1
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F32

    #define ARM_NN_CONV2D_PACKED_NAME arm_nn_conv2d_2x2_packed_f32
    #define ARM_NN_CONV2D_PACKED_KERNEL_H 2
    #define ARM_NN_CONV2D_PACKED_KERNEL_W 2
    #include "Internal/arm_nn_conv2d_packed_f32_template.h"
    #undef ARM_NN_CONV2D_PACKED_KERNEL_W
    #undef ARM_NN_CONV2D_PACKED_KERNEL_H
    #undef ARM_NN_CONV2D_PACKED_NAME

#endif /* ARM_NN_ENABLE_F32 */
