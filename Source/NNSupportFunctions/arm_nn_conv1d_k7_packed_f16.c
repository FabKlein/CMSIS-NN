/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_conv1d_k7_packed_f16.c
 * Description:  Support: NHWC 1D convolution kernel size 7 for packed f16 weights
 *
 * $Date:        8 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 *
 * -------------------------------------------------------------------- */

#include "arm_nnsupportfunctions.h"

#if ARM_NN_ENABLE_F16

#define ARM_NN_CONV1D_PACKED_NAME arm_nn_conv1d_k7_packed_f16
#define ARM_NN_CONV1D_PACKED_KERNEL_W 7
#include "Internal/arm_nn_conv1d_packed_f16_template.h"
#undef ARM_NN_CONV1D_PACKED_KERNEL_W
#undef ARM_NN_CONV1D_PACKED_NAME

#endif /* ARM_NN_ENABLE_F16 */
