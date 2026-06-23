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
 * Title:        arm_nn_tensor_utils.h
 * Description:  Internal tensor-size validation helper
 *
 * $Date:        8 September 2026
 * $Revision:    V.1.0.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#ifndef ARM_NN_TENSOR_UTILS_H
#define ARM_NN_TENSOR_UTILS_H

#include <stdbool.h>
#include <stdint.h>

/*
 * Require positive dimensions and an element count no greater than INT32_MAX.
 * Check each partial product before extending it so every int64_t multiplication
 * combines at most two positive int32_t values. This bounds elements, not bytes.
 */
static inline bool arm_nn_tensor_size_is_valid(int32_t n, int32_t h, int32_t w, int32_t c)
{
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0)
    {
        return false;
    }

    int64_t size = (int64_t)n * h;
    if (size > INT32_MAX)
    {
        return false;
    }

    size *= w;
    if (size > INT32_MAX)
    {
        return false;
    }

    return size * c <= INT32_MAX;
}

#endif /* ARM_NN_TENSOR_UTILS_H */
