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
 * Title:        arm_py_common.hpp
 * Description:  Common helpers and types for optional Python module
 *
 * $Date:        27 October 2026
 * $Revision:    V.1.0.1
 *
 * Target :  Host/Python
 * -------------------------------------------------------------------- */

#ifndef ARM_PY_BINDINGS_COMMON_HPP
#define ARM_PY_BINDINGS_COMMON_HPP

#include <array>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

extern "C" {
#include "arm_nn_types.h"
}

namespace py = pybind11;

enum class Backend
{
    MVE,
    DSP,
    SCALAR,
};

enum class DataType
{
    A8W4,
    A8W8,
    A16W8,
};

enum class CortexM
{
    M0,
    M0PLUS,
    M3,
    M4,
    M7,
    M23,
    M33,
    M35P,
    M55,
    M85,
};

[[noreturn]] static inline void throw_data_type_not_compiled(DataType data_type)
{
    switch (data_type)
    {
    case DataType::A8W4:
        throw py::value_error("A8W4 support was not compiled; enable ARM_NN_ENABLE_INT4");
    case DataType::A8W8:
        throw py::value_error("A8W8 support was not compiled; enable ARM_NN_ENABLE_INT8");
    case DataType::A16W8:
        throw py::value_error("A16W8 support was not compiled; enable ARM_NN_ENABLE_INT16");
    }

    throw py::value_error("unknown data type");
}

static inline cmsis_nn_dims make_dims(const std::array<int32_t, 4> &nhwc)
{
    cmsis_nn_dims d;
    d.n = nhwc[0];
    d.h = nhwc[1];
    d.w = nhwc[2];
    d.c = nhwc[3];
    return d;
}

#endif // ARM_PY_BINDINGS_COMMON_HPP
