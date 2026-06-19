#
# SPDX-FileCopyrightText: Copyright 2019-2023, 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
set(SRC "")
if (ARM_NN_ENABLE_INT8)
  file(GLOB SRC_S8 "./*_s8*.c")
  list(APPEND SRC ${SRC_S8} arm_relu_q7.c)
endif()
if (ARM_NN_ENABLE_INT8 OR ARM_NN_ENABLE_INT16)
  list(APPEND SRC arm_nn_activation_s16.c)
endif()
if (ARM_NN_ENABLE_INT16)
  file(GLOB SRC_S16 "./*_s16*.c")
  list(REMOVE_ITEM SRC_S16 "${CMAKE_CURRENT_LIST_DIR}/arm_nn_activation_s16.c")
  list(APPEND SRC ${SRC_S16} arm_relu_q15.c)
endif()
set(SRC_FLT "")
if (ARM_NN_ENABLE_F32)
  list(APPEND SRC_FLT arm_nn_activation_f32.c)
endif()
if (ARM_NN_ENABLE_F16)
  list(APPEND SRC_FLT arm_nn_activation_f16.c)
endif()
target_sources(cmsis-nn PRIVATE ${SRC} ${SRC_FLT})
