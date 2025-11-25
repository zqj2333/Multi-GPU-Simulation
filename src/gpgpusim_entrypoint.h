// Copyright (c) 2009-2011, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef GPGPUSIM_ENTRYPOINT_H_INCLUDED
#define GPGPUSIM_ENTRYPOINT_H_INCLUDED

#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include "abstract_hardware_model.h"

// extern time_t g_simulation_starttime;
class gpgpu_context;

class GPGPUsim_ctx {
 public:
  GPGPUsim_ctx(gpgpu_context *ctx) {
    g_sim_active = false;
    g_sim_done = true;
    break_limit = false;
    g_sim_lock = PTHREAD_MUTEX_INITIALIZER;

    g_the_gpu_config = NULL;
    //g_the_gpu = NULL;
    g_multi_gpu = NULL;
    g_nvlink = NULL;
    n_gpu = 0;
    compiler_assist = true;
    malloc_cnt = 0;
    kernel_cnt = 0;
    sharp_enable = false;
    identifier_pattern = 0; // 0: two optimal kernel implementation; 1: original GEMM
    fence_size = 8; // how many warps in a TB
    nvlink_ctrl_size = 16;
    nvlink_data_size = 128;
    ldst_tag = 0;
    pc_table_selection = 0;

    profile_mode = 0;

    // with_TB_sync = false;
    // with_inst_sync = false;
    with_TB_sync = false;
    with_inst_sync = false;

    g_stream_manager = NULL;
    the_cude_device = NULL;
    the_context = NULL;
    gpgpu_ctx = ctx;
    current_kernel_done = NULL;
  }

  // struct gpgpu_ptx_sim_arg *grid_params;

  sem_t g_sim_signal_start;
  sem_t g_sim_signal_finish;
  sem_t g_sim_signal_exit;
  time_t g_simulation_starttime;
  pthread_t g_simulation_thread;

  class gpgpu_sim_config *g_the_gpu_config;
  //class gpgpu_sim *g_the_gpu;

  int n_gpu;
  class gpgpu_sim **g_multi_gpu;
  class nvlink_system * g_nvlink;
  class nvlink_hub ** g_hub;
  class synchronizer** g_sync;
  bool * current_kernel_done;

  class stream_manager *g_stream_manager;

  struct _cuda_device_id *the_cude_device;
  struct CUctx_st *the_context;
  gpgpu_context *gpgpu_ctx;

  pthread_mutex_t g_sim_lock;
  bool g_sim_active;
  bool g_sim_done;
  bool break_limit;

  int profile_mode;

  bool compiler_assist;
  int malloc_cnt;
  int kernel_cnt;
  bool sharp_enable;
  int identifier_pattern;
  int fence_size;
  bool with_TB_sync;
  bool with_inst_sync;
  int nvlink_data_size;
  int nvlink_ctrl_size;
  unsigned long long int ldst_tag;
  int pc_table_selection;
  int nvls_cache_size;
  
  // stats for nvlink system
  // int*gpu2switch_cnt;
  // int*switch2gpu_cnt;
};

#endif
