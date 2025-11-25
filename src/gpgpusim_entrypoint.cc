// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ivan Sham,
// Andrew Turner, Ali Bakhoda, The University of British Columbia
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

#include "gpgpusim_entrypoint.h"
#include <stdio.h>

#include "../libcuda/gpgpu_context.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx_parser.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "option_parser.h"
#include "stream_manager.h"
#include "gpgpu-sim/local_interconnect.h"
#include "gpgpu-sim/synchronizer.h"
#include "gpgpu-sim/nvlink.h"
#include "gpgpu-sim/mem_fetch.h"
#include "gpgpu-sim/hub.h"

// #define MF_TUP_BEGIN(X) enum X {
// #define MF_TUP(X) X
// #define MF_TUP_END(X) \
//   }                   \
//   ;
// #include "gpgpu-sim/mem_fetch_status.tup"
// #undef MF_TUP_BEGIN
// #undef MF_TUP
// #undef MF_TUP_END

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

static int sg_argc = 3;
static const char *sg_argv[] = {"", "-config", "gpgpusim.config"};

/*
void *gpgpu_sim_thread_sequential(void *ctx_ptr) {
  gpgpu_context *ctx = (gpgpu_context *)ctx_ptr;
  // at most one kernel running at a time
  bool done;
  do {
    sem_wait(&(ctx->the_gpgpusim->g_sim_signal_start));
    done = true;
    if (ctx->the_gpgpusim->g_the_gpu->get_more_cta_left()) {
      done = false;
      ctx->the_gpgpusim->g_the_gpu->init();
      while (ctx->the_gpgpusim->g_the_gpu->active()) {
        ctx->the_gpgpusim->g_the_gpu->cycle();
        ctx->the_gpgpusim->g_the_gpu->deadlock_check();
      }
      ctx->the_gpgpusim->g_the_gpu->print_stats();
      ctx->the_gpgpusim->g_the_gpu->update_stats();
      ctx->print_simulation_time();
    }
    sem_post(&(ctx->the_gpgpusim->g_sim_signal_finish));
  } while (!done);
  sem_post(&(ctx->the_gpgpusim->g_sim_signal_exit));
  return NULL;
}*/

static void termination_callback() {
  printf("GPGPU-Sim: *** exit detected ***\n");
  fflush(stdout);
}



void *gpgpu_sim_thread_concurrent(void *ctx_ptr) {
  gpgpu_context *ctx = (gpgpu_context *)ctx_ptr;
  atexit(termination_callback);
  int n_gpu = ctx->the_gpgpusim->n_gpu;
  // int remote_access_ld[4] = {0,0,0,0};
  // int remote_access_st = 0;
  int nvlink_system_freq_cnt = 0;
  // concurrent kernel execution simulation thread
  do {
    //printf("outer loop\n");
    //if (g_debug_execution >= 3) {
      printf(
          "GPGPU-Sim: *** simulation thread starting and spinning waiting for "
          "work ***\n");
      fflush(stdout);
    //}
    while (ctx->the_gpgpusim->g_stream_manager->empty_protected() &&
           !ctx->the_gpgpusim->g_sim_done)
      ;
    
    //if (g_debug_execution >= 3) {
      printf("GPGPU-Sim: ** START simulation thread (detected work) **\n");
      ctx->the_gpgpusim->g_stream_manager->print(stdout);
      fflush(stdout);
    //}
    //printf("enter lock 1\n");
    pthread_mutex_lock(&(ctx->the_gpgpusim->g_sim_lock));
    ctx->the_gpgpusim->g_sim_active = true;
    pthread_mutex_unlock(&(ctx->the_gpgpusim->g_sim_lock));
    //printf("pass lock 1\n");
    bool active = false;
    bool sim_cycles = false;
    for(int dev_id=0;dev_id<n_gpu;++dev_id)
    {
      ctx->the_gpgpusim->g_multi_gpu[dev_id]->init();
    }
    // ctx->the_gpgpusim->g_the_gpu->init();
    do {
      // check if a kernel has completed
      // launch operation on device if one is pending and can be run

      // Need to break this loop when a kernel completes. This was a
      // source of non-deterministic behaviour in GPGPU-Sim (bug 147).
      // If another stream operation is available, g_the_gpu remains active,
      // causing this loop to not break. If the next operation happens to be
      // another kernel, the gpu is not re-initialized and the inter-kernel
      // behaviour may be incorrect. Check that a kernel has finished and
      // no other kernel is currently running.
      //printf("DEBUG: start this loop\n");
      bool gpu_active = false;
      for(int dev_id=0;dev_id<n_gpu;++dev_id)
      {
        gpu_active = gpu_active || ctx->the_gpgpusim->g_multi_gpu[dev_id]->active();
      }
      gpu_active = gpu_active || ctx->the_gpgpusim->g_nvlink->busy();
      //printf("signal0; %d\n",gpu_active);
      if (ctx->the_gpgpusim->g_stream_manager->operation(&sim_cycles) && !gpu_active)
      {
        //printf("break;\n");
        break;
      }
      //printf("signal1;\n");
      // functional simulation
      if (ctx->the_gpgpusim->g_multi_gpu[0]->is_functional_sim()) {
        assert(0);
      }
      //printf("signal2;\n");
      // performance simulation
      
      if (gpu_active) {
        //printf("gpu_active\n");

        bool core_domain = ctx->the_gpgpusim->g_multi_gpu[0]->is_core_domain();
        if(core_domain)
        {
          nvlink_system_freq_cnt = (nvlink_system_freq_cnt + 1) % 2;
          //printf("Yes, core_domain\n");
        }
        bool nvlink_system_domain = core_domain && !nvlink_system_freq_cnt;

        //printf("deadlock flag 1\n");
if(nvlink_system_domain)
{
        //printf("Yes, core_domain\n");

        // pop from interconnect to nvlink
        //printf("deadlock flag 2\n");
        if(1)
        {
          ctx->the_gpgpusim->g_nvlink->cycle();

          //printf("deadlock flag 9\n");

          for(int dev_id=0;dev_id<n_gpu;++dev_id)
          {
            ctx->the_gpgpusim->g_sync[dev_id]->cycle();
          }

          //printf("deadlock flag 3\n");
          //int idle_cnt = 0;
          for (unsigned dev_id = 0; dev_id < n_gpu; dev_id++) 
          {
          //idle_cnt = 0;
          while(1)
          {
            //printf("deadlock3\n");
            mem_fetch *mf = (mem_fetch*)ctx->the_gpgpusim->g_multi_gpu[dev_id]->m_interconnect->icnt_probe_io(); // probe_message();
            if (mf) 
            {
              // printf("yes, gpu ready\n");
              // fflush(stdout);
              // printf("no support for remote access\n");
              // assert(0);
              unsigned packet_size;
              int in_gpu, out_gpu;
              bool is_modify_request = false, is_read_request = false, is_modify_response = false, is_read_response = false;
              int trace_size = 0;
              if(mf->is_inst_sync())
              {
                //
              }
              else if(mf->get_is_write() || mf->isatomic())
              {
                if(mf->get_status() == IN_ICNT_TO_MEM)
                {
                  is_modify_request = true;
                  packet_size = mf->size();
                  in_gpu = mf->get_src_gpu();
                  out_gpu = mf->get_dst_gpu();
                  trace_size = mf->size() - 8;
                }
                else if(mf->get_status() == IN_ICNT_TO_SHADER)
                {
                  is_modify_response = true;
                  packet_size = mf->get_ctrl_size();
                  in_gpu = mf->get_dst_gpu();
                  out_gpu = mf->get_src_gpu();
                  trace_size = mf->size() - 8;
                }
              }
              else
              {
                if(mf->get_status() == IN_ICNT_TO_MEM)
                {
                  is_read_request = true;
                  packet_size = mf->get_ctrl_size();
                  in_gpu = mf->get_src_gpu();
                  out_gpu = mf->get_dst_gpu();
                  trace_size = mf->size() - 8;
                }
                else if(mf->get_status() == IN_ICNT_TO_SHADER)
                {
                  is_read_response = true;
                  packet_size = mf->size();
                  in_gpu = mf->get_dst_gpu();
                  out_gpu = mf->get_src_gpu();
                  trace_size = mf->size() - 8;
                }
              }

              // if(in_gpu!=dev_id)
              // {
              //   printf("GPU ID Error in NVLink!\n");
              //   assert(0);
              // }

              if (ctx->the_gpgpusim->g_hub[dev_id]->can_send(mf)) 
              {
                //idle_cnt = 0;
                ctx->the_gpgpusim->g_hub[dev_id]->send(mf);
                ctx->the_gpgpusim->g_multi_gpu[dev_id]->m_interconnect->icnt_pop_io();

                if(is_read_request)
                {
                  printf("Remote Access Trace: Type = R_0 , VA = %llx , GPU = %d , SrcGPU = %d , DstGPU = %d , Cycle = %d\n",mf->get_vaddr(),dev_id,mf->get_src_gpu(),mf->get_dst_gpu(),ctx->the_gpgpusim->g_multi_gpu[dev_id]->gpu_sim_cycle);
                }
                if(is_modify_request)
                {
                  printf("Remote Access Trace: Type = M_0 , VA = %llx , GPU = %d , SrcGPU = %d , DstGPU = %d , Cycle = %d\n",mf->get_vaddr(),dev_id,mf->get_src_gpu(),mf->get_dst_gpu(),ctx->the_gpgpusim->g_multi_gpu[dev_id]->gpu_sim_cycle);
                }
                if(is_read_response)
                {
                  printf("Remote Access Trace: Type = R_5 , VA = %llx , GPU = %d , SrcGPU = %d , DstGPU = %d , Cycle = %d\n",mf->get_vaddr(),dev_id,mf->get_src_gpu(),mf->get_dst_gpu(),ctx->the_gpgpusim->g_multi_gpu[dev_id]->gpu_sim_cycle);
                }
                if(is_modify_response)
                {
                  printf("Remote Access Trace: Type = M_5 , VA = %llx , GPU = %d , SrcGPU = %d , DstGPU = %d , Cycle = %d\n",mf->get_vaddr(),dev_id,mf->get_src_gpu(),mf->get_dst_gpu(),ctx->the_gpgpusim->g_multi_gpu[dev_id]->gpu_sim_cycle);
                }

                // if(mf->get_status() == IN_ICNT_TO_MEM&&!(mf->get_is_write() || mf->isatomic()))
                // {
                //   // remote_access_ld[0] += (mf->size() - 8);
                //   size_t cur_addr = mf->get_addr();
                //   if(cur_addr>=0xc0000000&&cur_addr<0xc0010000)
                //     remote_access_ld[0] += (mf->size() - 8);
                //   else if(cur_addr>=0xc0010000&&cur_addr<0xc0020000)
                //     remote_access_ld[1] += (mf->size() - 8);
                //   else if(cur_addr>=0xc0020000&&cur_addr<0xc0040000)
                //     remote_access_ld[2] += (mf->size() - 8);
                //   else if(cur_addr>=0xc0040000&&cur_addr<0xc0060000)
                //     remote_access_ld[3] += (mf->size() - 8);
                //   else
                //   {
                //     printf("error addr\n");
                //     assert(0);
                //   }
                // }
                //if(mf->get_status() == IN_ICNT_TO_MEM&&(mf->get_is_write() || mf->isatomic()))ctx->the_gpgpusim->gpu2switch_cnt[dev_id] += (mf->size() - 8);
                //printf("GPU %d send a request into nvlink, to GPU %d, size: %d\n",in_gpu, out_gpu, packet_size);
                //fflush(stdout);
              }
              else
              {
                // idle_cnt += 1;
                break;
              }
            }
            else
            {
              break;
            }
            // else
            // {
            //   idle_cnt += 1;
            // }
            // if(idle_cnt > n_gpu)
            // {
            //   break;
            // }
          }
          }
          //printf("deadlock flag 4\n");
        }

        // printf("remote_access_ld: %d,%d,%d,%d\n", remote_access_ld[0],remote_access_ld[1],remote_access_ld[2],remote_access_ld[3]);
}
        //printf("deadlock flag 2\n");
        //printf("deadlock flag 5\n");
        for(int dev_id=0;dev_id<n_gpu;++dev_id)
        {
          ctx->the_gpgpusim->g_multi_gpu[dev_id]->cycle();
        }
        //printf("deadlock flag 6\n");
        //printf("deadlock flag 3\n");

if(nvlink_system_domain)
{
        for(int dev_id=0;dev_id<n_gpu;++dev_id)
        {
          ctx->the_gpgpusim->g_hub[dev_id]->cycle();
        }

        //printf("deadlock flag 7\n");

        // move memory request from nvlink into interconnect
        if(1)
        {
          //printf("deadlock flag 7\n");
          for (unsigned dev_id = 0; dev_id < n_gpu; dev_id++) {
          // idle_cnt = 0;
          while(1)
          {
            //printf("deadlock4\n");
            //if(!ctx->the_gpgpusim->g_multi_gpu[dev_id]->icnt_io_has_buffer())
            //{
              mem_fetch *mf = (mem_fetch*)(ctx->the_gpgpusim->g_hub[dev_id]->probe(dev_id));
              if (mf)
              {

                // printf("nvlink ready\n");
                //fflush(stdout);
                // assert(0);
                unsigned packet_size;
                int in_gpu, out_gpu;
                int out_device;
                if(mf->get_is_write() || mf->isatomic())
                {
                  if(mf->get_status() == IN_ICNT_TO_MEM)
                  {
                    packet_size = mf->size();
                    out_device = ctx->the_gpgpusim->g_multi_gpu[dev_id]->m_shader_config->mem2device(mf->get_sub_partition_id());
                    printf("Remote Access Trace: Type = M_2 , VA = %llx , GPU = %d , SrcGPU = %d , DstGPU = %d , Cycle = %d\n",mf->get_vaddr(),dev_id,mf->get_src_gpu(),mf->get_dst_gpu(),ctx->the_gpgpusim->g_multi_gpu[dev_id]->gpu_sim_cycle);
                    //printf("To mem packet size = %d, data size = %d\n",packet_size,mf->size());
                  }
                  else if(mf->get_status() == IN_ICNT_TO_SHADER)
                  {
                    packet_size = mf->get_ctrl_size();
                    out_device = mf->get_tpc();
                    printf("Remote Access Trace: Type = M_7 , VA = %llx , GPU = %d , SrcGPU = %d , DstGPU = %d , Cycle = %d\n",mf->get_vaddr(),dev_id,mf->get_src_gpu(),mf->get_dst_gpu(),ctx->the_gpgpusim->g_multi_gpu[dev_id]->gpu_sim_cycle);
                    //printf("To shader packet size = %d, data size = %d\n",packet_size,mf->size());
                  }
                  else
                  {
                    printf("status error\n");
                    assert(0);
                  }
                }
                else
                {
                  if(mf->get_status() == IN_ICNT_TO_MEM)
                  {
                    packet_size = mf->get_ctrl_size();
                    out_device = ctx->the_gpgpusim->g_multi_gpu[dev_id]->m_shader_config->mem2device(mf->get_sub_partition_id());
                    printf("Remote Access Trace: Type = R_2 , VA = %llx , GPU = %d , SrcGPU = %d , DstGPU = %d , Cycle = %d\n",mf->get_vaddr(),dev_id,mf->get_src_gpu(),mf->get_dst_gpu(),ctx->the_gpgpusim->g_multi_gpu[dev_id]->gpu_sim_cycle);
                    //printf("To mem packet size = %d, data size = %d\n",packet_size,mf->size());
                  }
                  else if(mf->get_status() == IN_ICNT_TO_SHADER)
                  {
                    packet_size = mf->size();
                    out_device = mf->get_tpc();
                    printf("Remote Access Trace: Type = R_7 , VA = %llx , GPU = %d , SrcGPU = %d , DstGPU = %d , Cycle = %d\n",mf->get_vaddr(),dev_id,mf->get_src_gpu(),mf->get_dst_gpu(),ctx->the_gpgpusim->g_multi_gpu[dev_id]->gpu_sim_cycle);
                    //printf("To shader packet size = %d, data size = %d\n",packet_size,mf->size());
                  }
                  else
                  {
                    printf("status error\n");
                    assert(0);
                  }
                }
                if(ctx->the_gpgpusim->g_multi_gpu[dev_id]->m_interconnect->icnt_io_has_buffer(packet_size,out_device)) // can_push_message(mf, packet_size))
                {
                  ctx->the_gpgpusim->g_multi_gpu[dev_id]->m_interconnect->icnt_push_io(out_device,mf,packet_size);//push_message(mf, packet_size);
                  ctx->the_gpgpusim->g_hub[dev_id]->pop(dev_id);
                  //printf("GPU %d recv a request from nvlink, size: %d\n",dev_id, packet_size);
                  //fflush(stdout);
                }
                else
                {
                  break;
                }
              }
              // }
              else
              {
                break;
              }
          }
          }
          //printf("deadlock flag 8\n");
        }
        //printf("deadlock flag 8\n");
}
        fflush(stdout);
        //printf("deadlock flag 4\n");

        ctx->decrement_kernel_latency();
        for(int dev_id=0;dev_id<n_gpu;++dev_id)
        {
          if(ctx->the_gpgpusim->g_multi_gpu[dev_id]->active())
          {
            ctx->the_gpgpusim->g_multi_gpu[dev_id]->deadlock_check();
          }
        }
        //printf("deadlock flag 9\n");
        //printf("sim_cycles = true;\n");
        sim_cycles = true;
      } else {
        //printf("deadlock flag 10\n");
        bool is_cycle_insn_cta_max_hit = false;
        for(int dev_id=0;dev_id<n_gpu;++dev_id)
        {
          is_cycle_insn_cta_max_hit = is_cycle_insn_cta_max_hit || ctx->the_gpgpusim->g_multi_gpu[dev_id]->cycle_insn_cta_max_hit();
        }
        if (is_cycle_insn_cta_max_hit) {
          ctx->the_gpgpusim->g_stream_manager->stop_all_running_kernels();
          ctx->the_gpgpusim->g_sim_done = true;
          ctx->the_gpgpusim->break_limit = true;
        }
      }
      //printf("signal3;\n");
      gpu_active = false;
      for(int dev_id=0;dev_id<n_gpu;++dev_id)
      {
        gpu_active = gpu_active || ctx->the_gpgpusim->g_multi_gpu[dev_id]->active();
      }
      //printf("signal4;%d\n",gpu_active);
      active = gpu_active ||
               !(ctx->the_gpgpusim->g_stream_manager->empty_protected());
      
      active = active || ctx->the_gpgpusim->g_nvlink->busy();
      //printf("signal5;%d\n",active);
      // printf("signal5;\n");
    } while (active && !ctx->the_gpgpusim->g_sim_done);
    // printf("signal3;\n");
    // printf("GPGPU-Sim: ** STOP simulation thread (no work) **\n");
    // fflush(stdout);
    if (g_debug_execution >= 3) {
      printf("GPGPU-Sim: ** STOP simulation thread (no work) **\n");
      fflush(stdout);
    }
    if (sim_cycles) {
      for(int dev_id=0;dev_id<n_gpu;++dev_id)
      {
        ctx->the_gpgpusim->g_multi_gpu[dev_id]->print_stats();
        ctx->the_gpgpusim->g_multi_gpu[dev_id]->update_stats();
        ctx->the_gpgpusim->g_nvlink->print_stats();
        ctx->print_simulation_time();
      }
    }
    //printf("enter lock 2\n");
    pthread_mutex_lock(&(ctx->the_gpgpusim->g_sim_lock));
    ctx->the_gpgpusim->g_sim_active = false;
    pthread_mutex_unlock(&(ctx->the_gpgpusim->g_sim_lock));
    //printf("pass lock 2\n");
  } while (!ctx->the_gpgpusim->g_sim_done);

  // printf("remote_access_ld: %d\n", remote_access_ld);
  // printf("remote_access_st: %d\n", remote_access_st);

  printf("GPGPU-Sim: *** simulation thread exiting ***\n");
  fflush(stdout);

  if (ctx->the_gpgpusim->break_limit) {
    printf(
        "GPGPU-Sim: ** break due to reaching the maximum cycles (or "
        "instructions) **\n");
    exit(1);
  }

  sem_post(&(ctx->the_gpgpusim->g_sim_signal_exit));
  return NULL;
}

void addr_trans::map_addr(size_t va, int gpu_id, size_t pa)
{
  phy_addr the_pa;
  the_pa.gpu_id = gpu_id;
  the_pa.pa = pa;

  atmap<size_t,phy_addr>::iterator itr; 
  itr = page_table.find(va);
  if(itr!=page_table.end())
  {
    page_table.erase(itr);
  }

  // printf("map addr %llx to %d:%llx\n",va,gpu_id,pa);

  page_table[va] = the_pa;
  // page_table[va] = the_pa;
}

phy_addr addr_trans::get_pa(size_t va)
{
  atmap<size_t,phy_addr>::iterator itr; 
  itr = page_table.find(va);
  if(itr!=page_table.end())
  {
    return itr->second;
  }
  else
  {
    printf("can not find the virtual address %llx\n",va);
    assert(0);
  }
  return itr->second;
}

void gpgpu_context::synchronize() {
  printf("GPGPU-Sim: synchronize waiting for inactive GPU simulation\n");
  the_gpgpusim->g_stream_manager->print(stdout);
  fflush(stdout);
  //    sem_wait(&g_sim_signal_finish);
  bool done = false;
  do {
    pthread_mutex_lock(&(the_gpgpusim->g_sim_lock));
    done = (the_gpgpusim->g_stream_manager->empty() &&
            !the_gpgpusim->g_sim_active) ||
           the_gpgpusim->g_sim_done;
    pthread_mutex_unlock(&(the_gpgpusim->g_sim_lock));
  } while (!done);
  printf("GPGPU-Sim: detected inactive GPU simulation thread\n");
  fflush(stdout);
  //    sem_post(&g_sim_signal_start);
}

void gpgpu_context::exit_simulation() {
  the_gpgpusim->g_sim_done = true;
  printf("GPGPU-Sim: exit_simulation called\n");
  fflush(stdout);
  sem_wait(&(the_gpgpusim->g_sim_signal_exit));
  printf("GPGPU-Sim: simulation thread signaled exit\n");
  fflush(stdout);
}

// gpgpu_sim *gpgpu_context::gpgpu_ptx_sim_init_perf() {
//   srand(1);
//   print_splash();
//   func_sim->read_sim_environment_variables();
//   ptx_parser->read_parser_environment_variables();
//   option_parser_t opp = option_parser_create();

//   ptx_reg_options(opp);
//   func_sim->ptx_opcocde_latency_options(opp);

//   icnt_reg_options(opp);
//   the_gpgpusim->g_the_gpu_config = new gpgpu_sim_config(this);
//   the_gpgpusim->g_the_gpu_config->reg_options(
//       opp);  // register GPU microrachitecture options

//   option_parser_cmdline(opp, sg_argc, sg_argv);  // parse configuration options
//   fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
//   option_parser_print(opp, stdout);
//   // Set the Numeric locale to a standard locale where a decimal point is a
//   // "dot" not a "comma" so it does the parsing correctly independent of the
//   // system environment variables
//   assert(setlocale(LC_NUMERIC, "C"));
//   the_gpgpusim->g_the_gpu_config->init();

//   the_gpgpusim->g_the_gpu =
//       new exec_gpgpu_sim(*(the_gpgpusim->g_the_gpu_config), this);
//   the_gpgpusim->g_stream_manager = new stream_manager(
//       (the_gpgpusim->g_the_gpu), func_sim->g_cuda_launch_blocking);

//   the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

//   sem_init(&(the_gpgpusim->g_sim_signal_start), 0, 0);
//   sem_init(&(the_gpgpusim->g_sim_signal_finish), 0, 0);
//   sem_init(&(the_gpgpusim->g_sim_signal_exit), 0, 0);

//   return the_gpgpusim->g_the_gpu;
// }

// data placement
void *gpgpu_context::gpu_malloc(size_t size) {
  unsigned long long result = m_dev_malloc;
  if (g_debug_execution >= 3) {
    printf(
        "GPGPU-Sim PTX: allocating %zu bytes on GPU starting at address "
        "0x%Lx\n",
        size, m_dev_malloc);
    fflush(stdout);
  }

  //simple map
  size_t n_subpage = size / 256;
  if (size % 256)
    n_subpage++;
  
  if(!the_gpgpusim->compiler_assist)
  {
    for(size_t page_id = 0; page_id < n_subpage; page_id ++)
    {
      int gpu_id = page_id % the_gpgpusim->n_gpu;
      the_addrtrans->map_addr(m_dev_malloc + page_id * 256, gpu_id, the_gpgpusim->g_multi_gpu[gpu_id]->m_dev_malloc);
      the_gpgpusim->g_multi_gpu[gpu_id]->gpu_malloc(256);
    }
  }
  else
  {
    std::string plan_name = "data_placement_";
    plan_name = plan_name + std::to_string(the_gpgpusim->malloc_cnt);
    printf("malloc with %s\n",plan_name.c_str());
    std::ifstream ofs(plan_name,std::ios::in|std::ios::binary);
    int*data_placement = new int[n_subpage];
    printf("num_of_page:%d\n",n_subpage);
	  ofs.read((char*)data_placement,sizeof(int)*n_subpage);
    for(size_t page_id = 0; page_id < n_subpage; page_id ++)
    {
      int gpu_id = data_placement[page_id];
      // printf("%d ",gpu_id);
      the_addrtrans->map_addr(m_dev_malloc + page_id * 256, gpu_id, the_gpgpusim->g_multi_gpu[gpu_id]->m_dev_malloc);
      the_gpgpusim->g_multi_gpu[gpu_id]->gpu_malloc(256);
    }
    // printf("\n");
    ofs.close();
    the_gpgpusim->malloc_cnt++;
  }

  m_dev_malloc += size;
  if (size % 256)
    m_dev_malloc += (256 - size % 256);  // align to 256 byte boundaries

  return (void *)result;
}

void gpgpu_context::memcpy_to_gpu(size_t dst_start_addr, const void *src, size_t count)
{
  if(dst_start_addr % 256)
  {
    assert(0);
  }
  size_t cur_cnt = 0;
  while(cur_cnt < count)
  {
    phy_addr phyaddr = the_addrtrans->get_pa(dst_start_addr + cur_cnt);
    //printf("h2d: gpu_id %d, addr %llx\n",phyaddr.gpu_id,phyaddr.pa);
    if(count-cur_cnt>=256)
    {
      the_gpgpusim->g_multi_gpu[phyaddr.gpu_id]->memcpy_to_gpu(phyaddr.pa, (char*)src + cur_cnt, 256);
    }
    else
    {
      the_gpgpusim->g_multi_gpu[phyaddr.gpu_id]->memcpy_to_gpu(phyaddr.pa, (char*)src + cur_cnt, count - cur_cnt);
    }
    cur_cnt += 256;
  }
}
void gpgpu_context::memcpy_from_gpu(void *dst, size_t src_start_addr, size_t count)
{
  if(src_start_addr % 256)
  {
    assert(0);
  }
  size_t cur_cnt = 0;
  while(cur_cnt < count)
  {
    phy_addr phyaddr = the_addrtrans->get_pa(src_start_addr + cur_cnt);
    //printf("d2h: gpu_id %d, addr %llx\n",phyaddr.gpu_id,phyaddr.pa);
    if(count-cur_cnt>=256)
    {
      the_gpgpusim->g_multi_gpu[phyaddr.gpu_id]->memcpy_from_gpu((char*)dst + cur_cnt, phyaddr.pa, 256);
    }
    else
    {
      the_gpgpusim->g_multi_gpu[phyaddr.gpu_id]->memcpy_from_gpu((char*)dst + cur_cnt, phyaddr.pa, count - cur_cnt);
    }
    cur_cnt += 256;
  }
}
void gpgpu_context::memcpy_gpu_to_gpu(size_t dst, size_t src, size_t count)
{
  assert(0);
}

void gpgpu_context::launch(kernel_info_t *kinfo) {
  unsigned cta_size = kinfo->threads_per_cta();
  int n_gpu = the_gpgpusim->n_gpu;
  if ((cta_size / n_gpu) > the_gpgpusim->g_multi_gpu[0]->m_shader_config->n_thread_per_shader) {
    printf(
        "Execution error: Shader kernel CTA (block) size is too large for "
        "microarch config.\n");
    printf("                 CTA size (x*y*z) = %u, max supported = %u\n",
           cta_size, the_gpgpusim->g_multi_gpu[0]->m_shader_config->n_thread_per_shader * n_gpu);
    printf(
        "                 => either change -gpgpu_shader argument in "
        "gpgpusim.config file or\n");
    printf(
        "                 modify the CUDA source to decrease the kernel block "
        "size.\n");
    abort();
  }
  unsigned n = 0;
  for (n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done()) {
      m_running_kernels[n] = kinfo;
      break;
    }
  }
  assert(n < m_running_kernels.size());
}

void icnt_reg_options(unsigned& g_network_mode, char*& g_network_config_filename, inct_config& g_inct_config, class OptionParser* opp) {
      option_parser_register(opp, "-network_mode", OPT_INT32, &g_network_mode,
                         "Interconnection network mode", "1");
      option_parser_register(opp, "-inter_config_file", OPT_CSTR,
                         &g_network_config_filename,
                         "Interconnection network config file", "mesh");

      // parameters for local xbar
      option_parser_register(opp, "-icnt_in_buffer_limit", OPT_UINT32,
                         &g_inct_config.in_buffer_limit, "in_buffer_limit",
                         "64");
      option_parser_register(opp, "-icnt_out_buffer_limit", OPT_UINT32,
                         &g_inct_config.out_buffer_limit, "out_buffer_limit",
                         "64");
      option_parser_register(opp, "-icnt_subnets", OPT_UINT32,
                         &g_inct_config.subnets, "subnets", "2");
      option_parser_register(opp, "-icnt_arbiter_algo", OPT_UINT32,
                         &g_inct_config.arbiter_algo, "arbiter_algo", "1");
      option_parser_register(opp, "-icnt_verbose", OPT_UINT32,
                         &g_inct_config.verbose, "inct_verbose", "0");
      option_parser_register(opp, "-icnt_grant_cycles", OPT_UINT32,
                         &g_inct_config.grant_cycles, "grant_cycles", "1");
      option_parser_register(opp, "-n_io_port", OPT_UINT32,
                         &g_inct_config.n_io_port, "n_io_port", "32");
    }

gpgpu_sim **gpgpu_context::gpgpu_ptx_sim_init_perf() {
  srand(1);
  print_splash();
  func_sim->read_sim_environment_variables();
  ptx_parser->read_parser_environment_variables();
  option_parser_t opp = option_parser_create();
  
  ptx_reg_options(opp);

  func_sim->ptx_opcocde_latency_options(opp);

  struct systeminfo
  {
    int n_gpu;
    bool sharp_enable;
    int identifier_pattern;
    bool with_TB_sync;
    bool with_inst_sync;
    int profile_mode;
    int pc_table_selection;
    int nvls_cache_size;
  };
  // initialize configuration parameters
  systeminfo sysinfo;
  std::string sys_name = "SystemConfig";
  printf("system config with %s\n",sys_name.c_str());
  std::ifstream sysfs(sys_name,std::ios::in|std::ios::binary);
  sysfs.read((char*)&sysinfo,sizeof(systeminfo));
  sysfs.close();
  // the_gpgpusim->n_gpu = sysinfo.n_gpu;
  // the_gpgpusim->sharp_enable = sysinfo.sharp_enable;
  // the_gpgpusim->identifier_pattern = sysinfo.identifier_pattern;
  // the_gpgpusim->with_TB_sync = sysinfo.with_TB_sync;
  // the_gpgpusim->with_inst_sync = sysinfo.with_inst_sync;
  // the_gpgpusim->profile_mode = sysinfo.profile_mode;
  // the_gpgpusim->pc_table_selection = sysinfo.pc_table_selection;
  // the_gpgpusim->nvls_cache_size = sysinfo.nvls_cache_size;

  the_gpgpusim->n_gpu = 4;//sysinfo.n_gpu;
  the_gpgpusim->sharp_enable = 0;//sysinfo.sharp_enable;
  the_gpgpusim->identifier_pattern = 0;//sysinfo.identifier_pattern;
  the_gpgpusim->with_TB_sync = 0;//sysinfo.with_TB_sync;
  the_gpgpusim->with_inst_sync = 0;//sysinfo.with_inst_sync;
  the_gpgpusim->profile_mode = 0;//sysinfo.profile_mode;
  the_gpgpusim->pc_table_selection = 0;//sysinfo.pc_table_selection;
  the_gpgpusim->nvls_cache_size = 0;//sysinfo.nvls_cache_size;

  unsigned icnt_mode;
  char* icnt_filename;
  inct_config m_icnt_config;
  icnt_reg_options(icnt_mode,icnt_filename,m_icnt_config,opp);
  the_gpgpusim->g_the_gpu_config = new gpgpu_sim_config(this);
  the_gpgpusim->g_the_gpu_config->reg_options(
      opp);  // register GPU microrachitecture options

  option_parser_cmdline(opp, sg_argc, sg_argv);  // parse configuration options
  fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
  option_parser_print(opp, stdout);
  // Set the Numeric locale to a standard locale where a decimal point is a
  // "dot" not a "comma" so it does the parsing correctly independent of the
  // system environment variables
  assert(setlocale(LC_NUMERIC, "C"));

  struct kernelinfo
  {
      int switch_reuse;
      int hub_reuse;
      int gpu2switch_bw;
      int switch2gpu_bw;
  };
  kernelinfo kc;
  std::string plan_name = "KernelConfig";
  printf("kernel config with %s\n",plan_name.c_str());
  std::ifstream ofs(plan_name,std::ios::in|std::ios::binary);
  ofs.read((char*)&kc,sizeof(kernelinfo));
  ofs.close();

  the_gpgpusim->g_the_gpu_config->init();
  int n_gpu = the_gpgpusim->n_gpu;
  the_gpgpusim->g_multi_gpu = new gpgpu_sim*[n_gpu];
  the_gpgpusim->g_hub = new nvlink_hub*[n_gpu];
  the_gpgpusim->g_sync = new synchronizer*[n_gpu];
  printf("f_size = %d, latency = %d, s2g_bw = %d, g2s_bw = %d, switch_resue = %d, hub_reuse = %d \n", 1, 50, kc.switch2gpu_bw, kc.gpu2switch_bw, kc.switch_reuse, kc.hub_reuse);
  // the_gpgpusim->g_nvlink = new nvlink_system(the_gpgpusim->sharp_enable, n_gpu, this, 1, 50, kc.gpu2switch_bw, kc.switch2gpu_bw, kc.switch_reuse);
  if(the_gpgpusim->profile_mode != 0)
  {
    the_gpgpusim->g_nvlink = new nvlink_system(the_gpgpusim->sharp_enable, n_gpu, this, 8, 3, 1024, 1024, kc.switch_reuse);
  }
  else
  {
    the_gpgpusim->g_nvlink = new nvlink_system(the_gpgpusim->sharp_enable, n_gpu, this, 4, 3, 1024, 1024, kc.switch_reuse, the_gpgpusim->nvls_cache_size);
  }
  // the_gpgpusim->g_nvlink = new nvlink_system(the_gpgpusim->sharp_enable, n_gpu, this, 1, 50, 1024, 6, kc.switch_reuse);
  // the_gpgpusim->g_nvlink = new nvlink_system(the_gpgpusim->sharp_enable, n_gpu, this, 1, 50, 1, 1, kc.switch_reuse);
  // the_gpgpusim->gpu2switch_cnt = new int[n_gpu];
  // the_gpgpusim->switch2gpu_cnt = new int[n_gpu];
  the_gpgpusim->current_kernel_done = new bool[n_gpu];
  for(int dev_id=0;dev_id<n_gpu;++dev_id)
  {
    the_gpgpusim->g_multi_gpu[dev_id] = new exec_gpgpu_sim(*(the_gpgpusim->g_the_gpu_config), this, dev_id, m_icnt_config);
    the_gpgpusim->current_kernel_done[dev_id] = 0;
    the_gpgpusim->g_hub[dev_id] = new nvlink_hub(the_gpgpusim->g_multi_gpu[dev_id], the_gpgpusim->g_nvlink, dev_id, kc.hub_reuse);
    the_gpgpusim->g_sync[dev_id] = new synchronizer(the_gpgpusim->g_hub[dev_id], dev_id);
    the_gpgpusim->g_multi_gpu[dev_id]->set_sync(the_gpgpusim->g_sync[dev_id]);
    // the_gpgpusim->gpu2switch_cnt[dev_id] = 0;
    // the_gpgpusim->switch2gpu_cnt[dev_id] = 0;
  }

  the_gpgpusim->g_stream_manager = new stream_manager(
    (this), func_sim->g_cuda_launch_blocking);

  the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

  sem_init(&(the_gpgpusim->g_sim_signal_start), 0, 0);
  sem_init(&(the_gpgpusim->g_sim_signal_finish), 0, 0);
  sem_init(&(the_gpgpusim->g_sim_signal_exit), 0, 0);

  return the_gpgpusim->g_multi_gpu;
}

void gpgpu_context::decrement_kernel_latency() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && m_running_kernels[n]->m_kernel_TB_latency)
      m_running_kernels[n]->m_kernel_TB_latency--;
  }
}

unsigned gpgpu_context::finished_kernel() {
  if (m_finished_kernel.empty()) return 0;
  unsigned result = m_finished_kernel.front();
  m_finished_kernel.pop_front();
  return result;
}

void gpgpu_context::set_kernel_done(kernel_info_t *kernel) {
  // bool is_done = true;
  // for(int dev_id=0;dev_id<the_gpgpusim->n_gpu;++dev_id)
  // {
  //   is_done = is_done && the_gpgpusim->current_kernel_done[dev_id];
  // }
  // printf("set kernel done here 1\n");
  // if(is_done)
  // {
    unsigned uid = kernel->get_uid();
    //printf("set kernel done here 2\n");
    m_finished_kernel.push_back(uid);
    std::vector<kernel_info_t *>::iterator k;
    for (k = m_running_kernels.begin(); k != m_running_kernels.end(); k++) {
      if (*k == kernel) {
        // require correction
        kernel->end_cycle = the_gpgpusim->g_multi_gpu[0]->gpu_sim_cycle + the_gpgpusim->g_multi_gpu[0]->gpu_tot_sim_cycle;
        *k = NULL;
        break;
      }
    }
    // for(int dev_id=0;dev_id<the_gpgpusim->n_gpu;++dev_id)
    // {
    //   the_gpgpusim->current_kernel_done[dev_id] = 0;
    // }
    assert(k != m_running_kernels.end());
  //}
}

void gpgpu_context::stop_all_running_kernels() {
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); ++k) {
    if (*k != NULL) {       // If a kernel is active
      set_kernel_done(*k);  // Stop the kernel
      assert(*k == NULL);
    }
  }
}

bool gpgpu_context::can_start_kernel() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done())
      return true;
  }
  return false;
}

// void gpgpu_context::start_sim_thread(int api) {
//   if (the_gpgpusim->g_sim_done) {
//     the_gpgpusim->g_sim_done = false;
//     if (api == 1) {
//       pthread_create(&(the_gpgpusim->g_simulation_thread), NULL,
//                      gpgpu_sim_thread_concurrent, (void *)this);
//     } else {
//       pthread_create(&(the_gpgpusim->g_simulation_thread), NULL,
//                      gpgpu_sim_thread_sequential, (void *)this);
//     }
//   }
// }
void gpgpu_context::start_sim_thread(int api) {
  if (the_gpgpusim->g_sim_done) {
    the_gpgpusim->g_sim_done = false;
    if (api == 1) {
      pthread_create(&(the_gpgpusim->g_simulation_thread), NULL,
                     gpgpu_sim_thread_concurrent, (void *)this);
    } else {
      assert(0);
    }
  }
}

void gpgpu_context::print_simulation_time() {
  time_t current_time, difference, d, h, m, s;
  current_time = time((time_t *)NULL);
  difference = MAX(current_time - the_gpgpusim->g_simulation_starttime, 1);

  d = difference / (3600 * 24);
  h = difference / 3600 - 24 * d;
  m = difference / 60 - 60 * (h + 24 * d);
  s = difference - 60 * (m + 60 * (h + 24 * d));

  fflush(stderr);
  printf(
      "\n\ngpgpu_simulation_time = %u days, %u hrs, %u min, %u sec (%u sec)\n",
      (unsigned)d, (unsigned)h, (unsigned)m, (unsigned)s, (unsigned)difference);
  
  // printf("gpgpu_simulation_rate = %u (inst/sec)\n",
  //        (unsigned)(the_gpgpusim->g_the_gpu->gpu_tot_sim_insn / difference));
  // const unsigned cycles_per_sec =
  //     (unsigned)(the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle / difference);
  // printf("gpgpu_simulation_rate = %u (cycle/sec)\n", cycles_per_sec);
  // printf("gpgpu_silicon_slowdown = %ux\n",
  //        the_gpgpusim->g_multi_gpu[0]->shader_clock() * 1000 / cycles_per_sec);
  fflush(stdout);
}

// int gpgpu_context::gpgpu_opencl_ptx_sim_main_perf(kernel_info_t *grid) {
//   the_gpgpusim->g_the_gpu->launch(grid);
//   sem_post(&(the_gpgpusim->g_sim_signal_start));
//   sem_wait(&(the_gpgpusim->g_sim_signal_finish));
//   return 0;
// }

//! Functional simulation of OpenCL
/*!
 * This function call the CUDA PTX functional simulator
 */
// int cuda_sim::gpgpu_opencl_ptx_sim_main_func(kernel_info_t *grid) {
//   // calling the CUDA PTX simulator, sending the kernel by reference and a flag
//   // set to true, the flag used by the function to distinguish OpenCL calls from
//   // the CUDA simulation calls which it is needed by the called function to not
//   // register the exit the exit of OpenCL kernel as it doesn't register entering
//   // in the first place as the CUDA kernels does
//   gpgpu_cuda_ptx_sim_main_func(*grid, true);
//   return 0;
// }
