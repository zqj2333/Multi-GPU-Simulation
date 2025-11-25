// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda
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

#ifndef ICNT_WRAPPER_H
#define ICNT_WRAPPER_H

#include <stdio.h>
#include "../option_parser.h"
#include "local_interconnect.h"

class myinterconnect
{
  public:
    myinterconnect() {
    g_localicnt_interface = NULL;  // set by option parser
    g_network_mode = 1;
    in_RR_cnt = 0;
    out_RR_cnt = 0;
    pop_RR = 0;
    }
    int gpu_id;
    int g_shader;
    int g_mem;
    int g_io;

    int in_RR_cnt;
    int out_RR_cnt;
    
    int pop_RR;

    inct_config g_inct_config;
    unsigned g_network_mode;
    char* g_network_config_filename;
    class LocalInterconnect* g_localicnt_interface;
    //void icnt_wrapper_init();
    void icnt_create(unsigned int n_shader, unsigned int n_mem, unsigned int n_io) 
    {
        g_localicnt_interface->CreateInterconnect(n_shader + n_io, n_mem + n_io);
        g_shader = n_shader;
        g_mem = n_mem;
        g_io = n_io;
    }

    void icnt_init() { g_localicnt_interface->Init(); }
    bool icnt_has_buffer(unsigned input, unsigned int size) {
      if(input >= g_shader) input = input + g_io;
      return icnt_has_buffer_internal(input, size);
    }
    bool icnt_has_buffer_internal(unsigned input, unsigned int size) {
      return g_localicnt_interface->HasBuffer(input, size);
    }

    bool icnt_io_has_buffer(unsigned int size, unsigned output)
    {
      // printf("no support for remote access\n");
      //     assert(0);
      if(output >= g_shader) output = output + g_io;
      for(int itr = 0; itr < g_io; ++itr)
      {
        if(output < g_shader + g_io) // remote gpu is like a mem
        {
          if(icnt_has_buffer_internal(g_shader + g_mem + g_io + itr, size))
          {
            return true;
          }
        }
        else // remote gpu is like a SM
        {
          if(icnt_has_buffer_internal(g_shader + itr, size))
          {
            return true;
          }
        }
      }
      return false;
    }

    void icnt_push(unsigned input, unsigned output, void* data,
                                   unsigned int size, int dst_gpu) {
      if(input >= g_shader) input = input + g_io;
      if(output >= g_shader) output = output + g_io;
      if(dst_gpu != gpu_id)
      {
        // printf("no support for remote access\n");
        // assert(0);
        if(input < g_shader + g_io)
        {
          //printf("send %d message to hub: %d to %d port (g_shader=%d,g_mem=%d,g_io=%d)\n", size, input, g_shader + g_mem + g_io + out_RR_cnt,g_shader,g_mem,g_io);
          //fflush(stdout);
          g_localicnt_interface->Push(input, g_shader + g_mem + g_io + out_RR_cnt, data, size);
        }
        else
        {
          //printf("send %d message to hub: %d to %d port (g_shader=%d,g_mem=%d,g_io=%d)\n", size, input, g_shader + out_RR_cnt,g_shader,g_mem,g_io);
          //fflush(stdout);
          g_localicnt_interface->Push(input, g_shader + out_RR_cnt, data, size);
        }
        out_RR_cnt = (out_RR_cnt + 1) % g_io;
      }
      else
      {
        g_localicnt_interface->Push(input, output, data, size);
      }
    }

    void icnt_push_io(unsigned output, void* data, unsigned int size) {
      // printf("no support for remote access\n");
      // assert(0);
      if(output >= g_shader) output = output + g_io;
      for(int itr = 0; itr < g_io; ++itr)
      {
        if(output < g_shader + g_io) // remote gpu is like a mem
        {
          if(icnt_has_buffer_internal(g_shader + g_mem + g_io + in_RR_cnt, size))
          {
            g_localicnt_interface->Push(g_shader + g_mem + g_io + in_RR_cnt, output, data, size);
            in_RR_cnt = (in_RR_cnt + 1) % g_io;
            break;
          }
        }
        else // remote gpu is like a SM
        {
          if(icnt_has_buffer_internal(g_shader + in_RR_cnt, size))
          {
            g_localicnt_interface->Push(g_shader + in_RR_cnt, output, data, size);
            in_RR_cnt = (in_RR_cnt + 1) % g_io;
            break;
          }
        }
        in_RR_cnt = (in_RR_cnt + 1) % g_io;
      }
    }

    void* icnt_pop(unsigned output) {
      if(output >= g_shader) output = output + g_io;
      return g_localicnt_interface->Pop(output);
    }

    int translate_io(int io_id)
    {
      if(io_id >= g_shader + g_io)
      {
        return io_id + g_mem;
      }
      return io_id;
    }

    void* icnt_pop_io() {
      void * data = NULL;
      for(int i=0; i<g_io*2; ++i)
      {
        int tmp = g_shader + pop_RR;
        int io_id = translate_io(tmp);
        if(g_localicnt_interface->Can_Pop(io_id))
        {
          data = g_localicnt_interface->Pop(io_id);
          pop_RR = (pop_RR + 1) % (2 * g_io);
          break;
        }
        pop_RR = (pop_RR + 1) % (2 * g_io);
      }
      return data;
    }

    void* icnt_probe_io() {
      void * data = NULL;
      for(int i=0; i<g_io*2; ++i)
      {
        int tmp = g_shader + pop_RR;
        int io_id = translate_io(tmp);
        // printf("search port %d\n",io_id);
        // fflush(stdout);
        if(g_localicnt_interface->Can_Pop(io_id))
        {
          data = g_localicnt_interface->Probe(io_id);
          break;
        }
        pop_RR = (pop_RR + 1) % (2 * g_io);
      }
      return data;
    }

    void icnt_transfer() { g_localicnt_interface->Advance(); }

    bool icnt_busy() { return g_localicnt_interface->Busy(); }

    void icnt_display_stats() {
      g_localicnt_interface->DisplayStats();
    }

    void icnt_display_overall_stats() {
      g_localicnt_interface->DisplayOverallStats();
    }

    void icnt_display_state(FILE* fp) {
      g_localicnt_interface->DisplayState(fp);
    }

    unsigned icnt_get_flit_size() {
      return g_localicnt_interface->GetFlitSize();
    }
};








#endif
