#ifndef NVLINK_H
#define NVLINK_H

#include <stdio.h>
#include <assert.h>
#include <list>
#include "mem_fetch.h"
#include "hub_cache.h"
#include "../abstract_hardware_model.h"
#include "../../libcuda/gpgpu_context.h"

struct nvlink_packet
{
    nvlink_packet(int i_gpu = 0, int o_gpu = 0, void * mf = NULL, int pkt_size = 0)
    {
        in_gpu = i_gpu;
        out_gpu = o_gpu;
        data = mf;
        packet_size = pkt_size;
    }
    int in_gpu;
    int out_gpu;
    void * data;
    int packet_size;
};

struct nv_flit
{
    nv_flit(nvlink_packet* pkt = NULL, int idx = 0)
    {
        host_pkt = pkt;
        index = idx;
    }
    nvlink_packet* host_pkt;
    int index;
};

class nvlink
{
    public:
    nvlink(bool g2s, int gpu_id, int bw, int latency, int flit_size = 8, int i_buf_size = 1024, int o_buf_size = 1024); // 32768
    nvlink_packet*probe();
    void pop();
    void push(nvlink_packet*pkt);
    bool can_push(int packet_size);
    void direct_forward();
    void sharp_forward();
    void cycle();
    bool busy();
    int bandwidth;
    int flit_size;
    int in_buffer_size;
    int out_buffer_size;
    int m_latency;
    int pending_flit;
    std::list <nv_flit> ingress_port_buffer;
    std::list <nv_flit> egress_port_buffer;
    nv_flit* latency_queue;
    bool g2s;
    int gpu_id;
    private:
    bool can_move(int i);
    void move(int i);
};

struct sync_entry
{
    int ready;
    int count;
    mem_fetch** mf_list;
    sync_entry(int n_gpu)
    {
        ready = false;
        count = 0;
        mf_list = new mem_fetch*[n_gpu];
        for(int i=0;i<n_gpu;++i)
        {
            mf_list[i] = NULL;
        }
    }
};

class nvlink_system
{
    public:
    
    nvlink_system(bool sharp_en, int n_gpu, gpgpu_context *ctx, int f_size = 8, int latency = 50, int g2s_bw = 1024, int s2g_bw = 1024, int switch_reuse = 12, int cache_sz = 1073741824, int mshr_sz = 1073741824); // 2048
    void cycle();
    bool can_send(int dev_id, int packet_size);
    void send(int in_gpu, int out_gpu, int packet_size, void * mf);
    void pop(int dev_id);
    void * probe(int dev_id);
    void print_stats();

    // sharp related
    void set_reply(mem_fetch *mf);
    void writeback(new_addr_type vaddr, int data_size, int reduction);
    void handle_fused_request(fused_request *fr);
    bool busy();
    void sharp_forward();
    void direct_forward();
    
    nvlink ** gpu2switch_link;
    nvlink ** switch2gpu_link;
    int gpu2switch_bw;
    int switch2gpu_bw;
    int m_latency;
    int n_gpu;
    int RR_cnt;
    int RR_cnt_port;
    int flit_size;

    // stats
    int*gpu2switch_cnt;
    int*switch2gpu_cnt;
    int sim_cycle;
    int start_cycle;

    // sharp related
    // cache
    bool sharp_enable;
    int cache_size; // per port
    
    int * port_current_size;
    std::list<cache_tag>** port_order;
    std::unordered_map<new_addr_type, std::list<cache_tag>::iterator>** port_tag;
    std::list<fused_request*>** port_fr_order;
    std::unordered_map<fused_request*, std::list<fused_request*>::iterator>** port_fr_tag;

    int * this_current_size;
    std::list<cache_tag>* this_port_order;
    std::unordered_map<new_addr_type, std::list<cache_tag>::iterator>* this_port_tag;
    std::list<fused_request*>* this_port_fr_order;
    std::unordered_map<fused_request*, std::list<fused_request*>::iterator>* this_port_fr_tag;

    // reduction related
    int num_acc;
    int pending_reduction;
    int wait_reduction;
    int timeout_init;

    // sync related
    std::unordered_map<int, sync_entry> sync_table;
    int sync_cnt;

    gpgpu_context *gpgpu_ctx;
};





#endif