#ifndef HUB_H
#define HUB_H

#include <stdio.h>
#include <assert.h>
#include "nvlink.h"
#include "mem_fetch.h"
#include "gpu-sim.h"
#include <list>
#include <unordered_map>
#include "../abstract_hardware_model.h"
#include "../../libcuda/gpgpu_context.h"
// #include "hub_cache.h"


// struct cache_tag
// {
//     new_addr_type cacheline_addr;
//     bool fill;
//     int reduction;
//     cache_tag(new_addr_type addr = 0)
//     {
//         cacheline_addr = addr;
//         fill = 0;
//         reduction = 0;
//     }
//     cache_tag(new_addr_type addr, int red)
//     {
//         cacheline_addr = addr;
//         fill = 1;
//         reduction = red;
//     }
// };

// struct mshr_entry
// {
//     mem_fetch * mf;
//     new_addr_type cacheline_addr;
//     bool fill;
//     mshr_entry(mem_fetch * m_mf, new_addr_type vaddr)
//     {
//         cacheline_addr = vaddr;
//         mf = m_mf;
//         fill = 0;
//     }
// };

class nvlink_hub
{
    public:
    
    nvlink_hub(gpgpu_sim * gpu, nvlink_system * nvls, int dev_id, int hub_reuse = 4, int response_fifo_sz = 1073741824, int cache_sz = 1073741824, int mshr_sz = 1073741824);
    void cycle() {}
    bool can_send(mem_fetch * mf);
    bool can_send_signal();
    void send(mem_fetch * mf);
    void pop(int dev_id);
    void * probe(int dev_id);

    /*
    int probe_tag(mem_fetch * mf); // 0: hit, 1: miss, 2: hit-on-miss
    void access_tag(mem_fetch * mf); 
    void set_reply(mem_fetch *mf);
    bool can_mshr(mem_fetch *mf);
    void register_mshr(mem_fetch *mf);
    bool can_handle_miss(int packet_size);
    void handle_miss(mem_fetch *mf);
    void fill(mem_fetch *mf);
    bool search_mshr(mem_fetch *mf);
    void remove_filled_mshr();
    void writeback(new_addr_type vaddr, int data_size, int reduction);
    */

    bool busy() { return false; } //pending_reduction > 0; }
    
    public:
    gpgpu_sim * m_gpu;
    nvlink_system* m_nvls;
    int gpu_id;

    /*
    // hub2icnt fifo
    std::list<mem_fetch*> response_fifo;
    int response_fifo_size;

    // cache
    int cache_size;
    std::list<cache_tag>order;
    std::unordered_map<new_addr_type, std::list<cache_tag>::iterator> tag;

    // mshr
    int mshr_size;
    std::list<mshr_entry>mshr;

    // reduction related
    int num_acc;
    int pending_reduction;
    */
};

#endif