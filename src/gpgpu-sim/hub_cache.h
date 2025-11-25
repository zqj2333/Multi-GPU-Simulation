#ifndef HUB_CACHE_H
#define HUB_CACHE_H

#include "mem_fetch.h"
#include "gpu-sim.h"
#include "../abstract_hardware_model.h"
#include <list>

struct fused_request
{
    int fuse;
    int fill;
    int timeout;
    int m_identifier;
    mem_fetch** mf_list;
    fused_request(int n_gpu)
    {
        fuse = 0;
        fill = 0;
        timeout = 0;
        mf_list = new mem_fetch*[n_gpu];
        for(int i=0; i<n_gpu; ++i)
        {
            mf_list[i] = NULL;
        }
    }
};

struct cache_tag
{
    new_addr_type cacheline_addr;
    int fill;
    int reduction;
    int timeout;
    std::list <fused_request*> * fused_request_list;
    
    cache_tag(new_addr_type addr = 0)
    {
        cacheline_addr = addr;
        fill = 0;
        reduction = 0;
        fused_request_list = NULL;
    }
    cache_tag(new_addr_type addr, int red)
    {
        cacheline_addr = addr;
        fill = 1;
        reduction = red;
        fused_request_list = NULL;
    }
    cache_tag(new_addr_type addr, int m_fuse, int n_gpu)
    {
        cacheline_addr = addr;
        fill = 0;
        reduction = 0;
        fused_request_list = new std::list <fused_request*>;
    }
};

struct mshr_entry
{
    mem_fetch * mf;
    new_addr_type cacheline_addr;
    bool fill;
    mshr_entry(mem_fetch * m_mf, new_addr_type vaddr)
    {
        cacheline_addr = vaddr;
        mf = m_mf;
        fill = 0;
    }
};

#endif