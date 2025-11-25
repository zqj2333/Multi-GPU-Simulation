#ifndef __address_translation_h__
#define __address_translation_h__

#include "tr1_hash_map.h"
#define atmap tr1_hash_map

class gpgpu_context;

struct phy_addr
{
  int gpu_id;
  size_t pa;
};

class addr_trans {
  public:
    addr_trans(gpgpu_context*ctx)
    {
      addr_ctx = ctx;
    }
    void map_addr(size_t va, int gpu_id, size_t pa);
    phy_addr get_pa(size_t va);
    gpgpu_context* addr_ctx;
    atmap <size_t,phy_addr> page_table;
};

#endif