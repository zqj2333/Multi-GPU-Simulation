#ifndef SYNCHRONIZER_H
#define SYNCHRONIZER_H

#include <stdio.h>
#include <assert.h>
#include "mem_fetch.h"
#include "icnt_wrapper.h"
#include "hub.h"
#include "../abstract_hardware_model.h"
#include "../../libcuda/gpgpu_context.h"
#include <list>
#include <unordered_map>

class synchronizer
{
    public:
    synchronizer(nvlink_hub* nvhub, int dev_id, int size = 128);
    int get_status(int identifier); // 0: not found, 1: ready, 2: registered but not ready
    void register_sync(int identifier);
    void remove_sync(int identifier);
    void cycle();
    // bool can_send_signal();
    void send_signal(int identifier);

    nvlink_hub* m_hub;
    int gpu_id;
    int table_size;
    std::unordered_map<int, int> sync_table; // 0: not send, 1: send but not ready, 2: ready
};

class inst_synchronizer
{
    public:
    inst_synchronizer(gpgpu_sim* gpu, myinterconnect * icnt, int dev_id, int cluster_id, int size = 128);
    int get_status(int identifier); // 0: not found, 1: ready, 2: registered but not ready
    void register_sync(int identifier);
    void remove_sync(int identifier);
    void cycle();
    void fill(int identifier);
    // bool can_send_signal();
    void send_signal(int identifier);

    gpgpu_sim * m_gpu;
    myinterconnect * m_icnt;
    int m_gpu_id;
    int m_cluster_id;
    int table_size;
    std::unordered_map<int, int> sync_table; // 0: not send, 1: send but not ready, 2: ready
};


#endif