#include "synchronizer.h"

synchronizer::synchronizer(nvlink_hub* nvhub, int dev_id, int size)
{
    m_hub = nvhub;
    table_size = size;
    gpu_id = dev_id;
}

int synchronizer::get_status(int identifier)
{
    std::unordered_map<int, int>::iterator itr;
    itr = sync_table.find(identifier);
    if(itr!=sync_table.end())
    {
        if(itr->second==2)
        {
            return 1;
        }
        else
        {
            return 2;
        }
    }
    return 0;
}

void synchronizer::register_sync(int identifier)
{
    std::unordered_map<int, int>::iterator itr;
    itr = sync_table.find(identifier);
    if(itr!=sync_table.end())
    {
        printf("sync conflict\n");
        assert(0);
    }
    sync_table.insert(std::pair<int, int>(identifier, 0));
}

void synchronizer::remove_sync(int identifier)
{
    sync_table.erase(identifier);
}

void synchronizer::cycle()
{
    // send sync signal
    std::unordered_map<int, int>::iterator itr = sync_table.begin();
    for(;itr!=sync_table.end();itr++)
    {
        if(itr->second==0)
        {
            if(m_hub->can_send_signal())
            {
                send_signal(itr->first);
                itr->second = 1;
            }
            else
            {
                break;
            }
        }
    }

    // recv ack
    while(1)
    {
        mem_fetch *mf = (mem_fetch*)(m_hub->probe(gpu_id));
        if(mf)
        {
            if(mf->is_sync())
            {
                m_hub->pop(gpu_id);
                if(mf->get_status() == IN_ICNT_TO_SHADER)
                {
                    int identifier = mf->get_identifier();
                    // std::unordered_map<int, int>::iterator itr;
                    itr = sync_table.find(identifier);
                    if(itr!=sync_table.end())
                    {
                        if(itr->second!=1)
                        {
                            printf("wrong sync status\n");
                            assert(0);
                        }
                        itr->second = 2;
                        delete mf;
                    }
                    else
                    {
                        printf("identifier lost\n");
                        assert(0);
                    }
                }
                else
                {
                    printf("wrong sync signal\n");
                    assert(0);
                }
            }
            else
            {
                break;
            }
        }
        else
        {
            break;
        }
    }
}

void synchronizer::send_signal(int identifier)
{

    new_addr_type vaddr = 0;

    std::bitset<4> sector_mask;
    sector_mask.reset();
    sector_mask.set(0);

    std::bitset<128> byte_sector_mask;
    byte_sector_mask.reset();
    for (unsigned k = 0; k < 32; ++k)
      byte_sector_mask.set(k);

    std::bitset<32> warp_mask;
    warp_mask.reset();
    for (unsigned k = 0; k < 32; ++k)
      warp_mask.set(k);

    const mem_access_t *ma = new mem_access_t(
        GLOBAL_ACC_W, vaddr, 32,
        true, warp_mask,
        byte_sector_mask, sector_mask, m_hub->m_gpu->gpgpu_ctx);

    mem_fetch *new_mf =
        new mem_fetch(*ma, NULL, 8, 0,
                        0, 0, m_hub->m_gpu->getMemoryConfig(), 0);

    new_mf->set_sync();
    new_mf->set_identifier(identifier);
    new_mf->set_status(IN_ICNT_TO_MEM, 0);

    new_mf->set_src_gpu(gpu_id);
    new_mf->set_dst_gpu(-1);

    m_hub->send(new_mf);
}




inst_synchronizer::inst_synchronizer(gpgpu_sim* gpu, myinterconnect * icnt, int dev_id, int cluster_id, int size)
{
    m_gpu = gpu;
    m_icnt = icnt;
    m_gpu_id = dev_id;
    m_cluster_id = cluster_id;
    table_size = size;
}

int inst_synchronizer::get_status(int identifier)
{
    std::unordered_map<int, int>::iterator itr;
    itr = sync_table.find(identifier);
    if(itr!=sync_table.end())
    {
        if(itr->second==2)
        {
            return 1;
        }
        else
        {
            return 2;
        }
    }
    return 0;
}

void inst_synchronizer::register_sync(int identifier)
{
    std::unordered_map<int, int>::iterator itr;
    itr = sync_table.find(identifier);
    if(itr!=sync_table.end())
    {
        printf("sync conflict\n");
        assert(0);
    }
    sync_table.insert(std::pair<int, int>(identifier, 0));
}

void inst_synchronizer::remove_sync(int identifier)
{
    sync_table.erase(identifier);
}

void inst_synchronizer::cycle()
{
    // send sync signal
    std::unordered_map<int, int>::iterator itr = sync_table.begin();
    for(;itr!=sync_table.end();itr++)
    {
        if(itr->second==0)
        {
            if(m_icnt->icnt_has_buffer(m_cluster_id, 8))
            {
                send_signal(itr->first);
                itr->second = 1;
            }
            else
            {
                break;
            }
        }
    }
}

void inst_synchronizer::fill(int identifier)
{
    std::unordered_map<int, int>::iterator itr = sync_table.begin();
    itr = sync_table.find(identifier);
    if(itr!=sync_table.end())
    {
        if(itr->second!=1)
        {
            printf("wrong sync status\n");
            assert(0);
        }
        itr->second = 2;
    }
    else
    {
        printf("identifier lost\n");
        assert(0);
    }
}

void inst_synchronizer::send_signal(int identifier)
{

    new_addr_type vaddr = 0;

    std::bitset<4> sector_mask;
    sector_mask.reset();
    sector_mask.set(0);

    std::bitset<128> byte_sector_mask;
    byte_sector_mask.reset();
    for (unsigned k = 0; k < 32; ++k)
      byte_sector_mask.set(k);

    std::bitset<32> warp_mask;
    warp_mask.reset();
    for (unsigned k = 0; k < 32; ++k)
      warp_mask.set(k);

    const mem_access_t *ma = new mem_access_t(
        GLOBAL_ACC_W, vaddr, 32,
        true, warp_mask,
        byte_sector_mask, sector_mask, m_gpu->gpgpu_ctx);

    mem_fetch *new_mf =
        new mem_fetch(*ma, NULL, 8, 0,
                        0, m_cluster_id, m_gpu->getMemoryConfig(), 0);

    new_mf->set_inst_sync();
    new_mf->set_identifier(identifier);
    new_mf->set_status(IN_ICNT_TO_MEM, 0);

    new_mf->set_src_gpu(m_gpu_id);
    new_mf->set_dst_gpu(-1);

    // m_hub->send(new_mf);
    m_icnt->icnt_push(m_cluster_id, 0, (void *)new_mf, 8, -1);
}






