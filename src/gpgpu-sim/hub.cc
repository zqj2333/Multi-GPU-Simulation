#include "hub.h"

nvlink_hub::nvlink_hub(gpgpu_sim * gpu, nvlink_system * nvls, int dev_id, int hub_reuse, int response_fifo_sz, int cache_sz, int mshr_sz)
{
    m_gpu = gpu;
    m_nvls = nvls;
    gpu_id = dev_id;
    /*
    response_fifo_size = response_fifo_sz;
    cache_size = cache_sz;
    mshr_size = mshr_sz;
    num_acc = hub_reuse;
    pending_reduction = 0;
    */
}

/*
void nvlink_hub::cycle()
{
    // process ready reduction
    std::list<cache_tag>::iterator itr;
    for(itr=order.begin();itr!=order.end();)
    {
        if(itr->reduction==num_acc)
        {
            bool can_out = m_nvls->can_send(gpu_id, 40);
            if(can_out)
            {
                new_addr_type key;
                key = itr->cacheline_addr;
                writeback(key,32,num_acc);
                order.erase(itr++);
                tag.erase(key);
            }
            else
            {
                break;
            }
        }
        else if(itr->reduction>num_acc)
        {
            printf("invalid reduction count %d\n",itr->reduction);
            assert(0);
        }
        else
        {
            itr++;
        }
    }


    // process response
    while(1)
    {
        //printf("deadlock2\n");
        mem_fetch* head = (mem_fetch*)m_nvls->probe(gpu_id);
        // printf("GPU %d received a packet, source GPU = %d, data size = %d, vaddr = %llx, phyaddr = %llx\n", gpu_id, head->get_src_gpu(), head->size(), head->get_vaddr(), head->get_addr());
        if(head)
        {
            //printf("GPU %d received a packet, source GPU = %d, dst GPU = %d, data size = %d, vaddr = %llx, phyaddr = %llx\n", gpu_id, head->get_src_gpu(), head->get_dst_gpu(), head->size(), head->get_vaddr(), head->get_addr());
            if(head->get_status()==IN_ICNT_TO_SHADER)
            {
                //printf("in hub to sharder\n");
            }
            else if(head->get_status()==IN_ICNT_TO_MEM)
            {
                //printf("in hub to mem\n");
            }
            else
            {
                printf("status error\n");
                assert(0);
            }

            if(head->get_is_write() || head->isatomic())
            {
                bool is_cache_ack = ((head->get_src_gpu() == gpu_id) && (head->get_hub_flag()));
                if(!is_cache_ack)
                {
                    if(response_fifo.size()<response_fifo_size)
                    {
                        response_fifo.push_back(head);
                        m_nvls->pop(gpu_id);
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    m_nvls->pop(gpu_id);
                    delete head;
                    pending_reduction--;
                }
            }
            else
            {
                bool is_cache_miss_response = ((head->get_hub_flag()) && search_mshr(head));
                if(!is_cache_miss_response)
                {
                    // directly push into fifo
                    //printf("not miss response\n");
                    if(response_fifo.size()<response_fifo_size)
                    {
                        response_fifo.push_back(head);
                        m_nvls->pop(gpu_id);
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    // fill mshr
                    //printf("miss response\n");
                    m_nvls->pop(gpu_id);
                    fill(head);
                    delete head;
                }
            }
        }
        else
        {
            break;
        }
    }

    // process ready mshr
    remove_filled_mshr();
}
*/

bool nvlink_hub::can_send(mem_fetch * mf)
{
    // response
    if(mf->get_status() == IN_ICNT_TO_SHADER)
    {
        int packet_size, in_gpu, out_gpu;
        if(mf->get_is_write() || mf->isatomic())
        {
            packet_size = m_nvls->gpgpu_ctx->the_gpgpusim->nvlink_ctrl_size;
            in_gpu = mf->get_dst_gpu();
            out_gpu = mf->get_src_gpu();
        }
        else
        {
            packet_size = m_nvls->gpgpu_ctx->the_gpgpusim->nvlink_ctrl_size + mf->get_data_size();
            in_gpu = mf->get_dst_gpu();
            out_gpu = mf->get_src_gpu();
        }
        if(in_gpu!=gpu_id)
        {
            printf("GPU ID Error in NVLink 1!\n");
            //printf("current GPU = %d, src GPU = %d , packet size = %d, data size = %d, vaddr = %llx, phyaddr = %llx, dst GPU = %d\n", gpu_id, mf->get_src_gpu(), mf->get_ctrl_size(),mf->size(), mf->get_vaddr(), mf->get_addr(), mf->get_dst_gpu());
            assert(0);
        }
        return m_nvls->can_send(gpu_id, packet_size);
    }

    // request
    if(mf->get_status() == IN_ICNT_TO_MEM)
    {
        int packet_size, in_gpu, out_gpu;
        if(mf->is_inst_sync())
        {
            packet_size = m_nvls->gpgpu_ctx->the_gpgpusim->nvlink_ctrl_size;
            in_gpu = mf->get_src_gpu();
            out_gpu = mf->get_dst_gpu();
        }
        else if(mf->get_is_write() || mf->isatomic())
        {
            packet_size = m_nvls->gpgpu_ctx->the_gpgpusim->nvlink_ctrl_size + mf->get_data_size();
            in_gpu = mf->get_src_gpu();
            out_gpu = mf->get_dst_gpu();
        }
        else
        {
            packet_size = m_nvls->gpgpu_ctx->the_gpgpusim->nvlink_ctrl_size;
            in_gpu = mf->get_src_gpu();
            out_gpu = mf->get_dst_gpu();
        }
        if(in_gpu!=gpu_id)
        {
            printf("GPU ID Error in NVLink 1!\n");
            //printf("current GPU = %d, src GPU = %d , packet size = %d, data size = %d, vaddr = %llx, phyaddr = %llx, dst GPU = %d\n", gpu_id, mf->get_src_gpu(), mf->get_ctrl_size(),mf->size(), mf->get_vaddr(), mf->get_addr(), mf->get_dst_gpu());
            assert(0);
        }
        return m_nvls->can_send(gpu_id, packet_size);
    }

    /*
    int access_type = probe_tag(mf);
    if(access_type == 0)
    {
        // hit
        return response_fifo.size()<response_fifo_size;
    }
    else if(access_type == 1)
    {
        // miss
        int packet_size, in_gpu, out_gpu;
        if(mf->get_is_write() || mf->isatomic())
        {
            packet_size = mf->size();
            in_gpu = mf->get_src_gpu();
            out_gpu = mf->get_dst_gpu();
            return (order.size() < cache_size) || can_handle_miss(packet_size);
        }
        else
        {
            packet_size = mf->get_ctrl_size();
            in_gpu = mf->get_src_gpu();
            out_gpu = mf->get_dst_gpu();
            return can_mshr(mf) && can_handle_miss(packet_size);
        }
        // return can_mshr(mf) && can_handle_miss(packet_size);
    }
    else
    {
        // hit-on-miss
        return can_mshr(mf);
    }
    return false;
    */
    printf("wrong send status\n");
    assert(0);
    return false;
}

bool nvlink_hub::can_send_signal()
{
    return m_nvls->can_send(gpu_id, m_nvls->gpgpu_ctx->the_gpgpusim->nvlink_ctrl_size);
}

void nvlink_hub::send(mem_fetch * mf)
{
    // sync
    if(mf->is_sync())
    {
        m_nvls->send(mf->get_src_gpu(), mf->get_dst_gpu(), m_nvls->gpgpu_ctx->the_gpgpusim->nvlink_ctrl_size, mf);
        // printf("send sync\n");
        // assert(0);
        return;
    }

    // response
    if(mf->get_status() == IN_ICNT_TO_SHADER)
    {
        int packet_size, in_gpu, out_gpu;
        if(mf->get_is_write() || mf->isatomic())
        {
            packet_size = m_nvls->gpgpu_ctx->the_gpgpusim->nvlink_ctrl_size;
            in_gpu = mf->get_dst_gpu();
            out_gpu = mf->get_src_gpu();
        }
        else
        {
            // printf("0 request arrives hub\n");
            packet_size = m_nvls->gpgpu_ctx->the_gpgpusim->nvlink_ctrl_size + mf->get_data_size();
            in_gpu = mf->get_dst_gpu();
            out_gpu = mf->get_src_gpu();
        }
        if(in_gpu!=gpu_id)
        {
            printf("GPU ID Error in NVLink 2!\n");
            //printf("current GPU = %d, src GPU = %d , packet size = %d, data size = %d, vaddr = %llx, phyaddr = %llx, dst GPU = %d\n",gpu_id, mf->get_src_gpu(), mf->get_ctrl_size(),mf->size(), mf->get_vaddr(), mf->get_addr(), mf->get_dst_gpu());
            assert(0);
        }
        m_nvls->send(in_gpu, out_gpu, packet_size, mf);
        //printf("current GPU = %d, src GPU = %d , packet size = %d, data size = %d, vaddr = %llx, phyaddr = %llx, dst GPU = %d\n",gpu_id, mf->get_src_gpu(), mf->get_ctrl_size(),mf->size(), mf->get_vaddr(), mf->get_addr(), mf->get_dst_gpu());
        return;
    }
    else if(mf->get_status() == IN_ICNT_TO_MEM)
    {
        int packet_size, in_gpu, out_gpu;
        if(mf->is_inst_sync())
        {
            packet_size = m_nvls->gpgpu_ctx->the_gpgpusim->nvlink_ctrl_size;
            in_gpu = mf->get_src_gpu();
            out_gpu = mf->get_dst_gpu();
        }
        else if(mf->get_is_write() || mf->isatomic())
        {
            packet_size = m_nvls->gpgpu_ctx->the_gpgpusim->nvlink_ctrl_size + mf->get_data_size();
            in_gpu = mf->get_src_gpu();
            out_gpu = mf->get_dst_gpu();
        }
        else
        {
            packet_size = m_nvls->gpgpu_ctx->the_gpgpusim->nvlink_ctrl_size;
            in_gpu = mf->get_src_gpu();
            out_gpu = mf->get_dst_gpu();
        }
        if(in_gpu!=gpu_id)
        {
            printf("GPU ID Error in NVLink 1!\n");
            //printf("current GPU = %d, src GPU = %d , packet size = %d, data size = %d, vaddr = %llx, phyaddr = %llx, dst GPU = %d\n", gpu_id, mf->get_src_gpu(), mf->get_ctrl_size(),mf->size(), mf->get_vaddr(), mf->get_addr(), mf->get_dst_gpu());
            assert(0);
        }
        m_nvls->send(in_gpu, out_gpu, packet_size, mf);
        return;
    }
    else
    {
        printf("status error\n");
        assert(0);
    }

    // request
    /*
    int access_type = probe_tag(mf);
    access_tag(mf);
    if(access_type == 0)
    {
        // hit
        //printf("GPU %d mf %llx hit\n",gpu_id,mf);
        set_reply(mf);
        response_fifo.push_back(mf);
    }
    else if(access_type == 1)
    {
        // miss
        //printf("GPU %d mf %llx miss\n",gpu_id,mf);
        //printf("1 GPU %d hub miss, packet size = %d, data size = %d, vaddr = %llx, phyaddr = %llx, src = %d, dst = %d\n", gpu_id, mf->get_ctrl_size(), mf->size(), mf->get_vaddr(), mf->get_addr(), mf->get_src_gpu(), mf->get_dst_gpu());
        // register_mshr(mf);
        // handle_miss(mf);
        int packet_size, in_gpu, out_gpu;
        if(mf->get_is_write() || mf->isatomic())
        {
            // handle_miss(mf);
            set_reply(mf);
            response_fifo.push_back(mf);
        }
        else
        {
            register_mshr(mf);
            handle_miss(mf);
        }
    }
    else
    {
        // hit-on-miss
        //printf("GPU %d mf %llx hit-on-miss\n",gpu_id,mf);
        register_mshr(mf);
    }
    */
    return;
}

void nvlink_hub::pop(int dev_id)
{
    m_nvls->pop(dev_id);
}

void * nvlink_hub::probe(int dev_id)
{
    return m_nvls->probe(dev_id);
}

/*
int nvlink_hub::probe_tag(mem_fetch * mf)
{
    new_addr_type cacheline_vaddr = mf->get_vaddr();
    std::unordered_map<new_addr_type, std::list<cache_tag>::iterator>::iterator itr;
    itr = tag.find(cacheline_vaddr);
    if(itr!=tag.end())
    {
        if(itr->second->fill)
        {
            return 0;
        }
        else
        {
            return 2;
        }
    }
    return 1;
}

void nvlink_hub::access_tag(mem_fetch * mf)
{
    if(mf->get_is_write() || mf->isatomic())
    {
        new_addr_type cacheline_vaddr = mf->get_vaddr();
        std::unordered_map<new_addr_type, std::list<cache_tag>::iterator>::iterator itr;
        itr = tag.find(cacheline_vaddr);
        if(itr!=tag.end())
        {
            //hit
            cache_tag cur_tag;
            cur_tag = *(itr->second);
            cur_tag.reduction++;
            order.erase(itr->second);
            order.push_back(cur_tag);

            std::list<cache_tag>::iterator tmp = order.end();
            itr->second = --tmp;
        }
        else
        {
            //miss
            if(order.size()>=cache_size)
            {
                new_addr_type key;
                key = order.begin()->cacheline_addr;
                writeback(key,32,order.begin()->reduction);
                order.erase(order.begin());
                tag.erase(key);
            }
            order.push_back(cache_tag(cacheline_vaddr, 1));

            std::list<cache_tag>::iterator tmp = order.end();
            tag.insert(std::pair<new_addr_type, std::list<cache_tag>::iterator>(cacheline_vaddr,--tmp));
        }
    }
    else
    {
        new_addr_type cacheline_vaddr = mf->get_vaddr();
        std::unordered_map<new_addr_type, std::list<cache_tag>::iterator>::iterator itr;
        itr = tag.find(cacheline_vaddr);
        if(itr!=tag.end())
        {
            //hit or hit-on-miss
            cache_tag cur_tag;
            cur_tag = *(itr->second);
            order.erase(itr->second);
            order.push_back(cur_tag);

            std::list<cache_tag>::iterator tmp = order.end();
            itr->second = --tmp;
        }
        else
        {
            //miss
            if(order.size()>=cache_size)
            {
                new_addr_type key;
                key = order.begin()->cacheline_addr;
                order.erase(order.begin());
                tag.erase(key);
            }
            order.push_back(cache_tag(cacheline_vaddr));

            std::list<cache_tag>::iterator tmp = order.end();
            tag.insert(std::pair<new_addr_type, std::list<cache_tag>::iterator>(cacheline_vaddr,--tmp));
        }
    }
}

void nvlink_hub::set_reply(mem_fetch *mf)
{
    mf->set_reply();
    mf->set_status(
              IN_ICNT_TO_SHADER,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
}

bool nvlink_hub::can_mshr(mem_fetch *mf)
{
    return mshr.size()<mshr_size;
}

void nvlink_hub::register_mshr(mem_fetch *mf)
{
    mshr.push_back(mshr_entry(mf, mf->get_vaddr()));
}

bool nvlink_hub::can_handle_miss(int packet_size)
{
    bool can_access_cache = (order.size() < cache_size) || (order.begin()->fill);
    return can_access_cache && m_nvls->can_send(gpu_id, packet_size);
}

void nvlink_hub::handle_miss(mem_fetch *mf)
{
    // std::bitset<128> byte_mask;
    // byte_mask.reset();
    // for (unsigned k = 0; k < 128; ++k)
    //   byte_mask.set(k);
    std::bitset<4> sector_mask;
    sector_mask.reset();
    sector_mask.set(mf->get_vaddr()%128/32);
    // for (unsigned k = 0; k < 4; ++k)
    //   sector_mask.set(k);

    const mem_access_t *ma = new mem_access_t(
        mf->get_access_type(), mf->get_vaddr(), 32,
        mf->is_write(), mf->get_access_warp_mask(),
        mf->get_access_byte_mask(), sector_mask, m_gpu->gpgpu_ctx);

    mem_fetch *new_mf =
        new mem_fetch(*ma, NULL, mf->get_ctrl_size(), mf->get_wid(),
                        mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
                        m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
    new_mf->set_hub_flag();

    new_mf->set_status(IN_ICNT_TO_MEM, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);

    size_t va = (new_mf->get_addr() / 256) * 256;
    size_t offset = new_mf->get_addr() % 256;
    if(va>=0xC0000000 && va<0xf0000000)
    {
        phy_addr the_pa = m_gpu->gpgpu_ctx->the_addrtrans->get_pa(va);
        new_mf->set_src_gpu(gpu_id);
        new_mf->set_dst_gpu(the_pa.gpu_id);
        if(the_pa.gpu_id != gpu_id)
        {
            new_mf->set_phy_addr(the_pa.pa + offset);
        }
        else
        {
            new_mf->set_phy_addr(the_pa.pa + offset);
        }
        new_mf->set_vaddr(va + offset);
        m_nvls->send(gpu_id, the_pa.gpu_id, new_mf->get_ctrl_size(), new_mf);
        //printf("2 GPU %d hub miss, packet size = %d, data size = %d, vaddr = %llx, phyaddr = %llx, home GPU = %d\n",gpu_id,new_mf->get_ctrl_size(),new_mf->size(), new_mf->get_vaddr(), new_mf->get_addr(), the_pa.gpu_id);
        //assert(0);
    }
    else
    {
        printf("invalid access\n");
        assert(0);
    }
}

void nvlink_hub::fill(mem_fetch *mf)
{
    // fill cache tag array
    std::unordered_map<new_addr_type, std::list<cache_tag>::iterator>::iterator itr;
    // printf("fill addr %llx\n",mf->get_vaddr());
    itr = tag.find(mf->get_vaddr());
    if(itr!=tag.end())
    {
        itr->second->fill = true;
    }
    else
    {
        printf("fill error\n");
        fflush(stdout);
        assert(0);
    }

    // fill mshr
    new_addr_type cacheline_addr = mf->get_vaddr();
    std::list<mshr_entry>::iterator mshr_itr;
    for(mshr_itr=mshr.begin();mshr_itr!=mshr.end();mshr_itr++)
    {
        if(mshr_itr->cacheline_addr == cacheline_addr)
        {
            mshr_itr->fill = true;
        }
    }
}

bool nvlink_hub::search_mshr(mem_fetch *mf)
{
    new_addr_type cacheline_addr = mf->get_vaddr();
    //printf("GPU %d search mshr with %llx\n", gpu_id, cacheline_addr);
    std::list<mshr_entry>::iterator itr;
    for(itr=mshr.begin();itr!=mshr.end();itr++)
    {
        if(itr->cacheline_addr == cacheline_addr)
        {
            return true;
        }
    }
    return false;
}

void nvlink_hub::remove_filled_mshr()
{
    std::list<mshr_entry>::iterator itr;
    for(itr=mshr.begin();itr!=mshr.end();)
    {
        if(response_fifo.size()>=response_fifo_size)
        {
            break;
        }

        if(itr->fill)
        {
            //printf("remove_filled_mshr\n");
            set_reply(itr->mf);
            response_fifo.push_back(itr->mf);
            mshr.erase(itr++);
        }
        else
        {
            itr++;
        }
    }
    //printf("remove_filled_mshr finish\n");
}

void nvlink_hub::writeback(new_addr_type vaddr, int data_size, int reduction)
{
    // printf("writeback\n");
    pending_reduction++;

    std::bitset<4> sector_mask;
    sector_mask.reset();
    sector_mask.set(vaddr%128/32);

    std::bitset<128> byte_sector_mask;
    byte_sector_mask.reset();
    for (unsigned k = vaddr%128; k < vaddr%128+32; ++k)
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
                        0, 0, m_gpu->getMemoryConfig(),
                        m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);
    new_mf->set_hub_flag();
    new_mf->set_status(IN_ICNT_TO_MEM, m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);

    size_t va = (new_mf->get_addr() / 256) * 256;
    size_t offset = new_mf->get_addr() % 256;
    if(va>=0xC0000000 && va<0xf0000000)
    {
        phy_addr the_pa = m_gpu->gpgpu_ctx->the_addrtrans->get_pa(va);
        new_mf->set_src_gpu(gpu_id);
        new_mf->set_dst_gpu(the_pa.gpu_id);
        if(the_pa.gpu_id != gpu_id)
        {
            new_mf->set_phy_addr(the_pa.pa + offset);
        }
        else
        {
            new_mf->set_phy_addr(the_pa.pa + offset);
        }
        new_mf->set_vaddr(va + offset);
        new_mf->set_accumulation(num_acc);
        m_nvls->send(gpu_id, the_pa.gpu_id, new_mf->size(), new_mf);
        //printf("2 GPU %d hub miss, packet size = %d, data size = %d, vaddr = %llx, phyaddr = %llx, home GPU = %d\n",gpu_id,new_mf->get_ctrl_size(),new_mf->size(), new_mf->get_vaddr(), new_mf->get_addr(), the_pa.gpu_id);
        //assert(0);
    }
    else
    {
        printf("invalid access\n");
        assert(0);
    }
}*/