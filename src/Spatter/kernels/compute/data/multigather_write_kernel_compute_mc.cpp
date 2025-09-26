#include "dataflow_api.h"

void kernel_main(){
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t core_id = get_arg_val<uint32_t>(1);
    uint32_t num_cores = get_arg_val<uint32_t>(2);
    
    //DeviceZoneScopedN("TEST-FULL");

    if(core_id == (num_cores - 1)){
        constexpr uint32_t cb_id_out0 = tt::CBIndex::c_4;
        uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);

        const InterleavedAddrGenFast<true> dest = {
            .bank_base_address = dst_addr,
            .page_size = ublock_size_bytes,
            .data_format = DataFormat::UInt32,
        };
        //Writing final tile of last thread to DRAM
        cb_wait_front(cb_id_out0, 1);
        uint32_t cb_out0_addr = get_read_ptr(cb_id_out0);
        
        noc_async_write_tile(0, dest, cb_out0_addr); // Writing only one tile 
        
        //noc_async_write_tile(core_id, dest, cb_out0_addr);
        
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    }
}