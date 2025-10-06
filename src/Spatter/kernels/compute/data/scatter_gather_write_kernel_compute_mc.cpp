#include "dataflow_api.h"

void kernel_main(){
    
    //DeviceZoneScopedN("TEST-FULL");

    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t n_tiles_scatter =  get_arg_val<uint32_t>(1);
    uint32_t num_tiles_written  = get_arg_val<uint32_t>(2);
    uint32_t num_output_tiles_per_core  = get_arg_val<uint32_t>(3);
    
    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_4;
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);

    const InterleavedAddrGenFast<true> dest = {
        .bank_base_address = dst_addr,
        .page_size = ublock_size_bytes,
        .data_format = DataFormat::UInt32,
    };
    //writing tiles to the DRAM.
    for(uint32_t tile_id = num_tiles_written; tile_id < (num_tiles_written+num_output_tiles_per_core); tile_id++) {
        cb_wait_front(cb_id_out0, 1);
        uint32_t cb_out0_addr = get_read_ptr(cb_id_out0); //read sparse_spatter 
        noc_async_write_tile(tile_id, dest, cb_out0_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, 1);
    }
}