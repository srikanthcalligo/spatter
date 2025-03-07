// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
//#include "tools/profiler/kernel_profiler.hpp"

#include "debug/dprint.h"
#include "debug/dprint_tensix.h"
namespace NAMESPACE {
void MAIN {

    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_in2 = tt::CBIndex::c_2;
    constexpr auto cb_out0 = tt::CBIndex::c_3;
    
    constexpr uint32_t dst_reg = 0;

    binary_op_init_common(cb_in0, cb_in2, cb_out0);
    
    //add_tiles_init(cb_in0, cb_in2);
    copy_tile_init(cb_in2);
    for(uint32_t i = 0; i < n_tiles; i++)
    {
        //get_tile_info(i);
        //DeviceZoneScopedN("TEST-FULL");
        acquire_dst();
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in2, 1);

        //add_tiles(cb_in0, cb_in2, 0, 0, dst_reg);

        copy_tile(cb_in2, 0, 0);

        //dprint_tensix_dest_reg(dst_reg);


        cb_pop_front(cb_in0, 1); 
        cb_pop_front(cb_in2, 1);   
    
        cb_reserve_back(cb_out0, 1);

        pack_tile(dst_reg, cb_out0);
    
        cb_push_back(cb_out0, 1);
       
        release_dst();
    }
}
}