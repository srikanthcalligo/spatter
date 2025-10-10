// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/copy_dest_values.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {

    //DeviceZoneScopedN("TEST-FULL");
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t pattern_length =  get_arg_val<uint32_t>(1);    
    uint32_t delta =  get_arg_val<uint32_t>(2);
    uint32_t extra_tile =  get_arg_val<uint32_t>(3);
    uint32_t stride =  get_arg_val<uint32_t>(4);
    uint32_t single_tile_size =  get_arg_val<uint32_t>(5);
    uint32_t count = get_arg_val<uint32_t>(6);
    uint32_t wrap = get_arg_val<uint32_t>(7);
    uint32_t loop_count = single_tile_size / delta;

    //DPRINT << pattern_length << ENDL();
    uint32_t extra_itr = 0;

    if(pattern_length % delta){
        extra_itr = 1;
    }

    //Calculate loop count
    loop_count = loop_count - extra_itr - (stride - 1);

    constexpr auto cb_sparse = tt::CBIndex::c_0;
    constexpr auto cb_pattern = tt::CBIndex::c_1;
    constexpr auto cb_dense_inter = tt::CBIndex::c_2;
    constexpr auto cb_dense = tt::CBIndex::c_3;
    constexpr uint32_t dst_reg = 0;

    unary_op_init_common(cb_dense_inter, cb_dense);
    copy_tile_init(cb_dense_inter);

    DPRINT << "TRISC" << ENDL();

    for(uint32_t tile_id = 0; tile_id < n_tiles; tile_id++)
    {
        acquire_dst();
        
        //Wait for the read kernel to write into CB
        cb_wait_front(cb_sparse, 1);
        cb_wait_front(cb_pattern, 1);
        cb_wait_front(cb_dense_inter, 1);

        volatile uint32_t* pattern_addr_ptr;
        cb_get_tile(cb_pattern, 0, &pattern_addr_ptr);
        pattern_addr_ptr = pattern_addr_ptr + 4; //Need to add 4 because read ptr is off by 1 << 4

        volatile uint32_t* dense_addr_ptr;
        cb_get_tile(cb_dense_inter, 0, &dense_addr_ptr);
        dense_addr_ptr = dense_addr_ptr + 4;

        volatile uint32_t* sparse_addr_ptr;
        cb_get_tile(cb_sparse, 0, &sparse_addr_ptr);
        sparse_addr_ptr = sparse_addr_ptr + 4;

        if((tile_id == (n_tiles - 1)) && (extra_tile != 0)){
            loop_count = count - (tile_id * loop_count);
        }

        for(uint32_t i = 0; i < loop_count; i++){
            uint32_t sparse_index = delta * i;
            #pragma GCC unroll 8
            for(uint32_t j = 0; j < pattern_length; j++){
                //dense_addr_ptr[(j + pattern_length * (i % wrap))] = sparse_addr_ptr[(pattern_addr_ptr[j] + delta * i)];
                dense_addr_ptr[j] = sparse_addr_ptr[(j * stride + sparse_index)];
            }
        }
       
       if(tile_id == (n_tiles - 1)){
            //Write final tile to the CB
            copy_tile(cb_dense_inter, 0, dst_reg);
            
            cb_reserve_back(cb_dense, 1);

            pack_tile(dst_reg, cb_dense);
            
            cb_push_back(cb_dense, 1);
        }
        //Remove tiles from CB for the next set.
        cb_pop_front(cb_sparse, 1); 
        cb_pop_front(cb_pattern, 1);   
        cb_pop_front(cb_dense_inter, 1);
        
        release_dst();
    }
}
}

