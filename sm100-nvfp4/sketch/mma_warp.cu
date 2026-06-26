#include <cstdint>

constexpr int CTA_GROUP = 1;



__device__ inline
uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{\n\t"
    ".reg .pred %%px;\n\t"
    "elect.sync _|%%px, %1;\n\t"
    "@%%px mov.s32 %0, 1;\n\t"
    "}"
    : "+r"(pred)
    : "r"(0xFFFFFFFF) // all 32 lanes in current warp
  );
  return pred;
}


__device__ inline
void mma_nvfp4_cta1_m128n128k64(uint32_t t_mem, 
                                uint64_t A_desc, 
                                uint64_t B_desc,
                                uint32_t I_desc,
                                uint32_t sA_tmem,
                                uint32_t sB_tmem,
                                uint32_t mma_bar_addr,
                                int enable_input_d)
{
    asm volatile("{\n\t" 
    ".reg .pred p; \n\t"
    "setp.ne.b32 p, %6, 0;\n\t" //set to true if not equal to 0
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5]; \n\t"
    "}"
    :: "r"(taddr), "l"(A_desc), "l"(B_desc), "r"(I_desc), "r"(sA_tmem), "r"(sB_tmem), "r"(enable_input_d), "r"(mma_bar_addr)
    );
}


__device__ inline
void mbarrier_wait(uint32_t mbar, uint32_t state)
{
    asm volatile("{\n\t"
    ".reg .pred P1;\n\t"
    "TRY_WAIT_LOOP:\n\t"
    "mbarrier.try_wait.parity.phase_type::primary.acquire.cta.shared::cta.b64 P1, [%0], %1; \n\t"
    "@!P1 bra TRY_WAIT_LOOP:\n\t"
    "}"
    :: "r"(mbar), "r"(state));
}

__device__ inline
void tcgen05_cp(uint32_t t_addr, uint64_t a_desc)
{
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
    :: "r"(t_addr), "l"(a_desc)
    );
}


template <int Q_STAGES>
__device__ __forceinline__
void mma_warp_loop(const int producer_q_mbar, // TODO redesign how all this is passed
                   const int producer_kv_mbar, 
                   //const int release_q_mbar, 
                   const int release_kv_mbar,
                   const int s_full_mbar,
                   const int p_o_read_mbar,
                   const int o_full)
{
    const int phase = 0;
    // assume we have elect sync outside of mma_warp_loop function
    
    for (int stage = 0; stage < Q_STAGES; stage++) {
        mbarrier_wait(producer_q_mbar + stage * 8, phase); // load q and s_q
    } 

    // lets say we have q_smem, sf_q_smem

    for (int iter = 0; iter < kv_iters; iter++) {
        const int k_slot_id =  2 * (iter % NUM_SLOTS);
        const int v_slot_id = k_slot_id + 1;

        mbarrier_wait(producer_kv_mbar + k_slot_id * 8, phase); // load k and s_q
        asm volatile("tcgen05.fence.after_thread_sync;");

        for (int stage = 0; stage < Q_STAGES; stage++) {

            // How to organise copy from smem to tensor for scales.
            const uint32_t k_smem = smem + stage * (k_size + v_size );
            const uint32_t sf_k_smem = k_smem + k_size;
            const uint32_t v_smem = s_k_smem + sf_k_size;
            const uint32_t sf_v_smem = v_smem + v_size;

            mma_nvfp4_cta1_m128n128k64()
            const int s_addr = s_full_bar + 8 * stage;
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]"
                :: "r"(s_addr)
                : "memory"
            );
        }

        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]"
            :: "r"(producer_kv_mbar + k_slot_id * 8)
            : "memory"
        );

        mbarrier_wait(producer_kv_mbar + v_slot_id * 8, phase); // load v and s_v

        for (int stage = 0; stage < Q_STAGES; stage++) {
            const int p_o_adr = p_o_read_mbar + 8 * stage ;
            mbarrier_wait(p_o_addr, phase)
            asm volatile("tcgen05.fence.after_thread_sync;");

            mma_nvfp4_cta1_m128n128k64() // multiply
        }

        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]"
            :: "r"(producer_kv_mbar + v_slot_id * 8)
            : "memory"
        );

        phase ^= 1;
    }

    for (int stage = 0; stage < Q_STAGES; stage++) {
        asm volatile(
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]"
            :: "r"(o_full + stage * 8)
            : "memory"
        );
    }





    //for (int stage_id = 0; stage_id < NUM_STAGES; stage_id++) {
    //    mbarrier_wait(tma_mbar_addr + 8 * stage_id, tma_state);

        //same mbarrier address as previously
    //    mma_nvfp4_cta1_m128n128k64();
    //    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0]"
    //    :: "r"{mma_stage_0_bar});
        
    //}

    //tma_state ^= 1;


}


//for loop 
// Wait for load to arrive
// issue S@V for both tiles
// issue K@V for both tiles
// Send instruction
