__kernel void LoadScalarConst(
__global uint* out) 
{
	//out[0] = 7;
    __asm ("s_load_dwordx2 s[0:1], s[4:5], null"); // load s0, s1 from global adress ([s4, s5] + offset null) // Without it - crash 
    __asm ("s_mov_b32 s10, 3");                    // 3 -> s10
    __asm ("v_mov_b32_e32 v0, 0");                 // v0 == offset == 0
	__asm ("v_mov_b32 v1, s10");                   // s10 -> v1
    __asm ("s_waitcnt lgkmcnt(0)");                // waiting ops complite
	__asm ("global_store_dword v0, v1, s[0:1]");   // offload v1 to global memory with adress ([s0, s1] + offset v0)
}