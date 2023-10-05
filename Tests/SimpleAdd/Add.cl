
__kernel void Add(
__constant uint* in,
__global uint* out) 
{
	const int i = get_global_id(0);
    const uint in_data = in[i];
    out[i] = in_data + (4 - i);
}