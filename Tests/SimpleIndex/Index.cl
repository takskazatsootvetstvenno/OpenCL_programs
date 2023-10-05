
__kernel void Index(
__global uint* out) 
{
	const int i = get_global_id(0);
    out[i] = i;
}