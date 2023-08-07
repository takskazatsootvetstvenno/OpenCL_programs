 R"(
 #pragma OPENCL EXTENSION cl_khr_fp16 : enable
 half4 mult(half16 mat, half4 vec)
    {
        half4 line_0 = (half4)(mat[0],  mat[1],  mat[2],  mat[3]);
        half4 line_1 = (half4)(mat[4],  mat[5],  mat[6],  mat[7]);
        half4 line_2 = (half4)(mat[8],  mat[9],  mat[10], mat[11]);
        half4 line_3 = (half4)(mat[12], mat[13], mat[14], mat[15]);

        half dot0 = dot(line_0, vec);
        half dot1 = dot(line_1, vec);
        half dot2 = dot(line_2, vec);
        half dot3 = dot(line_3, vec);

        return (half4)(dot0, dot1, dot2, dot3);
    }

    __kernel void vertex_shader(
    __constant float* position,
    __constant uint* index,
    __constant float* MVP,
    __constant float* screen,
    __global float* output) 
    {
	    const int i = get_global_id(0);
        const uint vertex_id = index[i];
        const half3 vertex = vload_half3(vertex_id * 2, position);
        const half3 normal = vload_half3(vertex_id * 2 + 1, position);
        const half4 point = (half4)(vertex, 1.0);

        half4 pos = mult(MVP, point);
        half w_reciprocal = 1.0f / pos.w;
        pos = pos * w_reciprocal;
        
        const half16 matrix = vload_half16(0, MVP);
        pos =  mult(screen, pos); 
        vstore_half3(pos.xyz, i * 2, output);
        vstore_half3(normal, i * 2 + 1, output);
    }
)"