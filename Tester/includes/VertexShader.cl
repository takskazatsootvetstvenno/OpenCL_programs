 R"(
 float4 mult(__constant float* mat, float4 vec)
    {
        float dot0 = dot(vload4(0, mat), vec);
        float dot1 = dot(vload4(1, mat), vec);
        float dot2 = dot(vload4(2, mat), vec);
        float dot3 = dot(vload4(3, mat), vec);

        return (float4)(dot0, dot1, dot2, dot3);
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
        const float3 vertex = vload3(vertex_id * 2, position);
        const float3 normal = vload3(vertex_id * 2 + 1, position);
        const float4 point = (float4)(vertex, 1.0);

        float4 pos = mult(MVP, point);
        float w_reciprocal = 1.0f / pos.w;
        pos = pos * w_reciprocal;
        
        pos =  mult(screen, pos); 
        vstore3(pos.xyz, i * 2, output);
        vstore3(normal, i * 2 + 1, output);
    }
)"