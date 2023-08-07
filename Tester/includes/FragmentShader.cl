 R"(
    uint getUintColor(float color) {
        uint3 bytes = (uint3)(color * 255.0);
        return (bytes.r << 24) | (bytes.g << 16) | (bytes.b << 8);
    }
    uint getWindowColor(float3 color)
    {
        uint r = getUintColor(color[0]);
        uint g = getUintColor(color[1]);
        uint b = getUintColor(color[2]);

        return (r << 16) + (g << 8) + b;
    }

    __kernel void fragment_shader(
    __constant float* VStoFSBuffer,
    __global uint* output) 
    {
	    const int i = get_global_id(0);
        float3 A = vload3(i * 2, VStoFSBuffer);
        float3 B = vload3(i * 2 + 1, VStoFSBuffer);
        float3 C = vload3(i * 2 + 2, VStoFSBuffer);
        uint color = getWindowColor(A);
        output[i] = color;
    }
)"