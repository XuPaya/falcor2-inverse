// Vertex-triangle proximity on the GPU for self-contact (falcor2_inverse.contact.SelfContact): the triangles'
// bounding boxes, grown by a margin, go into a ray-tracing acceleration structure, and a ray from each vertex
// (of nearly zero length, or along its displacement) visits the boxes it touches, as the photon gather of
// falcor2's PSDR does.

/// Distance from p to triangle (a, b, c), and the weights (w1, w2) of b and c at the closest point.
float closest_point(float3 p, float3 a, float3 b, float3 c, out float w1, out float w2)
{
    float3 ab = b - a, ac = c - a, ap = p - a;
    float d00 = dot(ab, ab), d01 = dot(ab, ac), d11 = dot(ac, ac), d20 = dot(ap, ab), d21 = dot(ap, ac);
    float den = max(d00 * d11 - d01 * d01, 1e-30);
    float v = (d11 * d20 - d01 * d21) / den, w = (d00 * d21 - d01 * d20) / den;
    float best = 1e30;
    w1 = 0;
    w2 = 0;
    if (v >= 0 && w >= 0 && v + w <= 1) {
        float3 q = a + v * ab + w * ac;
        best = dot(p - q, p - q);
        w1 = v;
        w2 = w;
    }
    float3 corners[3] = { a, b, c };
    [ForceUnroll]
    for (uint i = 0; i < 3; i++) {
        uint j = (i + 1) % 3;
        float3 e = corners[j] - corners[i];
        float t = clamp(dot(p - corners[i], e) / max(dot(e, e), 1e-30), 0.0, 1.0);
        float3 q = corners[i] + t * e;
        float d2 = dot(p - q, p - q);
        if (d2 < best) {
            best = d2;
            w1 = (i == 1 ? 1 - t : 0) + (j == 1 ? t : 0);
            w2 = (i == 2 ? 1 - t : 0) + (j == 2 ? t : 0);
        }
    }
    return sqrt(best);
}

/// Box of triangle t swept by scale times its vertices' displacements, grown by margin.
[shader("compute")]
[numthreads(64, 1, 1)]
void triangle_bounds(uint3 tid: SV_DispatchThreadID, uniform uint triangle_count, uniform float margin,
                     uniform float scale, StructuredBuffer<float3> positions, StructuredBuffer<float3> displacements,
                     StructuredBuffer<uint3> faces, RWStructuredBuffer<float3> bounds)
{
    uint t = tid.x;
    if (t >= triangle_count)
        return;
    float3 lo = float3(1e30), hi = float3(-1e30);
    [ForceUnroll]
    for (uint k = 0; k < 3; k++) {
        float3 p = positions[faces[t][k]], q = p + scale * displacements[faces[t][k]];
        lo = min(lo, min(p, q));
        hi = max(hi, max(p, q));
    }
    bounds[2 * t] = lo - margin;
    bounds[2 * t + 1] = hi + margin;
}

void add(RWByteAddressBuffer buffer, uint vertex, float3 value)
{
    [ForceUnroll]
    for (uint k = 0; k < 3; k++)
        buffer.InterlockedAddF32(12 * vertex + 4 * k, value[k]);
}

/// Adds the gradient of kappa * sum b(d) over the vertices and non-incident triangles closer than dhat, with IPC's
/// barrier b(d) = -(d - dhat)^2 log(d / dhat), to gradient (V x 3 floats); count[0] counts these pairs. The boxes
/// are grown by dhat.
[shader("compute")]
[numthreads(64, 1, 1)]
void barrier(uint3 tid: SV_DispatchThreadID, uniform uint vertex_count, uniform float dhat, uniform float kappa,
             RaytracingAccelerationStructure boxes, StructuredBuffer<float3> positions, StructuredBuffer<uint3> faces,
             RWByteAddressBuffer gradient, RWByteAddressBuffer count)
{
    uint v = tid.x;
    if (v >= vertex_count)
        return;
    float3 p = positions[v], own = float3(0.0);
    uint pairs = 0;
    RayDesc ray = { p, 0.0, float3(0.0, 0.0, 1.0), 1e-9 };
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(boxes, RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed()) {
        uint3 f = faces[q.CandidatePrimitiveIndex()];
        if (any(f == v))
            continue;
        float3 a = positions[f.x], b = positions[f.y], c = positions[f.z];
        float w1, w2;
        float d = closest_point(p, a, b, c, w1, w2);
        if (d >= dhat)
            continue;
        d = max(d, 1e-9 * dhat);
        float db = -2 * (d - dhat) * log(d / dhat) - (d - dhat) * (d - dhat) / d;
        float3 force = kappa * db / d * (p - ((1 - w1 - w2) * a + w1 * b + w2 * c));  // dE/dp
        own += force;
        add(gradient, f.x, -(1 - w1 - w2) * force);
        add(gradient, f.y, -w1 * force);
        add(gradient, f.z, -w2 * force);
        pairs++;
    }
    if (pairs > 0) {
        add(gradient, v, own);
        count.InterlockedAdd(0, pairs);
    }
}

/// Conservative advancement: count[0] = asuint(alpha) for the largest alpha <= 1 with which no vertex covers
/// safety times its distance to a non-incident triangle, relative to the triangle's motion; count[0] holds
/// asuint(1.0) before the dispatch (non-negative floats compare like their bits). The boxes are swept by the
/// displacements and grown by dhat, and each vertex's ray follows its displacement: the pairs that are not
/// visited stay farther apart than dhat over the whole step.
[shader("compute")]
[numthreads(64, 1, 1)]
void step_bound(uint3 tid: SV_DispatchThreadID, uniform uint vertex_count, uniform float safety,
                RaytracingAccelerationStructure boxes, StructuredBuffer<float3> positions,
                StructuredBuffer<float3> displacements, StructuredBuffer<uint3> faces, RWByteAddressBuffer count)
{
    uint v = tid.x;
    if (v >= vertex_count)
        return;
    float3 p = positions[v], dp = displacements[v];
    float length_dp = length(dp), alpha = 1.0;
    RayDesc ray = { p, 0.0, length_dp > 0 ? dp / length_dp : float3(0.0, 0.0, 1.0), max(length_dp, 1e-9) };
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(boxes, RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed()) {
        uint3 f = faces[q.CandidatePrimitiveIndex()];
        if (any(f == v))
            continue;
        float motion = max(length(dp - displacements[f.x]),
                           max(length(dp - displacements[f.y]), length(dp - displacements[f.z])));
        if (motion <= 0)
            continue;
        float w1, w2;
        float d = closest_point(p, positions[f.x], positions[f.y], positions[f.z], w1, w2);
        alpha = min(alpha, safety * d / motion);
    }
    if (alpha < 1)
        count.InterlockedMin(0, asuint(alpha));
}

/// count[0] = number of edges that pierce a non-incident triangle (self-intersections): each edge is a ray from
/// its first to its second vertex, tested against the triangles whose boxes it crosses.
[shader("compute")]
[numthreads(64, 1, 1)]
void edge_crossings(uint3 tid: SV_DispatchThreadID, uniform uint edge_count, RaytracingAccelerationStructure boxes,
                    StructuredBuffer<float3> positions, StructuredBuffer<uint3> faces, StructuredBuffer<uint2> edges,
                    RWByteAddressBuffer count)
{
    if (tid.x >= edge_count)
        return;
    uint2 e = edges[tid.x];
    float3 o = positions[e.x], dir = positions[e.y] - o;
    RayDesc ray = { o, 0.0, dir, 1.0 };
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(boxes, RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed()) {
        uint3 f = faces[q.CandidatePrimitiveIndex()];
        if (any(f == e.x) || any(f == e.y))
            continue;
        // Moeller-Trumbore
        float3 a = positions[f.x], e1 = positions[f.y] - a, e2 = positions[f.z] - a;
        float3 h = cross(dir, e2);
        float det = dot(e1, h);
        if (abs(det) < 1e-20)
            continue;
        float3 s = o - a, r = cross(s, e1);
        float u = dot(s, h) / det, w = dot(dir, r) / det, t = dot(e2, r) / det;
        if (u >= 0 && w >= 0 && u + w <= 1 && t >= 0 && t <= 1) {
            count.InterlockedAdd(0, 1);
            q.Abort();
        }
    }
}
