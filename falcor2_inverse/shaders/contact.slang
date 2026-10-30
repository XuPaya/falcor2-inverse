// Vertex-triangle proximity on the GPU for self-contact (falcor2_inverse.contact.SelfContact): the triangles'
// bounding boxes, grown by a margin, go into a ray-tracing acceleration structure, and a ray from each vertex
// (of nearly zero length, or along its displacement) visits the boxes it touches, as the photon gather of
// falcor2's PSDR does.

uint triangle_count;
uint vertex_count;
uint edge_count;
float margin;
float scale;
float dhat;
float kappa;
float safety;
RaytracingAccelerationStructure boxes;
StructuredBuffer<float3> positions;
StructuredBuffer<float3> displacements;
StructuredBuffer<uint3> faces;
StructuredBuffer<uint2> edges;
RWStructuredBuffer<float3> bounds;
RWByteAddressBuffer gradient;
RWByteAddressBuffer count;

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
void triangle_bounds(uint3 tid: SV_DispatchThreadID)
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

/// An exact slab test removes conservative hardware AABB candidates. Parallel axes are handled explicitly,
/// including the tiny rays used for stationary vertices. Return an in-box reporting point on the ray interval.
bool box_candidate(RayDesc ray, uint triangle, out float t_hit)
{
    float3 lo = bounds[2 * triangle], hi = bounds[2 * triangle + 1];
    float enter = ray.TMin, leave = ray.TMax;
    t_hit = ray.TMin;
    [ForceUnroll]
    for (uint axis = 0; axis < 3; axis++) {
        if (ray.Direction[axis] == 0) {
            if (ray.Origin[axis] < lo[axis] || ray.Origin[axis] > hi[axis])
                return false;
        } else {
            float a = (lo[axis] - ray.Origin[axis]) / ray.Direction[axis];
            float b = (hi[axis] - ray.Origin[axis]) / ray.Direction[axis];
            enter = max(enter, min(a, b));
            leave = min(leave, max(a, b));
        }
    }
    t_hit = enter;
    return enter <= leave;
}

RayDesc barrier_ray(uint v) { return { positions[v], 0.0, float3(0, 0, 1), 1e-9 }; }

RayDesc step_ray(uint v)
{
    float3 dp = displacements[v];
    float length_dp = length(dp);
    return { positions[v], 0.0, length_dp > 0 ? dp / length_dp : float3(0, 0, 1), max(length_dp, 1e-9) };
}

RayDesc edge_ray(uint edge)
{
    uint2 e = edges[edge];
    return { positions[e.x], 0.0, positions[e.y] - positions[e.x], 1.0 };
}

/// Shared narrow phases: both traversal implementations use the same exclusions, forces and step bounds.
void barrier_candidate(uint v, uint triangle, inout float3 own, inout uint pairs)
{
    uint3 f = faces[triangle];
    if (any(f == v))
        return;
    float3 p = positions[v], a = positions[f.x], b = positions[f.y], c = positions[f.z];
    float w1, w2;
    float d = closest_point(p, a, b, c, w1, w2);
    if (d >= dhat)
        return;
    d = max(d, 1e-9 * dhat);
    float db = -2 * (d - dhat) * log(d / dhat) - (d - dhat) * (d - dhat) / d;
    float3 force = kappa * db / d * (p - ((1 - w1 - w2) * a + w1 * b + w2 * c));
    own += force;
    add(gradient, f.x, -(1 - w1 - w2) * force);
    add(gradient, f.y, -w1 * force);
    add(gradient, f.z, -w2 * force);
    pairs++;
}

void step_candidate(uint v, uint triangle, inout float alpha)
{
    uint3 f = faces[triangle];
    if (any(f == v))
        return;
    float3 dp = displacements[v];
    float motion = max(length(dp - displacements[f.x]),
                       max(length(dp - displacements[f.y]), length(dp - displacements[f.z])));
    if (motion <= 0)
        return;
    float w1, w2;
    float d = closest_point(positions[v], positions[f.x], positions[f.y], positions[f.z], w1, w2);
    alpha = min(alpha, safety * d / motion);
}

bool edge_candidate(uint edge, uint triangle)
{
    uint2 e = edges[edge];
    uint3 f = faces[triangle];
    if (any(f == e.x) || any(f == e.y))
        return false;
    float3 o = positions[e.x], dir = positions[e.y] - o;
    // Moeller-Trumbore, including both segment endpoints.
    float3 a = positions[f.x], e1 = positions[f.y] - a, e2 = positions[f.z] - a;
    float3 h = cross(dir, e2);
    float det = dot(e1, h);
    if (abs(det) < 1e-20)
        return false;
    float3 s = o - a, r = cross(s, e1);
    float u = dot(s, h) / det, w = dot(dir, r) / det, t = dot(e2, r) / det;
    return u >= 0 && w >= 0 && u + w <= 1 && t >= 0 && t <= 1;
}

void finish_barrier(uint v, float3 own, uint pairs)
{
    if (pairs > 0) {
        add(gradient, v, own);
        count.InterlockedAdd(0, pairs);
    }
}

// Inline query path. No procedural candidate is committed, so every overlapping box is visited.
[shader("compute")]
[numthreads(64, 1, 1)]
void barrier(uint3 tid: SV_DispatchThreadID)
{
    uint v = tid.x;
    if (v >= vertex_count)
        return;
    float3 own = float3(0);
    uint pairs = 0;
    RayDesc ray = barrier_ray(v);
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(boxes, RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed()) {
        uint triangle = q.CandidatePrimitiveIndex();
        float t;
        if (box_candidate(ray, triangle, t))
            barrier_candidate(v, triangle, own, pairs);
    }
    finish_barrier(v, own, pairs);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void step_bound(uint3 tid: SV_DispatchThreadID)
{
    uint v = tid.x;
    if (v >= vertex_count)
        return;
    float alpha = 1.0;
    RayDesc ray = step_ray(v);
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(boxes, RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed()) {
        uint triangle = q.CandidatePrimitiveIndex();
        float t;
        if (box_candidate(ray, triangle, t))
            step_candidate(v, triangle, alpha);
    }
    if (alpha < 1)
        count.InterlockedMin(0, asuint(alpha));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void edge_crossings(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= edge_count)
        return;
    RayDesc ray = edge_ray(tid.x);
    if (!any(ray.Direction != 0))
        return;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(boxes, RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed()) {
        uint triangle = q.CandidatePrimitiveIndex();
        float t;
        if (box_candidate(ray, triangle, t) && edge_candidate(tid.x, triangle)) {
            count.InterlockedAdd(0, 1);
            q.Abort();
        }
    }
}

// Pipeline fallback. Only any-hit has side effects: the BLAS forbids duplicate any-hit invocations per primitive.
struct ContactAttributes { uint unused; };
struct BarrierPayload { uint vertex; float3 own; uint pairs; };
struct StepPayload { uint vertex; float alpha; };
struct EdgePayload { uint edge; bool crossing; };
static const uint CONTACT_RAY_FLAGS = RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;

[shader("intersection")]
void contact_intersection()
{
    RayDesc ray = { WorldRayOrigin(), RayTMin(), WorldRayDirection(), RayTCurrent() };
    float t;
    if (box_candidate(ray, PrimitiveIndex(), t)) {
        ContactAttributes attributes = { 0 };
        ReportHit(t, 0, attributes);
    }
}

[shader("anyhit")]
void barrier_any_hit(inout BarrierPayload payload, ContactAttributes attributes)
{
    barrier_candidate(payload.vertex, PrimitiveIndex(), payload.own, payload.pairs);
    IgnoreHit();
}

[shader("anyhit")]
void step_bound_any_hit(inout StepPayload payload, ContactAttributes attributes)
{
    step_candidate(payload.vertex, PrimitiveIndex(), payload.alpha);
    IgnoreHit();
}

[shader("anyhit")]
void edge_crossings_any_hit(inout EdgePayload payload, ContactAttributes attributes)
{
    if (edge_candidate(payload.edge, PrimitiveIndex())) {
        payload.crossing = true;
        AcceptHitAndEndSearch();
    } else {
        IgnoreHit();
    }
}

[shader("miss")] void barrier_miss(inout BarrierPayload payload) {}
[shader("miss")] void step_bound_miss(inout StepPayload payload) {}
[shader("miss")] void edge_crossings_miss(inout EdgePayload payload) {}

[shader("raygeneration")]
void barrier_trace()
{
    uint v = DispatchRaysIndex().x;
    if (v >= vertex_count)
        return;
    BarrierPayload payload = { v, float3(0), 0 };
    TraceRay(boxes, CONTACT_RAY_FLAGS, 0xff, 0, 0, 0, barrier_ray(v), payload);
    finish_barrier(v, payload.own, payload.pairs);
}

[shader("raygeneration")]
void step_bound_trace()
{
    uint v = DispatchRaysIndex().x;
    if (v >= vertex_count)
        return;
    StepPayload payload = { v, 1.0 };
    TraceRay(boxes, CONTACT_RAY_FLAGS, 0xff, 0, 0, 0, step_ray(v), payload);
    if (payload.alpha < 1)
        count.InterlockedMin(0, asuint(payload.alpha));
}

[shader("raygeneration")]
void edge_crossings_trace()
{
    uint edge = DispatchRaysIndex().x;
    if (edge >= edge_count)
        return;
    RayDesc ray = edge_ray(edge);
    if (!any(ray.Direction != 0))
        return;
    EdgePayload payload = { edge, false };
    TraceRay(boxes, CONTACT_RAY_FLAGS, 0xff, 0, 0, 0, ray, payload);
    if (payload.crossing)
        count.InterlockedAdd(0, 1);
}
