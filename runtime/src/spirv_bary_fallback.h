#pragma once

// Per-pipeline `--bary-fallback` geometry-shader synthesis for D3D12 graphics PSOs (RT-BARY-FALLBACK-PSO).
//
// On GPUs without `VK_KHR_fragment_shader_barycentric`, `tools/rdna2spirv.py --bary-fallback` declares the raw
// barycentric pair a pixel shader reads as an ordinary smooth `vec3<f32>` input at the location recorded in
// `index.json` under `"bary_fallback_input": {"location": N}`. At graphics PSO creation, after any
// `--f16-interp-fallback` VS/PS location linking (`spirv_f16_halves.h`), `build_bary_fallback_gs` synthesizes the
// triangle pass-through Geometry SPIR-V module (`tools/bary_fallback_gs.py` equivalent) in memory for the exact
// `(VS, PS)` pair:
//   - copies `BuiltIn Position` across the triangle's 3 vertices (`Triangles` -> `OutputTriangleStrip`, 3 vertices);
//   - copies each used `vec4<f32>` `Location` input other than `bary_location` that the VS exports. The runtime
//     keeps unused or unexported PS locations as zero-filled GS outputs, and packs the VS -> GS interface
//     independently of the GS -> PS interface when geometry input locations are scarce;
//   - forwards a VS `BuiltIn ClipDistance` or `BuiltIn CullDistance` `float[1..8]` array when present (rejecting
//     mixed clip+cull and `ShaderLayer` / `ShaderViewportIndex`, which a pass-through helper cannot forward);
//   - writes `(1, 0, 0)`, `(0, 1, 0)`, `(0, 0, 1)` at `bary_location` so perspective interpolation yields
//     `BaryCoordKHR` (`.y` / `.z` = AMD's `i` / `j`).
//
// Pure C++20 stdlib, no Vulkan or Windows headers required.

#include "spirv_f16_halves.h"

#include <cstdint>
#include <map>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace leonida {

struct BaryFallbackGsInfo {
    std::uint32_t clip_count = 0;
    std::uint32_t cull_count = 0;
};

struct BaryFallbackGsLimits {
    std::uint32_t input_components = 0;
    std::uint32_t output_components = 0;
    std::uint32_t total_output_components = 0;
};

namespace spirv_detail {

// The varyings are logical pointers to Input variables. Mark their first use, including when a copy, access chain
// or call passes a pointer onward. Inspect only pointer-capable ID operands: an ExtInst opcode, memory-access
// mask or similar literal may have the same numeric value as a live or dead variable's ID.
inline void visit_bary_input_pointer_operands(const Inst& inst, const auto& visit) {
    const auto at = [&](std::size_t i) {
        if (i < inst.words.size()) visit(inst.words[i]);
    };
    const auto from = [&](std::size_t i) {
        for (; i < inst.words.size(); ++i) visit(inst.words[i]);
    };
    switch (inst.opcode) {
    case kOpLoad:
    case kOpAccessChain:
    case kOpInBoundsAccessChain:
    case 67:  // OpPtrAccessChain
    case 70:  // OpInBoundsPtrAccessChain
    case 83:  // OpCopyObject
    case 124: // OpBitcast
    case 400: // OpCopyLogical
        at(3);
        break;
    case kOpStore:
    case 63:  // OpCopyMemory
    case 64:  // OpCopyMemorySized
        at(1);
        at(2);
        break;
    case 57:  // OpFunctionCall
    case 169: // OpSelect (the condition is not a pointer)
        from(4);
        break;
    case kOpExtInst:
    case 4433: // OpExtInstWithForwardRefsKHR
        from(5);  // skip the instruction-set ID and literal opcode
        break;
    case 245: // OpPhi
        for (std::size_t i = 3; i < inst.words.size(); i += 2) at(i);
        break;
    case 254: // OpReturnValue
        at(1);
        break;
    case kOpVariable: // initializer of a function-local pointer variable
        at(4);
        break;
    case 401: // OpPtrEqual
    case 402: // OpPtrNotEqual
    case 403: // OpPtrDiff
        at(3);
        at(4);
        break;
    }
}

inline bool emit_bary_fallback_gs(std::span<const std::uint32_t> vs_words,
                                 std::span<const std::uint32_t> ps_words, std::uint32_t bary_location,
                                 std::vector<std::uint32_t>& gs_words, BaryFallbackGsInfo* out_info,
                                 std::string* error, BaryFallbackGsLimits limits = {},
                                 std::vector<std::uint32_t>* packed_vs = nullptr) {
    const auto fail = [&](const std::string& msg) {
        if (error) *error = msg;
        return false;
    };

    constexpr std::uint32_t kSpirvVersion15 = 0x00010500u;
    constexpr std::uint16_t kOpTypeVoid = 19;
    constexpr std::uint16_t kOpTypeArray = 28;
    constexpr std::uint16_t kOpTypeStruct = 30;
    constexpr std::uint16_t kOpTypeFunction = 33;
    constexpr std::uint16_t kOpConstant = 43;
    constexpr std::uint16_t kOpConstantComposite = 44;
    constexpr std::uint16_t kOpEmitVertex = 218;
    constexpr std::uint16_t kOpEndPrimitive = 219;
    constexpr std::uint16_t kOpLabel = 248;

    constexpr std::uint32_t kExecModelGeometry = 3;
    constexpr std::uint32_t kCapGeometry = 2;
    constexpr std::uint32_t kCapClipDistance = 32;
    constexpr std::uint32_t kCapCullDistance = 33;
    constexpr std::uint32_t kCapShaderLayer = 69;
    constexpr std::uint32_t kCapShaderViewportIndex = 70;

    constexpr std::uint32_t kExecModeInvocations = 0;
    constexpr std::uint32_t kExecModeTriangles = 22;
    constexpr std::uint32_t kExecModeOutputVertices = 26;
    constexpr std::uint32_t kExecModeOutputTriangleStrip = 29;

    constexpr std::uint32_t kDecorationBuiltIn = 11;
    constexpr std::uint32_t kDecorationComponent = 31;
    constexpr std::uint32_t kBuiltInPosition = 0;
    constexpr std::uint32_t kBuiltInClipDistance = 3;
    constexpr std::uint32_t kBuiltInCullDistance = 4;
    constexpr std::uint32_t kBuiltInLayer = 9;
    constexpr std::uint32_t kBuiltInViewportIndex = 10;

    Module vs_mod, ps_mod;
    if (!vs_mod.parse(vs_words)) return fail("bary_fallback: malformed VS SPIR-V");
    if (!ps_mod.parse(ps_words)) return fail("bary_fallback: malformed PS SPIR-V");

    // --- Inspect VS ---
    bool has_vs_ep = false;
    std::set<std::uint32_t> vs_float32_tys;
    std::map<std::uint32_t, bool> vs_int32_signed;  // int_type_id -> is_signed
    std::set<std::uint32_t> vs_v4float_tys;
    std::map<std::uint32_t, std::uint32_t> vs_pos_int_consts;  // const_id -> value (> 0)
    std::map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> vs_arr_tys;  // arr_id -> (elem_ty, len_id)
    std::map<std::uint32_t, std::vector<std::uint32_t>> vs_struct_tys;            // struct_id -> member_tys
    std::map<std::uint32_t, std::uint32_t> vs_ptr_out_pointee;                    // ptr_id -> pointee_ty
    std::map<std::uint32_t, std::uint32_t> vs_out_var_pointee;                    // var_id -> pointee_ty
    std::map<std::uint32_t, std::uint32_t> vs_var_builtins;                       // var_id -> BuiltIn
    std::map<std::uint32_t, std::vector<std::pair<std::uint32_t, std::uint32_t>>> vs_member_builtins;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> vs_loc_decs;             // (target_id, loc)
    std::set<std::uint32_t> vs_component_targets;
    bool vs_member_locations = false;

    for (const Inst& inst : vs_mod.insts) {
        if (inst.opcode == kOpFunction) break;
        if (inst.opcode == kOpCapability && inst.words.size() >= 2) {
            if (inst.words[1] == kCapShaderLayer)
                return fail("bary_fallback: VS declares ShaderLayer, which the fallback GS cannot forward");
            if (inst.words[1] == kCapShaderViewportIndex)
                return fail("bary_fallback: VS declares ShaderViewportIndex, which the fallback GS cannot forward");
        } else if (inst.opcode == kOpEntryPoint && inst.words.size() >= 4 && inst.words[1] == kExecModelVertex) {
            has_vs_ep = true;
        } else if (inst.opcode == kOpDecorate && inst.words.size() >= 4) {
            if (inst.words[2] == kDecorationBuiltIn) vs_var_builtins[inst.words[1]] = inst.words[3];
            if (inst.words[2] == kDecorationLocation) vs_loc_decs.emplace_back(inst.words[1], inst.words[3]);
            if (inst.words[2] == kDecorationComponent) vs_component_targets.insert(inst.words[1]);
        } else if (inst.opcode == kOpMemberDecorate && inst.words.size() >= 5) {
            if (inst.words[3] == kDecorationBuiltIn)
                vs_member_builtins[inst.words[1]].emplace_back(inst.words[2], inst.words[4]);
            if (inst.words[3] == kDecorationLocation) vs_member_locations = true;
        } else if (inst.opcode == kOpTypeFloat && inst.words.size() == 3 && inst.words[2] == 32) {
            vs_float32_tys.insert(inst.words[1]);
        } else if (inst.opcode == kOpTypeInt && inst.words.size() == 4 && inst.words[2] == 32) {
            vs_int32_signed[inst.words[1]] = (inst.words[3] != 0);
        } else if (inst.opcode == kOpTypeVector && inst.words.size() == 4 && inst.words[3] == 4 &&
                   vs_float32_tys.count(inst.words[2])) {
            vs_v4float_tys.insert(inst.words[1]);
        } else if (inst.opcode == kOpConstant && inst.words.size() == 4) {
            const auto it = vs_int32_signed.find(inst.words[1]);
            if (it != vs_int32_signed.end()) {
                const std::uint32_t raw = inst.words[3];
                const bool positive = it->second ? (static_cast<std::int32_t>(raw) > 0) : (raw > 0);
                if (positive) vs_pos_int_consts[inst.words[2]] = raw;
            }
        } else if (inst.opcode == kOpTypeArray && inst.words.size() == 4) {
            vs_arr_tys[inst.words[1]] = {inst.words[2], inst.words[3]};
        } else if (inst.opcode == kOpTypeStruct && inst.words.size() >= 2) {
            vs_struct_tys[inst.words[1]].assign(inst.words.begin() + 2, inst.words.end());
        } else if (inst.opcode == kOpTypePointer && inst.words.size() == 4 && inst.words[2] == kStorageOutput) {
            vs_ptr_out_pointee[inst.words[1]] = inst.words[3];
        } else if (inst.opcode == kOpVariable && inst.words.size() >= 4 && inst.words[3] == kStorageOutput) {
            const auto pt = vs_ptr_out_pointee.find(inst.words[1]);
            if (pt != vs_ptr_out_pointee.end()) vs_out_var_pointee[inst.words[2]] = pt->second;
        }
    }
    if (!has_vs_ep) return fail("bary_fallback: VS has no Vertex entry point");

    std::uint32_t clip_count = 0;
    std::uint32_t cull_count = 0;
    const auto record_vs_builtin = [&](std::uint32_t bi, std::uint32_t ty_id) -> bool {
        if (bi == kBuiltInLayer)
            return fail("bary_fallback: VS writes BuiltIn Layer, which the fallback GS cannot forward");
        if (bi == kBuiltInViewportIndex)
            return fail("bary_fallback: VS writes BuiltIn ViewportIndex, which the fallback GS cannot forward");
        if (bi != kBuiltInClipDistance && bi != kBuiltInCullDistance) return true;
        const auto arr_it = vs_arr_tys.find(ty_id);
        if (arr_it == vs_arr_tys.end() || !vs_float32_tys.count(arr_it->second.first))
            return fail("bary_fallback: VS ClipDistance/CullDistance output is not float[N]");
        const auto len_it = vs_pos_int_consts.find(arr_it->second.second);
        if (len_it == vs_pos_int_consts.end() || len_it->second < 1 || len_it->second > 8)
            return fail("bary_fallback: VS ClipDistance/CullDistance length must be in 1..8");
        std::uint32_t& dst = (bi == kBuiltInClipDistance) ? clip_count : cull_count;
        if (dst != 0 && dst != len_it->second)
            return fail("bary_fallback: conflicting VS ClipDistance/CullDistance output declarations");
        dst = len_it->second;
        return true;
    };

    for (const auto& [var_id, pt_ty] : vs_out_var_pointee) {
        if (const auto bi_it = vs_var_builtins.find(var_id); bi_it != vs_var_builtins.end()) {
            if (!record_vs_builtin(bi_it->second, pt_ty)) return false;
        }
        if (const auto m_it = vs_member_builtins.find(pt_ty); m_it != vs_member_builtins.end()) {
            const auto st_it = vs_struct_tys.find(pt_ty);
            if (st_it != vs_struct_tys.end()) {
                for (const auto& [member_idx, bi] : m_it->second) {
                    if (member_idx >= st_it->second.size())
                        return fail("bary_fallback: out-of-range MemberDecorate on VS output struct");
                    if (!record_vs_builtin(bi, st_it->second[member_idx])) return false;
                }
            }
        }
    }
    if (clip_count && cull_count)
        return fail("bary_fallback: cannot forward mixed ClipDistance and CullDistance");

    std::map<std::uint32_t, std::uint32_t> vs_out_var_by_loc;
    for (const auto& [target, loc] : vs_loc_decs) {
        if (!vs_out_var_pointee.count(target)) continue;
        if (!vs_out_var_by_loc.emplace(loc, target).second)
            return fail("bary_fallback: duplicate VS Output Location " + std::to_string(loc));
    }

    // --- Inspect PS ---
    bool has_ps_ep = false;
    std::set<std::uint32_t> ps_float32_tys;
    std::set<std::uint32_t> ps_v3float_tys;
    std::set<std::uint32_t> ps_v4float_tys;
    std::map<std::uint32_t, std::uint32_t> ps_ptr_in_pointee;
    std::map<std::uint32_t, std::uint32_t> ps_in_var_pointee;
    std::map<std::uint32_t, std::uint32_t> ps_in_var_by_loc;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> ps_loc_decs;
    std::set<std::uint32_t> ps_component_targets;

    for (const Inst& inst : ps_mod.insts) {
        if (inst.opcode == kOpFunction) break;
        if (inst.opcode == kOpEntryPoint && inst.words.size() >= 4 && inst.words[1] == kExecModelFragment) {
            has_ps_ep = true;
        } else if (inst.opcode == kOpDecorate && inst.words.size() >= 4) {
            if (inst.words[2] == kDecorationLocation) ps_loc_decs.emplace_back(inst.words[1], inst.words[3]);
            if (inst.words[2] == kDecorationComponent) ps_component_targets.insert(inst.words[1]);
        } else if (inst.opcode == kOpTypeFloat && inst.words.size() == 3 && inst.words[2] == 32) {
            ps_float32_tys.insert(inst.words[1]);
        } else if (inst.opcode == kOpTypeVector && inst.words.size() == 4 && ps_float32_tys.count(inst.words[2])) {
            if (inst.words[3] == 3) ps_v3float_tys.insert(inst.words[1]);
            if (inst.words[3] == 4) ps_v4float_tys.insert(inst.words[1]);
        } else if (inst.opcode == kOpTypePointer && inst.words.size() == 4 && inst.words[2] == kStorageInput) {
            ps_ptr_in_pointee[inst.words[1]] = inst.words[3];
        } else if (inst.opcode == kOpVariable && inst.words.size() >= 4 && inst.words[3] == kStorageInput) {
            const auto pt = ps_ptr_in_pointee.find(inst.words[1]);
            if (pt != ps_ptr_in_pointee.end()) ps_in_var_pointee[inst.words[2]] = pt->second;
        }
    }
    if (!has_ps_ep) return fail("bary_fallback: PS has no Fragment entry point");

    std::map<std::uint32_t, std::uint32_t> ps_in_loc_pointee;
    for (const auto& [target, loc] : ps_loc_decs) {
        const auto it = ps_in_var_pointee.find(target);
        if (it == ps_in_var_pointee.end()) continue;
        if (ps_component_targets.count(target))
            return fail("bary_fallback: Component-decorated PS Input at Location " + std::to_string(loc));
        if (!ps_in_loc_pointee.emplace(loc, it->second).second)
            return fail("bary_fallback: duplicate PS Input Location " + std::to_string(loc));
        ps_in_var_by_loc.emplace(loc, target);
    }

    // The translator also declares inputs it never reads. A variable not referenced in any function is unused,
    // even if listed in OpEntryPoint.
    std::set<std::uint32_t> referenced_ps_inputs;
    if (packed_vs) {
        bool in_function = false;
        for (const Inst& inst : ps_mod.insts) {
            if (inst.opcode == kOpFunction) in_function = true;
            if (inst.opcode == kOpFunctionEnd) in_function = false;
            if (!in_function) continue;
            visit_bary_input_pointer_operands(inst, [&](std::uint32_t id) {
                if (ps_in_var_pointee.count(id)) referenced_ps_inputs.insert(id);
            });
        }
    }

    if (const auto bary_it = ps_in_loc_pointee.find(bary_location);
        bary_it != ps_in_loc_pointee.end() && !ps_v3float_tys.count(bary_it->second)) {
        return fail("bary_fallback: bary_location " + std::to_string(bary_location) +
                    " collides with a non-vec3 PS Input");
    }

    std::vector<std::uint32_t> pass_locs;
    std::vector<std::uint32_t> forwarded_locs;
    std::set<std::uint32_t> zero_loc_set;
    for (const auto& [loc, pt_ty] : ps_in_loc_pointee) {
        if (loc == bary_location) continue;
        if (!ps_v4float_tys.count(pt_ty))
            return fail("bary_fallback: PS pass-through Input at Location " + std::to_string(loc) + " is not vec4<f32>");
        pass_locs.push_back(loc);
        if (packed_vs && !referenced_ps_inputs.count(ps_in_var_by_loc.at(loc))) {
            zero_loc_set.insert(loc);
            continue;
        }
        const auto vs_it = vs_out_var_by_loc.find(loc);
        if (vs_it != vs_out_var_by_loc.end()) {
            if (!vs_v4float_tys.count(vs_out_var_pointee.at(vs_it->second)) ||
                vs_component_targets.count(vs_it->second))
                return fail("bary_fallback: VS Output at Location " + std::to_string(loc) + " is not vec4<f32>");
            forwarded_locs.push_back(loc);
        } else {
            zero_loc_set.insert(loc);
        }
    }

    const std::uint32_t builtin_components = 4 + clip_count + cull_count;
    std::map<std::uint32_t, std::uint32_t> remapped_inputs;   // PS location -> new GS input location
    std::map<std::uint32_t, std::uint32_t> vs_location_edits; // VS variable ID -> new output location
    if (limits.input_components) {
        const std::uint64_t required = builtin_components + 4ull * forwarded_locs.size();
        if (required > limits.input_components)
            return fail("bary_fallback: GS needs " + std::to_string(required) + " input components, device permits " +
                        std::to_string(limits.input_components));
        const std::uint32_t slots = (limits.input_components - builtin_components) / 4;
        std::set<std::uint32_t> used(forwarded_locs.begin(), forwarded_locs.end());
        std::uint32_t next_free = 0;
        for (const std::uint32_t loc : forwarded_locs) {
            if (loc < slots) continue;
            // A free low slot may still contain an unread VS output. Swap it into the moved output's old slot,
            // so the VS neither adds a location nor needs to rewrite any stores or pointer types.
            while (used.count(next_free)) ++next_free;
            const auto other = vs_out_var_by_loc.find(next_free);
            if (other != vs_out_var_by_loc.end()) {
                if (!vs_v4float_tys.count(vs_out_var_pointee.at(other->second)) ||
                    vs_component_targets.count(other->second))
                    return fail("bary_fallback: cannot repack a non-vec4 VS Output at Location " +
                                std::to_string(next_free));
                vs_location_edits.emplace(other->second, loc);
            }
            remapped_inputs.emplace(loc, next_free);
            vs_location_edits.emplace(vs_out_var_by_loc.at(loc), next_free++);
        }
        if (!vs_location_edits.empty()) {
            if (vs_member_locations)
                return fail("bary_fallback: cannot repack VS outputs with member Locations");
            for (const auto& [loc, var] : vs_out_var_by_loc) {
                if (!vs_v4float_tys.count(vs_out_var_pointee.at(var)) || vs_component_targets.count(var))
                    return fail("bary_fallback: cannot repack a non-vec4 VS Output at Location " + std::to_string(loc));
            }
        }
    }
    if (limits.output_components || limits.total_output_components) {
        const std::uint64_t count = builtin_components + 3 + 4ull * pass_locs.size();
        const std::uint64_t last = std::max(4ull * bary_location + 3,
                                            pass_locs.empty() ? 0ull : 4ull * pass_locs.back() + 4);
        if (const auto required = std::max(count, last); limits.output_components && required > limits.output_components)
            return fail("bary_fallback: GS needs " + std::to_string(required) + " output components per vertex, "
                        "device permits " + std::to_string(limits.output_components));
        if (limits.total_output_components && 3 * count > limits.total_output_components)
            return fail("bary_fallback: GS needs " + std::to_string(3 * count) + " total output components, "
                        "device permits " + std::to_string(limits.total_output_components));
    }

    // --- Synthesize Geometry SPIR-V module ---
    std::uint32_t next_id = 1;
    const auto alloc = [&]() { return next_id++; };

    const std::uint32_t id_main = alloc();
    const std::uint32_t id_entry = alloc();
    const std::uint32_t id_void = alloc();
    const std::uint32_t id_fn = alloc();
    const std::uint32_t id_uint = alloc();
    const std::uint32_t id_int = alloc();
    const std::uint32_t id_float = alloc();
    const std::uint32_t id_v3float = alloc();
    const std::uint32_t id_v4float = alloc();
    const std::uint32_t id_uint_3 = alloc();
    const std::uint32_t id_arr_v4_3 = alloc();
    const std::uint32_t id_ptr_in_arr = alloc();
    const std::uint32_t id_ptr_in_v4 = alloc();
    const std::uint32_t id_ptr_out_v4 = alloc();
    const std::uint32_t id_ptr_out_v3 = alloc();
    const std::uint32_t id_null_v4 = zero_loc_set.empty() ? 0 : alloc();

    const std::uint32_t dist_count = clip_count ? clip_count : cull_count;
    const std::uint32_t dist_cap = cull_count ? kCapCullDistance : kCapClipDistance;
    const std::uint32_t dist_builtin = cull_count ? kBuiltInCullDistance : kBuiltInClipDistance;
    std::uint32_t id_dist_count = 0;
    std::uint32_t id_arr_dist = 0;
    std::uint32_t id_tri_dist = 0;
    std::uint32_t id_ptr_in_tri_dist = 0;
    std::uint32_t id_ptr_in_dist = 0;
    std::uint32_t id_ptr_out_dist = 0;
    if (dist_count) {
        id_dist_count = (dist_count == 3) ? id_uint_3 : alloc();
        id_arr_dist = alloc();
        id_tri_dist = alloc();
        id_ptr_in_tri_dist = alloc();
        id_ptr_in_dist = alloc();
        id_ptr_out_dist = alloc();
    }

    const std::uint32_t id_f0 = alloc();
    const std::uint32_t id_f1 = alloc();
    const std::uint32_t id_bary[3] = {alloc(), alloc(), alloc()};
    const std::uint32_t id_int_v[3] = {alloc(), alloc(), alloc()};

    const std::uint32_t id_in_pos = alloc();
    const std::uint32_t id_out_pos = alloc();
    std::map<std::uint32_t, std::uint32_t> id_in_loc;
    std::map<std::uint32_t, std::uint32_t> id_out_loc;
    for (std::uint32_t loc : forwarded_locs) id_in_loc[loc] = alloc();
    for (std::uint32_t loc : pass_locs) id_out_loc[loc] = alloc();
    const std::uint32_t id_out_bary = alloc();
    const std::uint32_t id_in_dist = dist_count ? alloc() : 0;
    const std::uint32_t id_out_dist = dist_count ? alloc() : 0;

    Module gs;
    // 1. Capabilities
    gs.insts.push_back(make_inst(kOpCapability, {kCapGeometry}));
    if (dist_count) gs.insts.push_back(make_inst(kOpCapability, {dist_cap}));

    // 2. MemoryModel
    gs.insts.push_back(make_inst(kOpMemoryModel, {0, 1}));  // Logical GLSL450

    // 3. EntryPoint ("main\0")
    std::vector<std::uint32_t> ep = {kExecModelGeometry, id_main, 0x6e69616du, 0x00000000u, id_in_pos, id_out_pos};
    for (std::uint32_t loc : forwarded_locs) ep.push_back(id_in_loc[loc]);
    for (std::uint32_t loc : pass_locs) ep.push_back(id_out_loc[loc]);
    ep.push_back(id_out_bary);
    if (dist_count) {
        ep.push_back(id_in_dist);
        ep.push_back(id_out_dist);
    }
    gs.insts.push_back(make_inst_vec(kOpEntryPoint, ep));

    // 4. ExecutionModes
    gs.insts.push_back(make_inst(kOpExecutionMode, {id_main, kExecModeTriangles}));
    gs.insts.push_back(make_inst(kOpExecutionMode, {id_main, kExecModeInvocations, 1}));
    gs.insts.push_back(make_inst(kOpExecutionMode, {id_main, kExecModeOutputTriangleStrip}));
    gs.insts.push_back(make_inst(kOpExecutionMode, {id_main, kExecModeOutputVertices, 3}));

    // 5. Decorations
    gs.insts.push_back(make_inst(kOpDecorate, {id_in_pos, kDecorationBuiltIn, kBuiltInPosition}));
    gs.insts.push_back(make_inst(kOpDecorate, {id_out_pos, kDecorationBuiltIn, kBuiltInPosition}));
    if (dist_count) {
        gs.insts.push_back(make_inst(kOpDecorate, {id_in_dist, kDecorationBuiltIn, dist_builtin}));
        gs.insts.push_back(make_inst(kOpDecorate, {id_out_dist, kDecorationBuiltIn, dist_builtin}));
    }
    for (std::uint32_t loc : forwarded_locs) {
        const auto packed = remapped_inputs.find(loc);
        gs.insts.push_back(make_inst(kOpDecorate, {id_in_loc[loc], kDecorationLocation,
                                                  packed == remapped_inputs.end() ? loc : packed->second}));
    }
    for (std::uint32_t loc : pass_locs)
        gs.insts.push_back(make_inst(kOpDecorate, {id_out_loc[loc], kDecorationLocation, loc}));
    gs.insts.push_back(make_inst(kOpDecorate, {id_out_bary, kDecorationLocation, bary_location}));

    // 6. Types, Constants, Global Variables
    gs.insts.push_back(make_inst(kOpTypeVoid, {id_void}));
    gs.insts.push_back(make_inst(kOpTypeFunction, {id_fn, id_void}));
    gs.insts.push_back(make_inst(kOpTypeInt, {id_uint, 32, 0}));
    gs.insts.push_back(make_inst(kOpTypeInt, {id_int, 32, 1}));
    gs.insts.push_back(make_inst(kOpTypeFloat, {id_float, 32}));
    gs.insts.push_back(make_inst(kOpTypeVector, {id_v3float, id_float, 3}));
    gs.insts.push_back(make_inst(kOpTypeVector, {id_v4float, id_float, 4}));
    gs.insts.push_back(make_inst(kOpConstant, {id_uint, id_uint_3, 3}));
    gs.insts.push_back(make_inst(kOpTypeArray, {id_arr_v4_3, id_v4float, id_uint_3}));
    gs.insts.push_back(make_inst(kOpTypePointer, {id_ptr_in_arr, kStorageInput, id_arr_v4_3}));
    gs.insts.push_back(make_inst(kOpTypePointer, {id_ptr_in_v4, kStorageInput, id_v4float}));
    gs.insts.push_back(make_inst(kOpTypePointer, {id_ptr_out_v4, kStorageOutput, id_v4float}));
    gs.insts.push_back(make_inst(kOpTypePointer, {id_ptr_out_v3, kStorageOutput, id_v3float}));
    if (id_null_v4) gs.insts.push_back(make_inst(kOpConstantNull, {id_v4float, id_null_v4}));
    if (dist_count) {
        if (id_dist_count != id_uint_3) gs.insts.push_back(make_inst(kOpConstant, {id_uint, id_dist_count, dist_count}));
        gs.insts.push_back(make_inst(kOpTypeArray, {id_arr_dist, id_float, id_dist_count}));
        gs.insts.push_back(make_inst(kOpTypeArray, {id_tri_dist, id_arr_dist, id_uint_3}));
        gs.insts.push_back(make_inst(kOpTypePointer, {id_ptr_in_tri_dist, kStorageInput, id_tri_dist}));
        gs.insts.push_back(make_inst(kOpTypePointer, {id_ptr_in_dist, kStorageInput, id_arr_dist}));
        gs.insts.push_back(make_inst(kOpTypePointer, {id_ptr_out_dist, kStorageOutput, id_arr_dist}));
    }
    gs.insts.push_back(make_inst(kOpConstant, {id_float, id_f0, 0x00000000u}));
    gs.insts.push_back(make_inst(kOpConstant, {id_float, id_f1, 0x3f800000u}));
    gs.insts.push_back(make_inst(kOpConstantComposite, {id_v3float, id_bary[0], id_f1, id_f0, id_f0}));
    gs.insts.push_back(make_inst(kOpConstantComposite, {id_v3float, id_bary[1], id_f0, id_f1, id_f0}));
    gs.insts.push_back(make_inst(kOpConstantComposite, {id_v3float, id_bary[2], id_f0, id_f0, id_f1}));
    for (std::uint32_t v = 0; v < 3; ++v) gs.insts.push_back(make_inst(kOpConstant, {id_int, id_int_v[v], v}));

    gs.insts.push_back(make_inst(kOpVariable, {id_ptr_in_arr, id_in_pos, kStorageInput}));
    gs.insts.push_back(make_inst(kOpVariable, {id_ptr_out_v4, id_out_pos, kStorageOutput}));
    for (std::uint32_t loc : forwarded_locs)
        gs.insts.push_back(make_inst(kOpVariable, {id_ptr_in_arr, id_in_loc[loc], kStorageInput}));
    for (std::uint32_t loc : pass_locs)
        gs.insts.push_back(make_inst(kOpVariable, {id_ptr_out_v4, id_out_loc[loc], kStorageOutput}));
    gs.insts.push_back(make_inst(kOpVariable, {id_ptr_out_v3, id_out_bary, kStorageOutput}));
    if (dist_count) {
        gs.insts.push_back(make_inst(kOpVariable, {id_ptr_in_tri_dist, id_in_dist, kStorageInput}));
        gs.insts.push_back(make_inst(kOpVariable, {id_ptr_out_dist, id_out_dist, kStorageOutput}));
    }

    // 7. Function main
    gs.insts.push_back(make_inst(kOpFunction, {id_void, id_main, 0, id_fn}));
    gs.insts.push_back(make_inst(kOpLabel, {id_entry}));
    for (std::uint32_t v = 0; v < 3; ++v) {
        const std::uint32_t pos_ptr = alloc();
        const std::uint32_t pos_val = alloc();
        gs.insts.push_back(make_inst(kOpAccessChain, {id_ptr_in_v4, pos_ptr, id_in_pos, id_int_v[v]}));
        gs.insts.push_back(make_inst(kOpLoad, {id_v4float, pos_val, pos_ptr}));
        gs.insts.push_back(make_inst(kOpStore, {id_out_pos, pos_val}));

        for (std::uint32_t loc : pass_locs) {
            if (zero_loc_set.count(loc)) {
                gs.insts.push_back(make_inst(kOpStore, {id_out_loc[loc], id_null_v4}));
            } else {
                const std::uint32_t attr_ptr = alloc();
                const std::uint32_t attr_val = alloc();
                gs.insts.push_back(make_inst(kOpAccessChain, {id_ptr_in_v4, attr_ptr, id_in_loc[loc], id_int_v[v]}));
                gs.insts.push_back(make_inst(kOpLoad, {id_v4float, attr_val, attr_ptr}));
                gs.insts.push_back(make_inst(kOpStore, {id_out_loc[loc], attr_val}));
            }
        }
        if (dist_count) {
            const std::uint32_t d_ptr = alloc();
            const std::uint32_t d_val = alloc();
            gs.insts.push_back(make_inst(kOpAccessChain, {id_ptr_in_dist, d_ptr, id_in_dist, id_int_v[v]}));
            gs.insts.push_back(make_inst(kOpLoad, {id_arr_dist, d_val, d_ptr}));
            gs.insts.push_back(make_inst(kOpStore, {id_out_dist, d_val}));
        }
        gs.insts.push_back(make_inst(kOpStore, {id_out_bary, id_bary[v]}));
        gs.insts.push_back(make_inst(kOpEmitVertex, {}));
    }
    gs.insts.push_back(make_inst(kOpEndPrimitive, {}));
    gs.insts.push_back(make_inst(kOpReturn, {}));
    gs.insts.push_back(make_inst(kOpFunctionEnd, {}));

    gs.header = {kSpirvMagic, kSpirvVersion15, 0u, next_id, 0u};
    if (!vs_location_edits.empty()) {
        for (Inst& inst : vs_mod.insts) {
            if (inst.opcode != kOpDecorate || inst.words.size() != 4 || inst.words[2] != kDecorationLocation) continue;
            const auto it = vs_location_edits.find(inst.words[1]);
            if (it != vs_location_edits.end()) inst.words[3] = it->second;
        }
        *packed_vs = vs_mod.serialize();
    }
    gs_words = gs.serialize();
    if (out_info) *out_info = {clip_count, cull_count};
    return true;
}

}  // namespace spirv_detail

inline bool build_bary_fallback_gs(std::span<const std::uint32_t> vs_words,
                                   std::span<const std::uint32_t> ps_words, std::uint32_t bary_location,
                                   std::vector<std::uint32_t>& gs_words, BaryFallbackGsInfo* out_info = nullptr,
                                   std::string* error = nullptr) {
    return spirv_detail::emit_bary_fallback_gs(vs_words, ps_words, bary_location, gs_words, out_info, error);
}

// Link the two interfaces separately. FP16 synthesis needs the ordinary varying namespace; only the VS outputs
// actually read by the GS need to land within its smaller input limit. The final GS -> PS locations stay put.
inline bool link_bary_fallback_stages(std::vector<std::uint32_t>& vs_words, const ShaderEntry& vs_entry,
                                     std::vector<std::uint32_t>& ps_words, const ShaderEntry& ps_entry,
                                     std::uint32_t f16_top_location, BaryFallbackGsLimits limits,
                                     std::vector<std::uint32_t>& gs_words, BaryFallbackGsInfo* info = nullptr,
                                     std::string* error = nullptr) {
    if (!ps_entry.bary_fallback_location) {
        if (error) *error = "bary_fallback: missing pixel shader input location";
        return false;
    }
    const bool link_halves = !vs_entry.f16_halves.empty() || !ps_entry.f16_halves.empty();
    std::vector<std::uint32_t> linked_vs, linked_ps;
    if (link_halves) {
        linked_vs = vs_words;
        linked_ps = ps_words;
        if (limits.output_components)
            f16_top_location = std::min(f16_top_location,
                                        f16_half_top_location(limits.output_components, limits.output_components));
        if (!link_f16_halves(linked_vs, vs_entry, linked_ps, ps_entry, error, f16_top_location)) return false;
    }
    std::vector<std::uint32_t> packed_vs;
    if (!spirv_detail::emit_bary_fallback_gs(link_halves ? linked_vs : vs_words, link_halves ? linked_ps : ps_words,
                                            *ps_entry.bary_fallback_location, gs_words, info, error, limits, &packed_vs))
        return false;
    if (!packed_vs.empty()) vs_words = std::move(packed_vs);
    else if (link_halves) vs_words = std::move(linked_vs);
    if (link_halves) ps_words = std::move(linked_ps);
    return true;
}

}  // namespace leonida
