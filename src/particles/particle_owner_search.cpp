/**
 * @file particle_owner_search.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Implementation of fluid-topology shadow + owner search.
 */
#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <grace/particles/particle_owner_search.hh>

#include <grace/amr/forest.hh>
#include <grace/amr/p4est_headers.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/utils/grace_utils.hh>

#include <algorithm>
#include <cmath>

namespace grace {
namespace particles {

//*****************************************************************************
// Internal: per-quad bbox computation from a p4est quadrant.
//*****************************************************************************

namespace {

quad_geometry_t compute_quad_geometry(p4est_t* p4est,
                                      p4est_topidx_t which_tree,
                                      p4est_quadrant_t* quad,
                                      int nx_per_side)
{
    auto* pconn = p4est->connectivity;
    double xyz_lo[3] = {0., 0., 0.};
    p4est_qcoord_to_vertex(pconn, which_tree,
                           quad->x, quad->y, quad->z, xyz_lo);

    // Tree spacing in physical units along x (assume isotropic per tree).
    auto nv1 = pconn->tree_to_vertex[which_tree * P4EST_CHILDREN];
    auto nv2 = pconn->tree_to_vertex[which_tree * P4EST_CHILDREN + 1];
    double xv1 = pconn->vertices[3UL * nv1];
    double xv2 = pconn->vertices[3UL * nv2];
    double dx_tree = xv2 - xv1;

    double dx_quad = dx_tree / static_cast<double>(1 << static_cast<int>(quad->level));

    quad_geometry_t g;
    g.bbox.xlo = xyz_lo[0];
    g.bbox.ylo = xyz_lo[1];
    g.bbox.zlo = xyz_lo[2];
    g.bbox.xhi = xyz_lo[0] + dx_quad;
    g.bbox.yhi = xyz_lo[1] + dx_quad;
    g.bbox.zhi = xyz_lo[2] + dx_quad;
    g.dx_cell  = dx_quad / static_cast<double>(nx_per_side);
    g.level    = static_cast<int>(quad->level);
    g.tree     = static_cast<int>(which_tree);
    return g;
}

// Per-thread-local payload for p4est_search_partition point callback.
struct point_query_t {
    double x, y, z;
    int    owner_rank;  // -1 if not yet found
};

int point_callback_partition(p4est_t* p4est,
                             p4est_topidx_t which_tree,
                             p4est_quadrant_t* quadrant,
                             int pfirst, int plast, void* point)
{
    (void)p4est;
    auto* pt = static_cast<point_query_t*>(point);

    auto* pconn = p4est->connectivity;
    double xyz_lo[3] = {0., 0., 0.};
    p4est_qcoord_to_vertex(pconn, which_tree,
                           quadrant->x, quadrant->y, quadrant->z, xyz_lo);

    auto nv1 = pconn->tree_to_vertex[which_tree * P4EST_CHILDREN];
    auto nv2 = pconn->tree_to_vertex[which_tree * P4EST_CHILDREN + 1];
    double xv1 = pconn->vertices[3UL * nv1];
    double xv2 = pconn->vertices[3UL * nv2];
    double dx_tree = xv2 - xv1;
    double dx_quad = dx_tree / static_cast<double>(1 << static_cast<int>(quadrant->level));

    // Half-open bbox check: [lo, hi) — particles exactly on the upper face
    // belong to the next quadrant.
    if (pt->x < xyz_lo[0] || pt->x >= xyz_lo[0] + dx_quad) return 0;
    if (pt->y < xyz_lo[1] || pt->y >= xyz_lo[1] + dx_quad) return 0;
    if (pt->z < xyz_lo[2] || pt->z >= xyz_lo[2] + dx_quad) return 0;

    if (pfirst == plast) {
        pt->owner_rank = pfirst;
        return 0; // resolved; no further descent needed for this point
    }
    return 1; // keep descending
}

// Local-only point search to recover local_quad on the owner rank.
struct local_query_t {
    double x, y, z;
    int    local_quad; // -1 if not found
};

int point_callback_local(p4est_t* p4est,
                         p4est_topidx_t which_tree,
                         p4est_quadrant_t* quadrant,
                         p4est_locidx_t local_num, void* point)
{
    (void)p4est;
    auto* pt = static_cast<local_query_t*>(point);

    auto* pconn = p4est->connectivity;
    double xyz_lo[3] = {0., 0., 0.};
    p4est_qcoord_to_vertex(pconn, which_tree,
                           quadrant->x, quadrant->y, quadrant->z, xyz_lo);

    auto nv1 = pconn->tree_to_vertex[which_tree * P4EST_CHILDREN];
    auto nv2 = pconn->tree_to_vertex[which_tree * P4EST_CHILDREN + 1];
    double xv1 = pconn->vertices[3UL * nv1];
    double xv2 = pconn->vertices[3UL * nv2];
    double dx_tree = xv2 - xv1;
    double dx_quad = dx_tree / static_cast<double>(1 << static_cast<int>(quadrant->level));

    if (pt->x < xyz_lo[0] || pt->x >= xyz_lo[0] + dx_quad) return 0;
    if (pt->y < xyz_lo[1] || pt->y >= xyz_lo[1] + dx_quad) return 0;
    if (pt->z < xyz_lo[2] || pt->z >= xyz_lo[2] + dx_quad) return 0;

    if (local_num >= 0) {
        // Leaf — record local index.
        pt->local_quad = static_cast<int>(local_num);
        return 0;
    }
    return 1; // internal node, keep descending
}

} // namespace

//*****************************************************************************
// Pimpl.
//*****************************************************************************

class fluid_topology_shadow_impl_t {
  public:
    std::vector<quad_geometry_t> local_geometry;
    int nx_per_side = 0; // cells per quad axis (cached from amr::get_quadrant_extents)
};

//*****************************************************************************
// Singleton lifecycle.
//*****************************************************************************

fluid_topology_shadow_t::fluid_topology_shadow_t()
  : _impl(new fluid_topology_shadow_impl_t())
{}

fluid_topology_shadow_t::~fluid_topology_shadow_t() {
    delete _impl;
}

fluid_topology_shadow_t& fluid_topology_shadow_t::get() {
    static fluid_topology_shadow_t instance;
    return instance;
}

//*****************************************************************************
// refresh()
//*****************************************************************************

void fluid_topology_shadow_t::refresh() {
    auto* p4est = grace::amr::forest::get().get();
    ASSERT(p4est != nullptr, "Cannot refresh particle topology shadow before forest init.");

    auto extents = grace::amr::get_quadrant_extents();
    int nx = static_cast<int>(std::get<0>(extents));
    ASSERT(nx > 0, "Quadrant extents not initialized.");
    _impl->nx_per_side = nx;

    const std::size_t n_local = static_cast<std::size_t>(p4est->local_num_quadrants);
    _impl->local_geometry.clear();
    _impl->local_geometry.reserve(n_local);

    for (p4est_topidx_t it = p4est->first_local_tree;
         it <= p4est->last_local_tree; ++it)
    {
        p4est_tree_t* tree = p4est_tree_array_index(p4est->trees, it);
        const std::size_t n_in_tree = tree->quadrants.elem_count;
        for (std::size_t iq = 0; iq < n_in_tree; ++iq) {
            p4est_quadrant_t* q = p4est_quadrant_array_index(&tree->quadrants, iq);
            _impl->local_geometry.push_back(
                compute_quad_geometry(p4est, it, q, nx));
        }
    }

    ASSERT(_impl->local_geometry.size() == n_local,
           "Geometry table size mismatch with forest local quad count.");
}

//*****************************************************************************
// Accessors.
//*****************************************************************************

std::size_t fluid_topology_shadow_t::local_num_quads() const noexcept {
    return _impl->local_geometry.size();
}

const std::vector<quad_geometry_t>&
fluid_topology_shadow_t::local_geometry() const noexcept {
    return _impl->local_geometry;
}

//*****************************************************************************
// find_owner: cross-rank lookup via p4est_search_partition, then recover
// local_quad if owner == self.
//*****************************************************************************

owner_t fluid_topology_shadow_t::find_owner(double x, double y, double z) const {
    owner_t out{-1, -1};

    auto* p4est = grace::amr::forest::get().get();
    ASSERT(p4est != nullptr, "find_owner called before forest init.");

    point_query_t q{x, y, z, -1};
    sc_array_t* points = sc_array_new_data(&q, sizeof(point_query_t), 1);

    p4est_search_partition(p4est, /*call_post=*/0,
                           /*quadrant_fn=*/nullptr,
                           &point_callback_partition, points);
    sc_array_destroy(points);

    out.rank = q.owner_rank;
    if (out.rank < 0) return out;

    if (out.rank == p4est->mpirank) {
        local_query_t lq{x, y, z, -1};
        sc_array_t* lpoints = sc_array_new_data(&lq, sizeof(local_query_t), 1);
        p4est_search_local(p4est, /*call_post=*/0,
                           /*quadrant_fn=*/nullptr,
                           &point_callback_local, lpoints);
        sc_array_destroy(lpoints);
        out.local_quad = lq.local_quad;
    }
    return out;
}

//*****************************************************************************
// Batched lookup. v1: loop over points; coalesce later if profiling demands.
//*****************************************************************************

void fluid_topology_shadow_t::find_owners_batch(
    const std::vector<std::array<double, 3>>& positions,
    std::vector<owner_t>& owners) const
{
    owners.clear();
    owners.resize(positions.size(), owner_t{-1, -1});
    for (std::size_t ip = 0; ip < positions.size(); ++ip) {
        const auto& p = positions[ip];
        owners[ip] = find_owner(p[0], p[1], p[2]);
    }
}

//*****************************************************************************
// Fast-path bbox check on a cached owner quad.
//*****************************************************************************

bool fluid_topology_shadow_t::fast_path_check(int local_quad,
                                              double x, double y, double z,
                                              int halo_extent_cells,
                                              stencil_t& out) const
{
    if (local_quad < 0 ||
        local_quad >= static_cast<int>(_impl->local_geometry.size()))
        return false;

    const auto& g = _impl->local_geometry[local_quad];
    const double halo = halo_extent_cells * g.dx_cell;
    if (x < g.bbox.xlo - halo || x >= g.bbox.xhi + halo) return false;
    if (y < g.bbox.ylo - halo || y >= g.bbox.yhi + halo) return false;
    if (z < g.bbox.zlo - halo || z >= g.bbox.zhi + halo) return false;

    out = make_stencil(local_quad, x, y, z);
    return true;
}

//*****************************************************************************
// Trilinear stencil construction.
//
// Cell centers at (i + 0.5) * dx_cell relative to bbox.lo. base_i = floor of
// the centered coordinate; fx in [0,1) is the trilinear weight along x.
//*****************************************************************************

stencil_t fluid_topology_shadow_t::make_stencil(int local_quad,
                                                double x, double y, double z) const
{
    ASSERT(local_quad >= 0 &&
           local_quad < static_cast<int>(_impl->local_geometry.size()),
           "make_stencil called with out-of-range local_quad.");

    const auto& g = _impl->local_geometry[local_quad];
    const double inv_dx = 1.0 / g.dx_cell;

    auto compute = [inv_dx](double pos, double lo, int& base, double& frac) {
        double cx = (pos - lo) * inv_dx - 0.5;
        double fb = std::floor(cx);
        base = static_cast<int>(fb);
        frac = cx - fb;
    };

    stencil_t s{};
    compute(x, g.bbox.xlo, s.base_i, s.fx);
    compute(y, g.bbox.ylo, s.base_j, s.fy);
    compute(z, g.bbox.zlo, s.base_k, s.fz);
    return s;
}

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES
