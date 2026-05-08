/**
 * @file particle_owner_search.hh
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief Fluid-topology shadow + owner search for particles.
 *
 * The shadow holds, on every rank, the geometry of every *local* fluid quad
 * (physical bbox + cell spacing). Cross-rank owner lookup is delegated to
 * p4est_search_partition, which is non-collective and uses only the global
 * partition layout already replicated by p4est.
 *
 * See doc/design/particles.md (sections "Architecture" and "Drift regimes").
 */
#ifndef GRACE_PARTICLES_PARTICLE_OWNER_SEARCH_HH
#define GRACE_PARTICLES_PARTICLE_OWNER_SEARCH_HH

#include <grace_config.h>

#ifdef GRACE_ENABLE_PARTICLES

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace grace {
namespace particles {

//*****************************************************************************
// Per-quad geometry (host).
//*****************************************************************************

struct quad_bbox_t {
    double xlo, ylo, zlo;
    double xhi, yhi, zhi;
};

struct quad_geometry_t {
    quad_bbox_t bbox;   //!< physical extent of the quad
    double dx_cell;     //!< physical cell spacing (uniform per-quad, all axes)
    int level;          //!< p4est refinement level
    int tree;           //!< tree index containing the quad
};

//*****************************************************************************
// Owner-lookup result.
//*****************************************************************************

struct owner_t {
    int rank;        //!< owning rank, or -1 if outside global domain
    int local_quad;  //!< local quad index on owner rank, or -1 if rank != self
};

//*****************************************************************************
// Trilinear stencil base + fractional offset.
//
// Convention: cell centers at (i + 0.5) * dx_cell relative to bbox.xlo.
// base_(i,j,k) is the lower-left index of the 8-cell stencil; (fx,fy,fz) in
// [0,1) are the trilinear weights along each axis.
//*****************************************************************************

struct stencil_t {
    int base_i, base_j, base_k;
    double fx, fy, fz;
};

//*****************************************************************************
// Fluid-topology shadow singleton.
//*****************************************************************************

class fluid_topology_shadow_impl_t;

class fluid_topology_shadow_t {
  public:
    static fluid_topology_shadow_t& get();

    /// Rebuild local geometry table from current forest state. Call once at
    /// init and once after every regrid.
    void refresh();

    /// Number of local quads on this rank (matches forest::local_num_quadrants()).
    std::size_t local_num_quads() const noexcept;

    /// Read-only access to per-local-quad geometry.
    const std::vector<quad_geometry_t>& local_geometry() const noexcept;

    /// Slow-path single-point lookup. Returns rank=-1 if outside the global
    /// domain. If the owner is this rank, local_quad is also populated.
    owner_t find_owner(double x, double y, double z) const;

    /// Batched lookup. Output is resized to positions.size().
    void find_owners_batch(
        const std::vector<std::array<double, 3>>& positions,
        std::vector<owner_t>& owners) const;

    /// Fast-path bbox check on a previously-cached owner quad on this rank.
    /// Returns true iff the position lies inside the quad's bbox extended by
    /// halo_extent_cells worth of ghost cells. When true, populates `out`
    /// with the trilinear stencil for that quad.
    bool fast_path_check(int local_quad, double x, double y, double z,
                         int halo_extent_cells, stencil_t& out) const;

    /// Compute trilinear stencil for a position known to lie in (or in the
    /// ghost halo of) the given local quad.
    stencil_t make_stencil(int local_quad,
                           double x, double y, double z) const;

  private:
    fluid_topology_shadow_t();
    ~fluid_topology_shadow_t();
    fluid_topology_shadow_t(const fluid_topology_shadow_t&) = delete;
    fluid_topology_shadow_t& operator=(const fluid_topology_shadow_t&) = delete;

    fluid_topology_shadow_impl_t* _impl;
};

} // namespace particles
} // namespace grace

#endif // GRACE_ENABLE_PARTICLES

#endif // GRACE_PARTICLES_PARTICLE_OWNER_SEARCH_HH
