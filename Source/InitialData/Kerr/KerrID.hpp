/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef KERRID_HPP_
#define KERRID_HPP_

#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "GRParmParse.hpp"
#include "StateVariables.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"

#include <AMReX_Array4.H>
#include <array>

//! Exact initial data for a single Kerr black hole, spin along +z.
//!
//! Slice: the Boyer-Lindquist t = const hypersurface (maximal, K = 0), written
//! in the quasi-isotropic radial coordinate of Liu, Etienne & Shapiro (2010),
//! arXiv:1001.4077,
//!   r_BL = r (1 + r_+ / (4 r))^2,   r_+- = M +- sqrt(M^2 - a^2).
//! The slice is a wormhole: r = r_+/4 is the horizon (minimal surface), and
//! r -> 0 is a second asymptotically flat end (a puncture), so there is no
//! ring singularity on the grid and no excision is needed. For a = 0 it reduces
//! to isotropic Schwarzschild, psi = 1 + M / (2 r).
//!
//! The 3-metric and K_ij are built directly in Cartesian components, in a form
//! regular on the rotation axis (see compute_adm_vars), then decomposed into
//! chi, h_ij (det h = 1), A_ij (trace free) and K. Gamma^i is NOT set here: it
//! depends on derivatives of h_ij, so compute it afterwards with GammaCalculator.
//!
//! Gauge: pre-collapsed lapse sqrt(chi), zero shift and B^i. (The analytic BL
//! lapse vanishes on the horizon and changes sign through the throat, so it is
//! not used.)
//!
//! Parameters (prefix "kerr"):
//!   kerr.mass    BH mass M (> 0)
//!   kerr.spin    dimensionless spin a / M, with |a / M| < 1; the sign gives
//!                the direction along z (default 0)
//!   kerr.offset  position relative to geometry.center (default 0 0 0)
//!
//! GPU usage: construct on the host (the constructor reads parameters), then
//! capture by value in an amrex::ParallelFor and call operator() per cell.
class KerrID
{
  public:
    struct params_t
    {
        amrex::Real mass{};
        amrex::Real spin{}; //!< dimensionless, a / M
        std::array<amrex::Real, AMREX_SPACEDIM> center{}; //!< absolute coords

        //! Validate parameters (called via SimulationParameters, host only)
        inline static void check_params();

        //! Read parameters (host only)
        inline void fill_params();
    };

    //! Host only: reads parameters with GRParmParse
    inline explicit KerrID(amrex::Real a_dx);

    //! Set all state variables at cell (i, j, k), including ghost cells.
    //! Gamma^i is left at zero (see class comment).
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    operator()(int i, int j, int k,
               const amrex::Array4<amrex::Real> &state) const;

    //! Physical (ADM) 3-metric gamma_ij and extrinsic curvature K_ij, in
    //! Cartesian components, at (x, y, z) relative to the BH centre, for mass
    //! a_M and spin parameter a_a = J / M (dimensionful, |a_a| < a_M).
    //! Static so that it can be tested on its own.
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
    compute_adm_vars(amrex::Real x, amrex::Real y, amrex::Real z,
                     amrex::Real a_M, amrex::Real a_a, Tensor::Rank2 &gamma_LL,
                     Tensor::Rank2 &K_LL);

    [[nodiscard]] const params_t &get_params() const { return m_params; }

  private:
    amrex::Real m_dx;
    params_t m_params;
};

#include "KerrID.impl.hpp"

#endif /* KERRID_HPP_ */
