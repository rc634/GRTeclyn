/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#if !defined(KERRID_HPP_)
#error "This file should only be included through KerrID.hpp"
#endif

#ifndef KERRID_IMPL_HPP_
#define KERRID_IMPL_HPP_

#include <AMReX_Algorithm.H>
#include <cmath>

void KerrID::params_t::check_params()
{
    GRParmParse kerr_pp("kerr");

    amrex::Real mass{};
    kerr_pp.get("mass", mass);
    if (mass <= 0.0)
    {
        kerr_pp.error("mass", "must be > 0");
    }

    amrex::Real spin{};
    kerr_pp.queryAdd("spin", spin);
    if (std::abs(spin) >= 1.0)
    {
        // a = M has r_+ = r_-: the throat becomes infinitely long and the
        // quasi-isotropic slice is singular
        kerr_pp.error("spin", "must satisfy |a / M| < 1");
    }

    GRParmParse geom_pp("geometry");
    std::array<amrex::Real, AMREX_SPACEDIM> center{};
    geom_pp.get("center", center);
    std::array<amrex::Real, AMREX_SPACEDIM> prob_extent{};
    geom_pp.get("prob_extent", prob_extent);

    std::array<amrex::Real, AMREX_SPACEDIM> offset{};
    kerr_pp.queryAdd("offset", offset);
    FOR (idir)
    {
        const amrex::Real bh_center = center[idir] + offset[idir];
        if (bh_center < 0.0 || bh_center > prob_extent[idir])
        {
            kerr_pp.warning("offset", "places the black hole outside the "
                                      "computational domain");
        }
    }
}

void KerrID::params_t::fill_params()
{
    GRParmParse kerr_pp("kerr");
    GRParmParse geom_pp("geometry");

    kerr_pp.get("mass", mass);
    kerr_pp.queryAdd("spin", spin); // defaults to 0 (Schwarzschild)
    geom_pp.get("center", center);

    std::array<amrex::Real, AMREX_SPACEDIM> offset{};
    kerr_pp.queryAdd("offset", offset); // defaults to 0 0 0
    FOR (idir)
    {
        center[idir] += offset[idir];
    }
}

KerrID::KerrID(amrex::Real a_dx) : m_dx(a_dx) { m_params.fill_params(); }

// The BL t = const slice in quasi-isotropic spherical coordinates (r, th, ph):
//   gamma = g_rr dr^2 + Sigma dth^2 + (A sin^2 th / Sigma) dph^2
//   K     = 2 K_rph dr dph + 2 K_thph dth dph
// with (r_BL is the Boyer-Lindquist radius)
//   Sigma = r_BL^2 + a^2 cos^2 th,   Delta = r_BL^2 - 2 M r_BL + a^2,
//   A     = (r_BL^2 + a^2)^2 - a^2 Delta sin^2 th,
//   g_rr  = Sigma (r + r_+/4)^2 / (r^3 (r_BL - r_-))    [regular at r = r_+/4]
// K_ij = (1 / 2 alpha) (L_beta gamma)_ij, with the BL lapse and shift
// beta^ph = -2 M a r_BL / A, gives (LES 2010, eqs. 10-11)
//   K_rph  = M a sin^2 th [3 r_BL^4 + 2 a^2 r_BL^2 - a^4
//            - a^2 (r_BL^2 - a^2) sin^2 th] (1 + r_+/(4r))
//            / (Sigma sqrt(A Sigma) sqrt(r (r_BL - r_-)))
//   K_thph = -2 a^3 M r_BL cos th sin^3 th (r - r_+/4) sqrt((r_BL - r_-) / r)
//            / (Sigma sqrt(A Sigma))
// Both are smooth through the throat.
//
// Cartesian form, regular on the axis. With n_i = x_i / r, m_i = (-y, x, 0),
// t_i = (x z, y z, -rho^2), rho^2 = x^2 + y^2:
//   dr = n,   r^2 (dth^2 + sin^2 th dph^2) = delta_ij - n_i n_j,
//   sin^2 th dph = m / r^2,   sin^3 th dth (x) dph = t (x) m / r^5.
// Writing gamma_phph = Sigma sin^2 th + (A - Sigma^2) sin^2 th / Sigma and using
// A - Sigma^2 = a^2 sin^2 th (Sigma + 2 M r_BL):
//   gamma_ij = g_rr n_i n_j + (Sigma / r^2)(delta_ij - n_i n_j)
//              + a^2 (Sigma + 2 M r_BL) / (Sigma r^4) m_i m_j
//   K_ij     = (K_rph / sin^2 th) / r^2 (n_i m_j + m_i n_j)
//              + (K_thph / (cos th sin^3 th)) (z / r^6) (t_i m_j + m_i t_j)
// None of these have 1 / sin th factors.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
KerrID::compute_adm_vars(amrex::Real x, amrex::Real y, amrex::Real z,
                         amrex::Real a_M, amrex::Real a_a,
                         Tensor::Rank2 &gamma_LL, Tensor::Rank2 &K_LL)
{
    const amrex::Real M  = a_M;
    const amrex::Real a  = a_a;
    const amrex::Real a2 = a * a;

    // quasi-isotropic radius, with the same floor as Coordinates::get_radius()
    const amrex::Real r =
        amrex::max(std::sqrt(x * x + y * y + z * z), amrex::Real(1.0e-6));
    const amrex::Real r2   = r * r;
    const amrex::Real rho2 = x * x + y * y;

    const amrex::Real cos2_theta = z * z / r2;
    const amrex::Real sin2_theta = rho2 / r2;

    const amrex::Real root    = std::sqrt(M * M - a2);
    const amrex::Real r_plus  = M + root;
    const amrex::Real r_minus = M - root;

    // Boyer-Lindquist radius; r_BL >= r_+ > r_- everywhere on the slice
    const amrex::Real r_BL_factor = 1.0 + 0.25 * r_plus / r;
    const amrex::Real r_BL        = r * r_BL_factor * r_BL_factor;
    const amrex::Real r_BL2       = r_BL * r_BL;
    const amrex::Real r_BL_m_rm   = r_BL - r_minus;

    const amrex::Real Sigma = r_BL2 + a2 * cos2_theta;
    const amrex::Real Delta = r_BL2 - 2.0 * M * r_BL + a2;
    const amrex::Real AA =
        (r_BL2 + a2) * (r_BL2 + a2) - a2 * Delta * sin2_theta;

    // ---- 3-metric
    const amrex::Real r_plus_quarter = r + 0.25 * r_plus;
    const amrex::Real g_rr =
        Sigma * r_plus_quarter * r_plus_quarter / (r * r2 * r_BL_m_rm);
    const amrex::Real coef_angular = Sigma / r2;
    const amrex::Real coef_mm =
        a2 * (Sigma + 2.0 * M * r_BL) / (Sigma * r2 * r2);

    const amrex::Real n[3] = {x / r, y / r, z / r};
    const amrex::Real m[3] = {-y, x, 0.0};
    const amrex::Real t[3] = {x * z, y * z, -rho2};

    // ---- extrinsic curvature: K_rph / sin^2 th and K_thph / (cos th sin^3 th)
    const amrex::Real sqrt_A_Sigma = std::sqrt(AA * Sigma);
    const amrex::Real bracket = 3.0 * r_BL2 * r_BL2 + 2.0 * a2 * r_BL2 -
                                a2 * a2 - a2 * (r_BL2 - a2) * sin2_theta;
    const amrex::Real K_rph_reduced = M * a * bracket * r_BL_factor /
                                      (Sigma * sqrt_A_Sigma *
                                       std::sqrt(r * r_BL_m_rm));
    const amrex::Real K_thph_reduced =
        -2.0 * a2 * a * M * r_BL * (r - 0.25 * r_plus) *
        std::sqrt(r_BL_m_rm / r) / (Sigma * sqrt_A_Sigma);

    const amrex::Real coef_nm = K_rph_reduced / r2;
    const amrex::Real coef_tm = K_thph_reduced * z / (r2 * r2 * r2);

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            const amrex::Real delta_ij = (i == j) ? 1.0 : 0.0;
            gamma_LL(i, j) = g_rr * n[i] * n[j] +
                             coef_angular * (delta_ij - n[i] * n[j]) +
                             coef_mm * m[i] * m[j];
            K_LL(i, j) = coef_nm * (n[i] * m[j] + m[i] * n[j]) +
                         coef_tm * (t[i] * m[j] + m[i] * t[j]);
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
KerrID::operator()(int i, int j, int k,
                   const amrex::Array4<amrex::Real> &state) const
{
    // Zero every component first: Theta, Gamma^i (set later by
    // GammaCalculator), shift and B^i stay zero.
    for (int n = 0; n < state.nComp(); ++n)
    {
        state(i, j, k, n) = 0.0;
    }

    const Coordinates coords(amrex::IntVect(AMREX_D_DECL(i, j, k)), m_dx,
                             m_params.center);

    // full-storage Rank2 tensors throughout (no Sym12 accumulation)
    Tensor::Rank2 gamma_LL{};
    Tensor::Rank2 K_LL{};
    compute_adm_vars(coords.x, coords.y, coords.z, m_params.mass,
                     m_params.spin * m_params.mass, gamma_LL, K_LL);

    // conformal decomposition
    const amrex::Real det_gamma   = TensorAlgebra::compute_determinant(gamma_LL);
    const Tensor::Rank2 gamma_UU  = TensorAlgebra::compute_inverse(gamma_LL);
    const amrex::Real K           = TensorAlgebra::compute_trace(K_LL, gamma_UU);
    const amrex::Real chi         = std::pow(det_gamma, -1.0 / 3.0);
    const amrex::Real one_third   = 1.0 / 3.0;

    state(i, j, k, c_chi) = chi;
    state(i, j, k, c_K)   = K;
    for (int a = 0; a < 3; ++a)
    {
        for (int b = a; b < 3; ++b)
        {
            state(i, j, k, sym_var_idx(c_h11, a, b)) = chi * gamma_LL(a, b);
            state(i, j, k, sym_var_idx(c_A11, a, b)) =
                chi * (K_LL(a, b) - one_third * gamma_LL(a, b) * K);
        }
    }

    // pre-collapsed lapse
    state(i, j, k, c_lapse) = std::sqrt(chi);
}

#endif /* KERRID_IMPL_HPP_ */
