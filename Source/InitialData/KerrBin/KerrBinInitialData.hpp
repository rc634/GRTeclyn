/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifdef USE_KERRBIN

#ifndef KERRBININITIALDATA_HPP_
#define KERRBININITIALDATA_HPP_

#include "Coordinates.hpp"
#include "GRParmParse.hpp"
#include "StateVariables.hpp"
#include "Tensor.hpp"

// kerrbin (KERRBIN_HOME/src, see Tools/GNUMake/packages/Make.kerrbin)
#include "evaluation/binary_evaluator.hpp"
#include "io/kid_file.hpp"

#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

#include <array>
#include <cmath>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

//! Initial data from kerrbin: a .kid file written by kerrbin's build/kerr_binary
//! (two boosted Kerr or Schwarzschild holes, constraints solved in the CTT
//! form, excision spheres filled with each hole's exact slice down to its
//! puncture). The file's user frame (the frame of kerrbin's parameter file) is
//! placed with its origin at geometry.center.
//!
//! The data are evaluated on the host: kerrbin is CPU code (std::function,
//! std::vector, exceptions), so it must not be called in a device kernel.
//! Parallelism: every MPI rank fills its own boxes, OpenMP threads share the
//! tiles of each box, and GPU builds fill a host MultiFab and copy it to the
//! device once. Gamma^i is left at zero; compute it afterwards with
//! GammaCalculator (it needs h_ij at neighbouring cells).
//!
//! Gauge: pre-collapsed lapse sqrt(chi), zero shift and B^i, Theta = 0.
//!
//! Parameters (prefix "kerrbin"):
//!   kerrbin.kid_file   path of the .kid file (relative to the run directory)
//!   kerrbin.only_A     true: hole A alone, its exact slice everywhere (no B,
//!                      no CTT solution), as a control run for A. B's puncture
//!                      is put on A's, since BHAmr tracks two (default false)
class KerrBinInitialData
{
  public:
    static constexpr int num_punctures = 2;

    static void check_params()
    {
        GRParmParse pp("kerrbin");
        std::string path;
        pp.get("kid_file", path);
        if (amrex::ParallelDescriptor::IOProcessor() &&
            !std::ifstream(path).good())
        {
            pp.error("kid_file", "cannot read '" + path + "'");
        }
    }

    //! The evaluator of the .kid file. Collective the first time (rank 0
    //! reads the file and broadcasts it, every rank parses its copy), so call
    //! it on all ranks.
    static const BinaryEvaluator &evaluator()
    {
        if (!s_evaluator)
        {
            BL_PROFILE("KerrBinInitialData::load");
            GRParmParse pp("kerrbin");
            std::string path;
            pp.get("kid_file", path);

            amrex::Vector<char> text;
            amrex::ParallelDescriptor::ReadAndBcastFile(path, text);
            try
            {
                s_evaluator = std::make_unique<const BinaryEvaluator>(
                    parse_kid(std::string(text.dataPtr())));
            }
            catch (const std::exception &e)
            {
                amrex::Abort("kerrbin: " + path + ": " + e.what());
            }
            print_summary(path);
        }
        return *s_evaluator;
    }

    //! Sets every state variable except Gamma^i (left at zero) in the valid
    //! and ghost cells of a_state
    static void set_state(amrex::MultiFab &a_state, amrex::Real a_dx)
    {
        BL_PROFILE("KerrBinInitialData::set_state");
        const BinaryEvaluator &kid = evaluator();
        const std::array<amrex::Real, AMREX_SPACEDIM> center = grid_center();
        std::optional<BoostedHole> hole_A;
        if (only_A())
        {
            hole_A.emplace(user_frame_A(kid.id()));
        }
        const BoostedHole *hole = hole_A ? &*hole_A : nullptr;

#ifdef AMREX_USE_GPU
        amrex::MFInfo mf_info;
        mf_info.SetArena(amrex::The_Cpu_Arena());
        amrex::MultiFab host_state(a_state.boxArray(),
                                   a_state.DistributionMap(), a_state.nComp(),
                                   a_state.nGrowVect(), mf_info);
#else
        amrex::MultiFab &host_state = a_state;
#endif

        // Always CPU work, so threaded even in GPU builds, and tiled
        // explicitly: GPU builds default to one tile per box. Dynamic
        // scheduling, since cells inside the excision spheres are much
        // cheaper than the series outside.
        const amrex::IntVect tile_size(AMREX_D_DECL(1024000, 8, 8));
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        for (amrex::MFIter mfi(
                 host_state,
                 amrex::MFItInfo().EnableTiling(tile_size).SetDynamic(true));
             mfi.isValid(); ++mfi)
        {
            const auto &state = host_state.array(mfi);
            amrex::LoopOnCpu(mfi.growntilebox(),
                             [&](int ix, int iy, int iz)
                             {
                                 set_cell(kid, hole, state, ix, iy, iz, a_dx,
                                          center);
                             });
        }

#ifdef AMREX_USE_GPU
        for (amrex::MFIter mfi(a_state); mfi.isValid(); ++mfi)
        {
            amrex::Gpu::htod_memcpy_async(
                a_state[mfi].dataPtr(), host_state[mfi].dataPtr(),
                host_state[mfi].size() * sizeof(amrex::Real));
        }
        amrex::Gpu::streamSynchronize(); // host_state is freed on return
#endif
    }

    //! Puncture positions in absolute coordinates, hole A then hole B
    static std::array<amrex::Real, AMREX_SPACEDIM * num_punctures>
    puncture_coords()
    {
        const BinaryID &id = evaluator().id();
        const HoleID &B    = only_A() ? id.A : id.B;
        const auto center  = grid_center();
        return {center[0] + id.A.position.x, center[1] + id.A.position.y,
                center[2] + id.A.position.z, center[0] + B.position.x,
                center[1] + B.position.y,    center[2] + B.position.z};
    }

    //! Bare masses, hole A then hole B
    static std::array<amrex::Real, num_punctures> puncture_masses()
    {
        const BinaryID &id = evaluator().id();
        return {id.A.mass, (only_A() ? id.A : id.B).mass};
    }

  private:
    static inline std::unique_ptr<const BinaryEvaluator> s_evaluator;

    static bool only_A()
    {
        bool only_A = false;
        GRParmParse pp("kerrbin");
        pp.query("only_A", only_A);
        return only_A;
    }

    //! Hole A alone in the user frame (a HoleID's vectors are in the solver
    //! frame, its position in the user frame)
    static BoostedHole user_frame_A(const BinaryID &a_id)
    {
        const HoleID &A = a_id.A;
        const Vec3 v    = a_id.frame.rotate(A.velocity);
        if (A.spin == 0.0)
        {
            return boosted_schwarzschild(A.mass, v, A.position);
        }
        return boosted_kerr(A.mass, A.spin, a_id.frame.rotate(A.axis), v,
                            A.position);
    }

    static std::array<amrex::Real, AMREX_SPACEDIM> grid_center()
    {
        std::array<amrex::Real, AMREX_SPACEDIM> center{};
        GRParmParse pp;
        pp.get("geometry.center", center);
        return center;
    }

    //! a_hole: if not null, the data are this hole's exact slice
    static void set_cell(const BinaryEvaluator &a_kid,
                         const BoostedHole *a_hole,
                         const amrex::Array4<amrex::Real> &a_state, int ix,
                         int iy, int iz, amrex::Real a_dx,
                         const std::array<amrex::Real, AMREX_SPACEDIM> &a_center)
    {
        for (int n = 0; n < a_state.nComp(); ++n)
        {
            a_state(ix, iy, iz, n) = 0.0;
        }

        const Coordinates coords(amrex::IntVect(AMREX_D_DECL(ix, iy, iz)),
                                 a_dx, a_center);
        const Vec3 x{coords.x, coords.y, coords.z};
        PhysicalData data;
        if (a_hole)
        {
            // the slice ends at the puncture: keep off it, as kerrbin does
            const HoleData h =
                a_hole->at(distance(x, a_hole->centre) > 1e-10
                               ? x
                               : a_hole->centre + Vec3{1e-10, 0.0, 0.0});
            data = {h.gamma, h.K};
        }
        else
        {
            data = a_kid.at(x);
        }

        // conformal decomposition
        double det_gamma{};
        const Sym3 gamma_UU = inverse(data.gamma, &det_gamma);
        double K = 0.0;
        for (int a = 0; a < 3; ++a)
        {
            for (int b = 0; b < 3; ++b)
            {
                K += gamma_UU[a][b] * data.K[a][b];
            }
        }
        const double chi = std::pow(det_gamma, -1.0 / 3.0);

        a_state(ix, iy, iz, c_chi) = chi;
        a_state(ix, iy, iz, c_K)   = K;
        for (int a = 0; a < 3; ++a)
        {
            for (int b = a; b < 3; ++b)
            {
                a_state(ix, iy, iz, sym_var_idx(c_h11, a, b)) =
                    chi * data.gamma[a][b];
                a_state(ix, iy, iz, sym_var_idx(c_A11, a, b)) =
                    chi * (data.K[a][b] - data.gamma[a][b] * K / 3.0);
            }
        }
        a_state(ix, iy, iz, c_lapse) = std::sqrt(chi);
    }

    static void print_summary(const std::string &a_path)
    {
        const BinaryID &id = s_evaluator->id();
        const auto center  = grid_center();
        amrex::Print() << "kerrbin initial data " << a_path << " (kerrbin "
                       << id.version << ", p_max " << id.Np << ", q_max "
                       << id.Nq << ", m_max " << id.M << ")\n";
        for (const HoleID *hole : {&id.A, &id.B})
        {
            amrex::Print() << "  hole " << (hole == &id.A ? 'A' : 'B')
                           << ": M " << hole->mass << ", a/M "
                           << hole->spin / hole->mass << ", |v| "
                           << norm(hole->velocity) << ", at ("
                           << center[0] + hole->position.x << ", "
                           << center[1] + hole->position.y << ", "
                           << center[2] + hole->position.z << ")\n";
        }
        if (only_A())
        {
            amrex::Print() << "  kerrbin.only_A: hole A alone, its exact "
                              "slice everywhere\n";
        }
    }
};

#endif /* KERRBININITIALDATA_HPP_ */
#endif /* USE_KERRBIN */
