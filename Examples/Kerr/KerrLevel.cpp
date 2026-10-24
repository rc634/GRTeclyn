/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#include "KerrLevel.hpp"

#include "AlgebraicConstraintsEnforcer.hpp"
#include "CCZ4RHS.hpp"
#include "ChiTagger.hpp"
#include "Constraints.hpp"
#include "ExtractionTagger.hpp"
#include "FourthOrderDerivatives.hpp"
#include "PositiveChiAndLapse.hpp"
#include "PunctureTagger.hpp"
#include "PunctureTracker.hpp"
#include "SixthOrderDerivatives.hpp"
#include "Weyl4.hpp"
#include "WeylExtraction.hpp"

#ifdef USE_TWOPUNCTURES
#include "BoostedBHInitialData.hpp"
#include "TwoPuncturesInitialData.hpp"
#else
#include "GammaCalculator.hpp"
#include "KerrID.hpp"
#endif

BHAmr<KerrLevel::num_punctures> *KerrLevel::get_bh_amr_ptr()
{
    return dynamic_cast<BHAmr<num_punctures> *>(get_gr_amr_ptr());
}

PunctureTracker<KerrLevel::num_punctures> &
KerrLevel::get_puncture_tracker()
{
    return get_bh_amr_ptr()->get_puncture_tracker();
}

void KerrLevel::variableSetUp()
{
    BL_PROFILE("KerrLevel::variableSetUp()");

    // Set up the state variables
    state_variable_set_up();

    Constraints::set_up(state_index);

    Weyl4::set_up(state_index);
}

// Things to do during the advance step after RK4 steps
void KerrLevel::specific_advance()
{
    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &state_arrays   = state_new.arrays();

    // The classes to be used
    AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    PositiveChiAndLapse positive_chi_lapse;

    // Enforce det(h)=1, the trace free A_ij condition and positive chi and
    // lapse
    amrex::ParallelFor(state_new,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           algebraic_constraints_enforcer(ix, iy, iz,
                                                          state_arrays[box_no]);
                           positive_chi_lapse(ix, iy, iz, state_arrays[box_no]);
                       });
}

// Initial data: TwoPunctures if built with USE_TWOPUNCTURES=TRUE, otherwise
// the single-BH KerrID data
void KerrLevel::initData()
{
    BL_PROFILE("KerrLevel::initData");
    if (get_gr_amr_ptr()->Verbose() > 0)
    {
        amrex::Print() << "KerrLevel::initData " << Level() << "\n";
    }
#ifdef USE_TWOPUNCTURES
    TwoPuncturesInitialData two_punctures_initial_data(Geom().CellSize(0));

    two_punctures_initial_data.solve(); // only solves first time

    amrex::MultiFab &state_new = get_new_data(state_index);
#ifdef AMREX_USE_GPU
    amrex::MFInfo mf_info;
    mf_info.SetArena(amrex::The_Cpu_Arena());
    amrex::MultiFab host_state(state_new.boxArray(),
                               state_new.DistributionMap(), state_new.nComp(),
                               state_new.nGrowVect(), mf_info);
#else
    amrex::MultiFab &host_state = state_new;
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(state_new, amrex::TilingIfNotGPU()); mfi.isValid();
         ++mfi)
    {
        const amrex::Box &grown_tile_box = mfi.growntilebox();
        const auto &state_array          = host_state.array(mfi);

        amrex::LoopOnCpu(
            grown_tile_box, [=](int ix, int iy, int iz)
            { two_punctures_initial_data(ix, iy, iz, state_array); });
#ifdef AMREX_USE_GPU
        // Copy to device
        amrex::Gpu::htod_memcpy_async(
            state_new[mfi].dataPtr(), host_state[mfi].dataPtr(),
            host_state[mfi].size() * sizeof(amrex::Real));
#endif
    }

#else
    // Single stationary BH (Source/InitialData/Kerr). Constructed on the host
    // (reads params), then captured by value in the kernel below.
    const amrex::Real dx = Geom().CellSize(0);
    const KerrID kerr_initial_data(dx);
    static_assert(std::is_trivially_copyable_v<KerrID>,
                  "KerrID needs to be device copyable");

    // KerrID sets every component except Gamma^i (zeroing first), including
    // ghost cells, since the data is analytic
    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &state_arrays   = state_new.arrays();
    amrex::ParallelFor(
        state_new, state_new.nGrowVect(),
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        { kerr_initial_data(ix, iy, iz, state_arrays[box_no]); });

    // Gamma^i = h^jk Gamma^i_jk from finite differences of h_ij (non-zero for
    // Kerr). Valid cells only: their stencils read the analytic ghost cells
    // set above, so no FillBoundary is needed. Same derivative order as the
    // evolution. Separate kernel since it needs h_ij at neighbouring cells.
    if (m_evolution_spatial_derivative_order == 6)
    {
        const GammaCalculator<SixthOrderDerivatives> gamma_calculator(dx);
        amrex::ParallelFor(
            state_new, amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            { gamma_calculator(ix, iy, iz, state_arrays[box_no]); });
    }
    else
    {
        const GammaCalculator<FourthOrderDerivatives> gamma_calculator(dx);
        amrex::ParallelFor(
            state_new, amrex::IntVect(0),
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            { gamma_calculator(ix, iy, iz, state_arrays[box_no]); });
    }
#endif
    amrex::Gpu::streamSynchronize();

    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        // need to set the puncture coordinates as we use it for the puncture
        // tagging
#ifdef USE_TWOPUNCTURES
        BoostedBHInitialData::params_t bh1_params(1);
        BoostedBHInitialData::params_t bh2_params(2);
        two_punctures_initial_data.set_bh_params(bh1_params, bh2_params);

        get_puncture_tracker().set_puncture_coords(
            {bh1_params.center[0], bh1_params.center[1], bh1_params.center[2],
             bh2_params.center[0], bh2_params.center[1], bh2_params.center[2]});
#else
        const auto &center = kerr_initial_data.get_params().center;
        get_puncture_tracker().set_puncture_coords(
            {center[0], center[1], center[2]});
#endif
        // can't call start_from_initial_punctures() because we need the full
        // AMR grid first
    }
}

// Calculate RHS during RK4 substeps
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void KerrLevel::specific_eval_rhs(amrex::MultiFab &a_soln,
                                      amrex::MultiFab &a_rhs,
                                      const amrex::Real /*a_time*/)
{
    BL_PROFILE("KerrLevel::specific_eval_rhs()");
    const auto &soln_arrays       = a_soln.arrays();
    const auto &const_soln_arrays = a_soln.const_arrays();
    const auto &rhs_arrays        = a_rhs.arrays();
    const auto soln_ghosts        = a_soln.nGrowVect();

    // The classes to be used
    AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    PositiveChiAndLapse positive_chi_lapse;

    // Enforce positive chi and lapse, det(h)=1 and trace free A
    amrex::ParallelFor(a_soln, soln_ghosts,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           algebraic_constraints_enforcer(ix, iy, iz,
                                                          soln_arrays[box_no]);
                           positive_chi_lapse(ix, iy, iz, soln_arrays[box_no]);
                       });

    // Calculate CCZ4 right hand side using dynamic derivative order
    if (m_evolution_spatial_derivative_order == 4)
    {
        CCZ4RHS<FourthOrderDerivatives> ccz4rhs(Geom().CellSize(0));
        MovingPunctureGauge<FourthOrderDerivatives> moving_puncture_gauge(
            Geom().CellSize(0));

        // NB: These are split up to avoid having to pre-compute all the
        //  first and second derivatives in memory on the GPU at once.

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                ccz4rhs.compute_chi_and_h_ij(ix, iy, iz, rhs_arrays[box_no],
                                             const_soln_arrays[box_no]);
            });

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                ccz4rhs.compute_A_ij_and_Theta_and_Gamma(
                    ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);
            });

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                moving_puncture_gauge.calculate_rhs(
                    ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);

                ccz4rhs.apply_dissipation(ix, iy, iz, rhs_arrays[box_no],
                                          const_soln_arrays[box_no]);
            });
    }
    else if (m_evolution_spatial_derivative_order == 6)
    {
        CCZ4RHS<SixthOrderDerivatives> ccz4rhs(Geom().CellSize(0));
        MovingPunctureGauge<SixthOrderDerivatives> moving_puncture_gauge(
            Geom().CellSize(0));

        // NB: These are split up to avoid having to pre-compute all the
        //  first and second derivatives in memory on the GPU at once.

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                ccz4rhs.compute_chi_and_h_ij(ix, iy, iz, rhs_arrays[box_no],
                                             const_soln_arrays[box_no]);
            });

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                ccz4rhs.compute_A_ij_and_Theta_and_Gamma(
                    ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);
            });

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                moving_puncture_gauge.calculate_rhs(
                    ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);

                ccz4rhs.apply_dissipation(ix, iy, iz, rhs_arrays[box_no],
                                          const_soln_arrays[box_no]);
            });
    }

    amrex::Gpu::streamSynchronize();
}

// enforce algebraic constraints during RK4 substeps
void KerrLevel::specific_update_ode(amrex::MultiFab &a_soln)
{

    AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    const auto soln_ghosts = amrex::IntVect(0); // zero ghost cells

    // Enforce the det(h)=1 and trace free A_ij conditions
    const auto &soln_arrays = a_soln.arrays();
    amrex::ParallelFor(
        a_soln, soln_ghosts,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        { algebraic_constraints_enforcer(ix, iy, iz, soln_arrays[box_no]); });

    amrex::Gpu::streamSynchronize();
}

void KerrLevel::pre_tag_cells()
{
    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto current_time    = get_state_data(state_index).curTime();

    // Fill ghosts for chi to calculate second derivatives
    // 4th-order d2 requires 2 ghost cells
    const int num_ghosts = 2;
    const int num_comps  = 1;

    FillPatch(*this, state_new, num_ghosts, current_time, state_index, c_chi,
              num_comps);
}

void KerrLevel::tag_cells(amrex::TagBoxArray &a_tag_box_array,
                              amrex::Real a_regrid_threshold)
{
    BL_PROFILE("KerrLevel::tag_cells()");
    amrex::MultiFab &state_new = get_new_data(state_index);

    const auto &tag_arrays         = a_tag_box_array.arrays();
    const auto &state_const_arrays = state_new.const_arrays();

    ChiTagger chi_tagger(Geom().CellSize(0), a_regrid_threshold);

    GRParmParse pp;
    spherical_extraction_params_t extraction_params("weyl_extraction");
    extraction_params.fill_params();
    ExtractionTagger extraction_tagger(Geom().CellSize(0), Level(),
                                       extraction_params);

    constexpr auto num_puncture_coords =
        static_cast<std::size_t>(AMREX_SPACEDIM * num_punctures);
    std::array<amrex::Real, num_puncture_coords> puncture_coords{};
    const bool puncture_tracking_enabled =
        get_bh_amr_ptr()->puncture_tracking_enabled();

    if (puncture_tracking_enabled)
    {
        puncture_coords = get_puncture_tracker().get_puncture_coords();
    }

#ifdef USE_TWOPUNCTURES
    TwoPuncturesInitialData two_punctures_initial_data(Geom().CellSize(0));
    two_punctures_initial_data.add_bh_mass_params();
    amrex::Real bh1_mass{};
    amrex::Real bh2_mass{};
    pp.get("bh1.mass", bh1_mass);
    pp.get("bh2.mass", bh2_mass);
    const std::array<amrex::Real, num_punctures> puncture_masses{bh1_mass,
                                                                 bh2_mass};
#else
    KerrID::params_t kerr_params;
    kerr_params.fill_params();
    const std::array<amrex::Real, num_punctures> puncture_masses{
        kerr_params.mass};
#endif

    PunctureTagger<num_punctures> puncture_tagger(
        Geom().CellSize(0), Level(), get_gr_amr_ptr()->maxLevel(),
        puncture_coords, puncture_masses);

    amrex::ParallelFor(state_new, amrex::IntVect(0),
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           chi_tagger(ix, iy, iz, tag_arrays[box_no],
                                      state_const_arrays[box_no]);

                           extraction_tagger(ix, iy, iz, tag_arrays[box_no]);

                           if (puncture_tracking_enabled)
                           {
                               puncture_tagger(ix, iy, iz, tag_arrays[box_no]);
                           }
                       });

    amrex::Gpu::streamSynchronize();
}

void KerrLevel::specific_post_init()
{
    BL_PROFILE("KerrLevel::specific_post_init()");

    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        get_puncture_tracker().start_from_initial_punctures();
    }
}

void KerrLevel::specific_post_restart()
{
    BL_PROFILE("KerrLevel::specific_post_restart()");

    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        std::string restart_checkpoint{};
        GRParmParse pp("amr");
        pp.get("restart", restart_checkpoint);
        get_puncture_tracker().restart(restart_checkpoint);
    }
}

void KerrLevel::specific_post_plotfile(const std::string &a_dir,
                                           std::ostream &a_os)
{
    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        get_puncture_tracker().write_plotfile(a_dir);
    }
}

void KerrLevel::specific_post_checkpoint(const std::string &a_chk_dir,
                                             std::ostream & /*a_os*/)
{
    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        get_puncture_tracker().checkpoint(a_chk_dir);
    }
}

void KerrLevel::specific_post_timestep()
{
    BL_PROFILE("KerrLevel::specific_post_timestep");

    if (get_bh_amr_ptr()->puncture_tracking_enabled())
    {
        GRParmParse puncture_tracking_pp("puncture_tracking");
        int puncture_tracking_level{};
        puncture_tracking_pp.get("level", puncture_tracking_level);
        int puncture_tracking_writeout_level{};
        puncture_tracking_pp.get("writeout_level",
                                 puncture_tracking_writeout_level);

        // do puncture tracking on requested level
        if (Level() == puncture_tracking_level)
        {
            BL_PROFILE("PunctureTracking");

            // only do the write out when we're at at a multiple of the
            // writeout_level
            bool write_punctures =
                at_level_timestep_multiple(puncture_tracking_writeout_level);
            amrex::Real current_time = get_state_data(state_index).curTime();
            amrex::Real dt           = get_gr_amr_ptr()->dtLevel(Level());
            get_puncture_tracker().track(current_time, dt, write_punctures);
        }
    }

    spherical_extraction_params_t extraction_params("weyl_extraction");
    extraction_params.fill_params();

    if (extraction_params.enabled)
    {
        const int min_level = extraction_params.min_extraction_level();
        bool calculate_weyl = at_level_timestep_multiple(min_level);

        if (calculate_weyl && Level() == min_level)
        {
            amrex::Real m_time       = get_state_data(state_index).curTime();
            amrex::Real m_dt         = get_gr_amr_ptr()->dtLevel(Level());
            amrex::Real restart_time = get_gr_amr_ptr()->get_restart_time();
            bool first_step          = (m_time <= m_dt);

            WeylExtraction my_extraction(extraction_params, m_dt, m_time,
                                         first_step, restart_time);
            my_extraction.execute_query(&get_bh_amr_ptr()->m_weyl_interpolator);
        }
    }
}
