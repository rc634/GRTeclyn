/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#include "KerrBinaryLevel.hpp"

#include "AlgebraicConstraintsEnforcer.hpp"
#include "CCZ4RHS.hpp"
#include "ChiTagger.hpp"
#include "Constraints.hpp"
#include "ExtractionTagger.hpp"
#include "FourthOrderDerivatives.hpp"
#include "GammaCalculator.hpp"
#include "KerrBinInitialData.hpp"
#include "PositiveChiAndLapse.hpp"
#include "PunctureTagger.hpp"
#include "PunctureTracker.hpp"
#include "SixthOrderDerivatives.hpp"
#include "Weyl4.hpp"
#include "WeylExtraction.hpp"

#ifdef USE_SAHF
#include "FilesystemTools.hpp"
#include "SmallDataIO.hpp"
#include <sstream>
#endif

#include <algorithm>

BHAmr<KerrBinaryLevel::num_punctures> *KerrBinaryLevel::get_bh_amr_ptr()
{
    return dynamic_cast<BHAmr<num_punctures> *>(get_gr_amr_ptr());
}

PunctureTracker<KerrBinaryLevel::num_punctures> &
KerrBinaryLevel::get_puncture_tracker()
{
    return get_bh_amr_ptr()->get_puncture_tracker();
}

#ifdef USE_SAHF
// sahf.* parameters, x y z f0 (and t0) one value per horizon
static SAHFParams sahf_params(int a_n)
{
    GRParmParse pp("sahf");
    SAHFParams p;
    p.nah = a_n;
    pp.query("lmax", p.lmax);
    p.mmax = p.lmax;
    pp.query("mmax", p.mmax);
    pp.query("nth", p.nth);
    pp.query("nph", p.nph);
    pp.query("ffA", p.ffA);
    pp.query("ffB", p.ffB);
    pp.query("tol", p.tol);
    pp.query("max_iter", p.maxit);
    pp.query("min_r", p.min_r);
    pp.query("max_r", p.max_r);
    pp.getarr("x", p.x0, 0, a_n);
    pp.getarr("y", p.y0, 0, a_n);
    pp.getarr("z", p.z0, 0, a_n);
    pp.getarr("f0", p.f0, 0, a_n);
    p.t0.assign(a_n, 0.);
    pp.queryarr("t0", p.t0, 0, a_n);
    return p;
}

static std::string sahf_dir()
{
    GRParmParse pp;
    std::string path;
    pp.get("grteclyn.output_path", path);
    return path + "/sahf_output";
}

// SmallDataIO adds .dat
static std::string sahf_file(int a_id)
{
    return sahf_dir() + "/horizon_" + std::to_string(a_id);
}

// On a restart, start each horizon from its last row at or before the restart
// time (centre and mean radius): horizons move, the sahf.x/y/z do not. Rows
// after it are removed and redone by sahf_write. Every rank flows the same
// surfaces, so the IO rank reads and broadcasts.
static void sahf_restart(SAHFUnion &a_sahf, amrex::Real a_restart_time)
{
    for (auto &h : a_sahf.ah)
    {
        const std::string file = sahf_file(h.id) + ".dat";
        int exists = static_cast<int>(amrex::ParallelDescriptor::IOProcessor() &&
                                      amrex::FileSystem::Exists(file));
        amrex::ParallelDescriptor::Bcast(
            &exists, 1, amrex::ParallelDescriptor::IOProcessorNumber());
        if (exists == 0)
            continue;
        amrex::Vector<char> text;
        amrex::ParallelDescriptor::ReadAndBcastFile(file, text);
        std::istringstream lines(text.dataPtr());
        std::string line;
        amrex::Real t{}, x{}, y{}, z{}, r{};
        bool have = false;
        while (std::getline(lines, line))
        {
            std::istringstream row(line);
            if (line.find('#') == std::string::npos && row >> t &&
                t < a_restart_time + 1e-8)
            {
                have = static_cast<bool>(row >> x >> y >> z >> r);
            }
        }
        if (have)
        {
            h.surface.x0 = x;
            h.surface.y0 = y;
            h.surface.z0 = z;
            h.surface.init(r);
            amrex::Print() << "SAHF restart: H" << h.id << " from (" << x
                           << ", " << y << ", " << z << "), r " << r << "\n";
        }
    }
}

// one line per found horizon in <output_path>/sahf_output/horizon_<id>.dat;
// _rot: spin from coordinate rotations, _akv: from approximate Killing vectors
static void sahf_write(const SAHFUnion &a_sahf, amrex::Real a_time,
                       amrex::Real a_dt, amrex::Real a_restart_time,
                       bool a_first_call)
{
    if (a_first_call)
        FilesystemTools::ensure_directory_exists(sahf_dir());
    const bool first_step = a_first_call && a_restart_time <= 0.;

    for (const auto &h : a_sahf.ah)
    {
        SmallDataIO file(sahf_file(h.id), a_dt, a_time, a_restart_time,
                         SmallDataIO::APPEND, first_step);
        file.remove_duplicate_time_data();
        if (first_step)
        {
            file.write_header_line(
                {"x", "y", "z", "r_mean", "area", "M_A", "M_T_rot", "chi_rot",
                 "M_T_akv", "chi_akv", "Px", "Py", "Pz", "Sx_rot", "Sy_rot",
                 "Sz_rot", "Sx_akv", "Sy_akv", "Sz_akv", "L2Theta", "steps"});
        }
        if (h.found)
        {
            file.write_time_data_line(std::vector<amrex::Real>{
                h.X[0], h.X[1], h.X[2],
                h.surface.coeffs(0, 0) / (2. * std::sqrt(M_PI)),
                16. * M_PI * h.M_A * h.M_A, h.M_A, h.M_T, h.spin, h.M_T_akv,
                h.spin_akv, h.P[0], h.P[1], h.P[2], h.S[0], h.S[1], h.S[2],
                h.S_akv[0], h.S_akv[1], h.S_akv[2], h.L2,
                static_cast<amrex::Real>(h.iters)});
        }
    }
}
#endif

void KerrBinaryLevel::variableSetUp()
{
    BL_PROFILE("KerrBinaryLevel::variableSetUp()");

    // Set up the state variables
    state_variable_set_up();

    Constraints::set_up(state_index);

    Weyl4::set_up(state_index);
}

// Things to do during the advance step after RK4 steps
void KerrBinaryLevel::specific_advance()
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

// Initial data from a kerrbin .kid file (KerrBinInitialData), evaluated on the
// host and copied to the device, then Gamma^i by finite differences of h_ij
void KerrBinaryLevel::initData()
{
    BL_PROFILE("KerrBinaryLevel::initData");
    if (get_gr_amr_ptr()->Verbose() > 0)
    {
        amrex::Print() << "KerrBinaryLevel::initData " << Level() << "\n";
    }

    // All components except Gamma^i, in the valid and ghost cells
    const amrex::Real dx       = Geom().CellSize(0);
    amrex::MultiFab &state_new = get_new_data(state_index);
    KerrBinInitialData::set_state(state_new, dx);

    // Gamma^i = h^jk Gamma^i_jk from finite differences of h_ij (non-zero:
    // the conformal metric is not flat). Valid cells only: their stencils read
    // the ghost cells set above, so no FillBoundary is needed. Same derivative
    // order as the evolution. Separate kernel since it needs h_ij at
    // neighbouring cells.
    const auto &state_arrays = state_new.arrays();
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
    amrex::Gpu::streamSynchronize();

    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        // need to set the puncture coordinates as we use it for the puncture
        // tagging
        const auto coords = KerrBinInitialData::puncture_coords();
        amrex::Array<amrex::ParticleReal, AMREX_SPACEDIM * num_punctures>
            puncture_coords{};
        std::copy(coords.begin(), coords.end(), puncture_coords.begin());
        get_puncture_tracker().set_puncture_coords(puncture_coords);
        // can't call start_from_initial_punctures() because we need the full
        // AMR grid first
    }
}

// Calculate RHS during RK4 substeps
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void KerrBinaryLevel::specific_eval_rhs(amrex::MultiFab &a_soln,
                                      amrex::MultiFab &a_rhs,
                                      const amrex::Real /*a_time*/)
{
    BL_PROFILE("KerrBinaryLevel::specific_eval_rhs()");
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
void KerrBinaryLevel::specific_update_ode(amrex::MultiFab &a_soln)
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

void KerrBinaryLevel::pre_tag_cells()
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

void KerrBinaryLevel::tag_cells(amrex::TagBoxArray &a_tag_box_array,
                              amrex::Real a_regrid_threshold)
{
    BL_PROFILE("KerrBinaryLevel::tag_cells()");
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

    // bare masses from the .kid file
    const std::array<amrex::Real, num_punctures> puncture_masses =
        KerrBinInitialData::puncture_masses();

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

void KerrBinaryLevel::specific_post_init()
{
    BL_PROFILE("KerrBinaryLevel::specific_post_init()");

    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        get_puncture_tracker().start_from_initial_punctures();
    }
}

void KerrBinaryLevel::specific_post_restart()
{
    BL_PROFILE("KerrBinaryLevel::specific_post_restart()");

    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        std::string restart_checkpoint{};
        GRParmParse pp("amr");
        pp.get("restart", restart_checkpoint);
        get_puncture_tracker().restart(restart_checkpoint);
    }
}

void KerrBinaryLevel::specific_post_plotfile(const std::string &a_dir,
                                           std::ostream &a_os)
{
    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        get_puncture_tracker().write_plotfile(a_dir);
    }
}

void KerrBinaryLevel::specific_post_checkpoint(const std::string &a_chk_dir,
                                             std::ostream & /*a_os*/)
{
    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        get_puncture_tracker().checkpoint(a_chk_dir);
    }
}

void KerrBinaryLevel::specific_post_timestep()
{
    BL_PROFILE("KerrBinaryLevel::specific_post_timestep");

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

#ifdef USE_SAHF
    // horizon finders on every time step of level sahf.level, each starting
    // from its last solution (printed by the IO rank)
    GRParmParse sahf_pp("sahf");
    int sahf_n     = 0;
    int sahf_level = 2;
    sahf_pp.query("n", sahf_n);
    sahf_pp.query("level", sahf_level);
    if (sahf_n > 0 && Level() == sahf_level)
    {
        BL_PROFILE("SAHF");
        auto &sahf            = get_bh_amr_ptr()->m_sahf;
        const bool first_call = !sahf;
        if (first_call)
        {
            sahf = std::make_unique<SAHFUnion>(sahf_params(sahf_n),
                                               get_gr_amr_ptr());
            const amrex::Real restart_time =
                get_gr_amr_ptr()->get_restart_time();
            if (restart_time > 0.)
                sahf_restart(*sahf, restart_time);
        }
        const amrex::Real time = get_state_data(state_index).curTime();
        sahf->run(time, amrex::ParallelDescriptor::IOProcessor());
        sahf_write(*sahf, time, get_gr_amr_ptr()->dtLevel(Level()),
                   get_gr_amr_ptr()->get_restart_time(), first_call);
    }
#endif
}
