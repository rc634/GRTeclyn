/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef SETUP_FUNCTIONS_HPP_
#define SETUP_FUNCTIONS_HPP_
// This file incldues several functions that need to be called to
// set up the runs but aren't very interesting for the normal user.

// xxxxx various setups
#include "FilesystemTools.hpp"
#include "GRAmr.hpp"
#include "GRParmParse.hpp"
#include "IntegrationMethodSetup.hpp"
#include "SimulationParameters.hpp"

#ifdef EQUATION_DEBUG_MODE
#include "DebuggingTools.hpp"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#ifndef GRTECLYN_VERSION
#define GRTECLYN_VERSION "unknown"
#endif

/// Return true when the GRTeclyn-only flag following the AMReX `--` separator
/// requests parameter validation without running the simulation.
[[nodiscard]] inline bool just_check_params()
{
    // AMReX reserves argument 0 for the executable name.
    for (int argument_index = 1;
         argument_index <= amrex::command_argument_count(); ++argument_index)
    {
        if (amrex::get_command_argument(argument_index) == "-check_params")
        {
            return true;
        }
    }
    return false;
}

/// Every rank's stdout goes to <grteclyn.output_path>/pout/pout.<rank>,
/// appended so that a restart continues it (stderr is unchanged);
/// grteclyn.pout = false keeps stdout on the terminal
inline void redirect_stdout_to_pout()
{
    GRParmParse pp("grteclyn");
    bool pout = true;
    pp.queryAdd("pout", pout);
    if (!pout)
    {
        return;
    }
    std::string output_path = ".";
    pp.query("output_path", output_path);
    const std::string dir = output_path + "/pout";
    FilesystemTools::ensure_directory_exists(dir);
    const std::string file =
        dir + "/pout." + std::to_string(amrex::ParallelDescriptor::MyProc());
    std::fflush(stdout);
    if (std::freopen(file.c_str(), "a", stdout) == nullptr)
    {
        amrex::Abort("cannot open " + file);
    }
    std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ); // for tail -f
}

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
/// This function calls MPI_Init
void mainSetup(int argc, char *argv[]);

/// This function calls all finalisations
void mainFinalize();

void mainSetup(int argc, char *argv[])
{
    // NOLINTEND(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    // NOLINTNEXTLINE(bugprone-casting-through-void)
    amrex::Initialize(
        argc, argv, std::function<void()>(SimulationParameters::check_params));

    if (!just_check_params())
    {
        redirect_stdout_to_pout();
    }

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        std::ofstream ofs("parameters_and_version.txt");
        ofs << "Using GRTeclyn version (" << GRTECLYN_VERSION << ")\n\n";
        GRParmParse::prettyPrintTable(ofs);
    }

    amrex::Print() << "GRTeclyn (" << GRTECLYN_VERSION << ") initialized"
                   << '\n';

    if (just_check_params())
    {
        if (GRParmParse::warnings_issued())
        {
            amrex::Print()
                << "User-specified parameter checks were applied and some "
                   "warnings were identified, which should be listed above. "
                   "You can use parameters_and_version.txt to check that the "
                   "parameters were set as you intended. Warnings won't "
                   "prevent the code from running, but you should check that "
                   "you understand why they can be safely ignored before "
                   "continuing.\n";
        }
        else
        {
            amrex::Print()
                << "User-specified parameter checks were applied and no "
                   "warnings or errors were identified. You can use "
                   "parameters_and_version.txt to check that the parameters "
                   "were set as you intended. This does not guarantee that "
                   "your simulation will work, but it is a good start.\n";
        }
    }

#ifdef EQUATION_DEBUG_MODE
    EquationDebugging::check_no_omp();
    amrex::Warning("GRTeclyn is running in equation debug mode. This mode is "
                   "intended only for debugging and leads to significantly "
                   "worse performance.");
#endif

    const int required_argc = 2;
    if (argc < required_argc)
    {
        amrex::Finalize();
        std::cerr << " usage " << argv[0] << " <input_file_name> " << '\n';
        exit(0);
    }
}

void mainFinalize() { amrex::Finalize(); }

#endif /* SETUP_FUNCTIONS_HPP_ */
