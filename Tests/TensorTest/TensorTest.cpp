/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

// Doctest header
#include "doctest.h"

// Test include
#include "TensorTest.hpp"

// Our includes
#include "DimensionDefinitions.hpp"
#include "Tensor.hpp"

// Scalar constructors and scalar assignment must set every element
void run_tensor_test()
{
    // Rank3 scalar assignment (its loop condition used to test the wrong
    // index and never terminate)
    Tensor::Rank3 rank3{0.};
    rank3 = 2.5;
    FOR (i, j, k)
    {
        INFO("Rank3 operator= component [" << i << "][" << j << "][" << k
                                           << "]");
        CHECK(rank3(i, j, k) == 2.5);
    }

    // Rank4 scalar constructor and assignment (they used to skip the last
    // element)
    Tensor::Rank4 rank4(1.5);
    FOR (i, j, k, l)
    {
        INFO("Rank4 constructor component [" << i << "][" << j << "][" << k
                                             << "][" << l << "]");
        CHECK(rank4(i, j, k, l) == 1.5);
    }

    rank4 = -3.0;
    FOR (i, j, k, l)
    {
        INFO("Rank4 operator= component [" << i << "][" << j << "][" << k
                                           << "][" << l << "]");
        CHECK(rank4(i, j, k, l) == -3.0);
    }

    Tensor::SpacetimeRank4 spacetime_rank4(4.0);
    for (int i = 0; i < SPACETIME_DIM; ++i)
    {
        for (int j = 0; j < SPACETIME_DIM; ++j)
        {
            for (int k = 0; k < SPACETIME_DIM; ++k)
            {
                for (int l = 0; l < SPACETIME_DIM; ++l)
                {
                    INFO("SpacetimeRank4 constructor component ["
                         << i << "][" << j << "][" << k << "][" << l << "]");
                    CHECK(spacetime_rank4(i, j, k, l) == 4.0);
                }
            }
        }
    }
}
