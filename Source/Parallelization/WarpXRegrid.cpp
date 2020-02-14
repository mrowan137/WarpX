/* Copyright 2019 Andrew Myers, Ann Almgren, Axel Huebl
 * David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang, levinem
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include <WarpX.H>
#include <AMReX_BLProfiler.H>

using namespace amrex;

std::unique_ptr< Vector<std::unique_ptr<amrex::MultiFab> > > m_costs;

void
WarpX::LoadBalance (WarpX::LBType load_balance_type/*=DEFAULT*/)
{
    BL_PROFILE_REGION("LoadBalance");
    BL_PROFILE("WarpX::LoadBalance()");

    switch (load_balance_type)
    {
    case DEFAULT:
        m_costs.reset(&costs);
        break;
    case HEURISTIC:
        for (int lev = 0; lev <= finest_level; ++lev)
        {
            WarpX::ComputeCostsHeuristic(lev);
        }
        m_costs.reset(&costs_heuristic);
        break;
    default:
        amrex::Error("Bad WarpX::LBType;  check warpx.load_balance_type in input file?");
    }

    AMREX_ALWAYS_ASSERT((*m_costs)[0] != nullptr);

    const int nLevels = finestLevel();
    for (int lev = 0; lev <= nLevels; ++lev)
    {
        const Real nboxes = (*m_costs)[lev]->size();
        const Real nprocs = ParallelDescriptor::NProcs();
        const int nmax = static_cast<int>(std::ceil(nboxes/nprocs*load_balance_knapsack_factor));
        const DistributionMapping newdm = (load_balance_with_sfc)
            ? DistributionMapping::makeSFC(*(*m_costs)[lev], false)
            : DistributionMapping::makeKnapSack(*(*m_costs)[lev], nmax);
        RemakeLevel(lev, t_new[lev], boxArray(lev), newdm);
    }

    mypc->Redistribute();

    m_costs.release();
}

void
WarpX::RemakeLevel (int lev, Real time, const BoxArray& ba, const DistributionMapping& dm)
{
    if (ba == boxArray(lev))
    {
        if (ParallelDescriptor::NProcs() == 1) return;

#ifdef WARPX_DO_ELECTROSTATIC
        AMREX_ALWAYS_ASSERT(masks[lev] == nullptr);
        AMREX_ALWAYS_ASSERT(gather_masks[lev] == nullptr);
#endif // WARPX_DO_ELECTROSTATIC

        // Fine patch

        const auto& period = Geom(lev).periodicity();
        for (int idim=0; idim < 3; ++idim)
        {
            {
                const IntVect& ng = Bfield_fp[lev][idim]->nGrowVect();
                auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Bfield_fp[lev][idim]->boxArray(),
                                                                  dm, Bfield_fp[lev][idim]->nComp(), ng));
                pmf->Redistribute(*Bfield_fp[lev][idim], 0, 0, Bfield_fp[lev][idim]->nComp(), ng);
                Bfield_fp[lev][idim] = std::move(pmf);
            }
            {
                const IntVect& ng = Efield_fp[lev][idim]->nGrowVect();
                auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Efield_fp[lev][idim]->boxArray(),
                                                                  dm, Efield_fp[lev][idim]->nComp(), ng));
                pmf->Redistribute(*Efield_fp[lev][idim], 0, 0, Efield_fp[lev][idim]->nComp(), ng);
                Efield_fp[lev][idim] = std::move(pmf);
            }
            {
                const IntVect& ng = current_fp[lev][idim]->nGrowVect();
                auto pmf = std::unique_ptr<MultiFab>(new MultiFab(current_fp[lev][idim]->boxArray(),
                                                                  dm, current_fp[lev][idim]->nComp(), ng));
                current_fp[lev][idim] = std::move(pmf);
            }
            if (current_store[lev][idim])
            {
                const IntVect& ng = current_store[lev][idim]->nGrowVect();
                auto pmf = std::unique_ptr<MultiFab>(new MultiFab(current_store[lev][idim]->boxArray(),
                                                                  dm, current_store[lev][idim]->nComp(), ng));
                // no need to redistribute
                current_store[lev][idim] = std::move(pmf);
            }
        }

        if (F_fp[lev] != nullptr) {
            const IntVect& ng = F_fp[lev]->nGrowVect();
            auto pmf = std::unique_ptr<MultiFab>(new MultiFab(F_fp[lev]->boxArray(),
                                                              dm, F_fp[lev]->nComp(), ng));
            pmf->Redistribute(*F_fp[lev], 0, 0, F_fp[lev]->nComp(), ng);
            F_fp[lev] = std::move(pmf);
        }

        if (rho_fp[lev] != nullptr) {
            const int nc = rho_fp[lev]->nComp();
            const IntVect& ng = rho_fp[lev]->nGrowVect();
            auto pmf = std::unique_ptr<MultiFab>(new MultiFab(rho_fp[lev]->boxArray(),
                                                              dm, nc, ng));
            rho_fp[lev] = std::move(pmf);
        }

        // Aux patch

        if (lev == 0 && Bfield_aux[0][0]->ixType() == Bfield_fp[0][0]->ixType())
        {
            for (int idim = 0; idim < 3; ++idim) {
                Bfield_aux[lev][idim].reset(new MultiFab(*Bfield_fp[lev][idim], amrex::make_alias, 0, Bfield_aux[lev][idim]->nComp()));
                Efield_aux[lev][idim].reset(new MultiFab(*Efield_fp[lev][idim], amrex::make_alias, 0, Efield_aux[lev][idim]->nComp()));
            }
        } else {
            for (int idim=0; idim < 3; ++idim)
            {
                {
                    const IntVect& ng = Bfield_aux[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Bfield_aux[lev][idim]->boxArray(),
                                                                      dm, Bfield_aux[lev][idim]->nComp(), ng));
                    // pmf->Redistribute(*Bfield_aux[lev][idim], 0, 0, Bfield_aux[lev][idim]->nComp(), ng);
                    Bfield_aux[lev][idim] = std::move(pmf);
                }
                {
                    const IntVect& ng = Efield_aux[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Efield_aux[lev][idim]->boxArray(),
                                                                      dm, Efield_aux[lev][idim]->nComp(), ng));
                    // pmf->Redistribute(*Efield_aux[lev][idim], 0, 0, Efield_aux[lev][idim]->nComp(), ng);
                    Efield_aux[lev][idim] = std::move(pmf);
                }
            }
        }

        // Coarse patch
        if (lev > 0) {
            const auto& cperiod = Geom(lev-1).periodicity();
            for (int idim=0; idim < 3; ++idim)
            {
                {
                    const IntVect& ng = Bfield_cp[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Bfield_cp[lev][idim]->boxArray(),
                                                                      dm, Bfield_cp[lev][idim]->nComp(), ng));
                    pmf->Redistribute(*Bfield_cp[lev][idim], 0, 0, Bfield_cp[lev][idim]->nComp(), ng);
                    Bfield_cp[lev][idim] = std::move(pmf);
                }
                {
                    const IntVect& ng = Efield_cp[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Efield_cp[lev][idim]->boxArray(),
                                                                      dm, Efield_cp[lev][idim]->nComp(), ng));
                    pmf->Redistribute(*Efield_cp[lev][idim], 0, 0, Efield_cp[lev][idim]->nComp(), ng);
                    Efield_cp[lev][idim] = std::move(pmf);
                }
                {
                    const IntVect& ng = current_cp[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>( new MultiFab(current_cp[lev][idim]->boxArray(),
                                                                       dm, current_cp[lev][idim]->nComp(), ng));
                    current_cp[lev][idim] = std::move(pmf);
                }
            }

            if (F_cp[lev] != nullptr) {
                const IntVect& ng = F_cp[lev]->nGrowVect();
                auto pmf = std::unique_ptr<MultiFab>(new MultiFab(F_cp[lev]->boxArray(),
                                                                  dm, F_cp[lev]->nComp(), ng));
                pmf->Redistribute(*F_cp[lev], 0, 0, F_cp[lev]->nComp(), ng);
                F_cp[lev] = std::move(pmf);
            }

            if (rho_cp[lev] != nullptr) {
                const int nc = rho_cp[lev]->nComp();
                const IntVect& ng = rho_cp[lev]->nGrowVect();
                auto pmf = std::unique_ptr<MultiFab>(new MultiFab(rho_cp[lev]->boxArray(),
                                                                  dm, nc, ng));
                rho_cp[lev] = std::move(pmf);
            }
        }

        if (lev > 0 && (n_field_gather_buffer > 0 || n_current_deposition_buffer > 0)) {
            for (int idim=0; idim < 3; ++idim)
            {
                if (Bfield_cax[lev][idim])
                {
                    const IntVect& ng = Bfield_cax[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Bfield_cax[lev][idim]->boxArray(),
                                                                      dm, Bfield_cax[lev][idim]->nComp(), ng));
                    // pmf->ParallelCopy(*Bfield_cax[lev][idim], 0, 0, Bfield_cax[lev][idim]->nComp(), ng, ng);
                    Bfield_cax[lev][idim] = std::move(pmf);
                }
                if (Efield_cax[lev][idim])
                {
                    const IntVect& ng = Efield_cax[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>(new MultiFab(Efield_cax[lev][idim]->boxArray(),
                                                                      dm, Efield_cax[lev][idim]->nComp(), ng));
                    // pmf->ParallelCopy(*Efield_cax[lev][idim], 0, 0, Efield_cax[lev][idim]->nComp(), ng, ng);
                    Efield_cax[lev][idim] = std::move(pmf);
                }
                if (current_buf[lev][idim])
                {
                    const IntVect& ng = current_buf[lev][idim]->nGrowVect();
                    auto pmf = std::unique_ptr<MultiFab>(new MultiFab(current_buf[lev][idim]->boxArray(),
                                                                      dm, current_buf[lev][idim]->nComp(), ng));
                    // pmf->ParallelCopy(*current_buf[lev][idim], 0, 0, current_buf[lev][idim]->nComp(), ng, ng);
                    current_buf[lev][idim] = std::move(pmf);
                }
            }
            if (charge_buf[lev])
            {
                const IntVect& ng = charge_buf[lev]->nGrowVect();
                auto pmf = std::unique_ptr<MultiFab>(new MultiFab(charge_buf[lev]->boxArray(),
                                                                  dm, charge_buf[lev]->nComp(), ng));
                // pmf->ParallelCopy(*charge_buf[lev][idim], 0, 0, charge_buf[lev]->nComp(), ng, ng);
                charge_buf[lev] = std::move(pmf);
            }
            if (current_buffer_masks[lev])
            {
                const IntVect& ng = current_buffer_masks[lev]->nGrowVect();
                auto pmf = std::unique_ptr<iMultiFab>(new iMultiFab(current_buffer_masks[lev]->boxArray(),
                                                                    dm, current_buffer_masks[lev]->nComp(), ng));
                // pmf->ParallelCopy(*current_buffer_masks[lev], 0, 0, current_buffer_masks[lev]->nComp(), ng, ng);
                current_buffer_masks[lev] = std::move(pmf);
            }
            if (gather_buffer_masks[lev])
            {
                const IntVect& ng = gather_buffer_masks[lev]->nGrowVect();
                auto pmf = std::unique_ptr<iMultiFab>(new iMultiFab(gather_buffer_masks[lev]->boxArray(),
                                                                    dm, gather_buffer_masks[lev]->nComp(), ng));
                // pmf->ParallelCopy(*gather_buffer_masks[lev], 0, 0, gather_buffer_masks[lev]->nComp(), ng, ng);
                gather_buffer_masks[lev] = std::move(pmf);
            }
        }

        if (costs[lev] != nullptr) {
            costs[lev].reset(new MultiFab(costs[lev]->boxArray(), dm, 1, 0));
            costs[lev]->setVal(0.0);
        }

    if (costs_heuristic[lev] != nullptr) {
            costs_heuristic[lev].reset(new MultiFab(costs_heuristic[lev]->boxArray(), dm, 1, 0));
            costs_heuristic[lev]->setVal(0.0);
        }

        SetDistributionMap(lev, dm);
    }
    else
    {
        amrex::Abort("RemakeLevel: to be implemented");
    }
}

void
WarpX::ComputeCostsHeuristic (int lev)
{
    MultiFab* cost = WarpX::getCostsHeuristic(lev);
    auto & mypc = WarpX::GetInstance().GetPartContainer();
    auto nSpecies = mypc.nSpecies();
    const IntVect& ng = current_fp[lev][0]->nGrowVect();

    // Species loop
    for (int i_s = 0; i_s < nSpecies; ++i_s)
    {
        auto & myspc = mypc.GetParticleContainer(i_s);

        // Particle loop
        for (WarpXParIter pti(myspc, lev); pti.isValid(); ++pti)
        {
            if (cost)
            {
                const Box& tbx = pti.tilebox();
                const long np = pti.numParticles();
                Real wt = (costs_heuristic_prtl_wt*(1.0*np)/tbx.d_numPts()); // prtl wt per (work) cell
                Array4<Real> const& costarr = cost->array(pti);
                amrex::ParallelFor(
                                   tbx,
                                   [=] AMREX_GPU_DEVICE (int i, int j, int k)
                                   {
                                       costarr(i,j,k) += wt;
                                   }
                                  );
            }
        }
    }

    // Cell loop
    for (MFIter mfi(*cost); mfi.isValid(); ++mfi)
    {
        if (cost) {
        const Box& tbx = mfi.tilebox();
    const Box& grown_tbx = mfi.growntilebox(ng);
    int tbx_size = 1;
    int grown_tbx_size = 1;

    // Compute n_cells for each box
    for (auto el : tbx.size())
    {
        tbx_size *= el;
    }

    for (auto el : grown_tbx.size())
    {
        grown_tbx_size *= el;
    }

        Real wt = (costs_heuristic_cell_wt*1.0*grown_tbx_size/tbx_size); // cell wt per (work) cell
        Array4<Real> const& costarr = cost->array(mfi);
        amrex::ParallelFor(
                           tbx,
                           [=] AMREX_GPU_DEVICE (int i, int j, int k)
                           {
                               costarr(i,j,k) += wt;
                           }
                          );
        }
    }
}

amrex::Real
WarpX::sumMultiFabOverLevs (const amrex::Vector<std::unique_ptr<amrex::MultiFab> >& VectorOfMultifabs)
{
    // Sum (over physical region of) MultiFab values over all levels
    // Can be used, e.g., to check the sum over levels of 'costs' or 'costs_heuristic'
    amrex::Real total = 0.0;

    const int nLevels = finestLevel();
    for (int lev = 0; lev <= nLevels; ++lev)
    {
        MultiFab* mf = VectorOfMultifabs[lev].get();
        for (MFIter mfi(*mf); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);
            Array4<Real> const& mfArray = (&mf)[lev]->array(mfi);
            for (int k = lo.z; k <= hi.z; ++k) {
                for (int j = lo.y; j <= hi.y; ++j) {
                    for (int i = lo.x; i <= hi.x; ++i) {
                        total += mfArray(i, j, k);
                    }
                }
            }
        }
    }

    ParallelDescriptor::ReduceRealSum(total);
    return total;
}
