/* Copyright 2019 Andrew Myers, Ann Almgren, Axel Huebl
 * David Grote, Maxence Thevenet, Remi Lehe
 * Weiqun Zhang, levinem
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include <WarpX.H>
#include <WarpXAlgorithmSelection.H>
#include <AMReX_BLProfiler.H>

#include <set>

#define MPI_CHECK(cmd) {int error = cmd; if(error!=MPI_SUCCESS){ printf("<%s>:%i ",__FILE__,__LINE__); throw std::runtime_error(std::string("[MPI] Error")); }}

using namespace amrex;

void
WarpX::LoadBalance ()
{
    WARPX_PROFILE_REGION("LoadBalance");
    WARPX_PROFILE("WarpX::LoadBalance()");

    if (WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
    {
        LoadBalanceTimers();
    } else if (WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Heuristic)
    {
        LoadBalanceHeuristic();
    }
}

void
WarpX::LoadBalanceTimers ()
{
    AMREX_ALWAYS_ASSERT(costs[0] != nullptr);

    const int nLevels = finestLevel();
    for (int lev = 0; lev <= nLevels; ++lev)
    {
        const Real nboxes = costs[lev]->size();
        const Real nprocs = ParallelDescriptor::NProcs();
        const int nmax = static_cast<int>(std::ceil(nboxes/nprocs*load_balance_knapsack_factor));
        const DistributionMapping newdm = (load_balance_with_sfc)
            ? DistributionMapping::makeSFC(*costs[lev], false)
            : DistributionMapping::makeKnapSack(*costs[lev], nmax);
        RemakeLevel(lev, t_new[lev], boxArray(lev), newdm);
    }
    mypc->Redistribute();
}

void
WarpX::LoadBalanceHeuristic ()
{
    AMREX_ALWAYS_ASSERT(costs_heuristic[0] != nullptr);
    //WarpX::ComputeCostsHeuristic(costs_heuristic);

    const int nLevels = finestLevel();
    for (int lev = 0; lev <= nLevels; ++lev)
    {
#ifdef AMREX_USE_MPI
        const DistributionMapping& currdm = DistributionMap(lev);
                
        // Hacking way to get hosts
        std::unique_ptr<Vector<Real> > hosts;
        hosts.reset(new Vector<Real>);
        hosts->resize(costs_heuristic[lev]->size());
        MultiFab* Ex = Efield_fp[lev][0].get();
        for (MFIter mfi(*Ex, false); mfi.isValid(); ++mfi)
        {
            const Box& tbx = mfi.tilebox();
            char hostname[MPI_MAX_PROCESSOR_NAME];
            int length;
            MPI_CHECK( MPI_Get_processor_name( hostname, &length ) );
            //amrex::AllPrint() << "I am : " << ParallelDescriptor::MyProc() << "; hostname: " << hostname << "\n";
            //amrex::AllPrint() << "HOSTNAME: "  << hostname
            //                  << "\n";
            std::string hostnameString(hostname);
            (*hosts)[mfi.index()] = float(std::stoi(hostnameString.erase(0, 1).erase(2, 1))); // hostname converted
        }
        
        // Parallel reduce to IO proc and get data over all procs
        amrex::Vector<Real>::iterator it = (*hosts).begin();
        amrex::Real* itAddr = &(*it);
        ParallelAllReduce::Sum(itAddr,
                               hosts->size(),
                               ParallelContext::CommunicatorSub());

        // Now the host ID are filled complete in hosts
        // Knowing the
        //     mfi.index()-->host
        //     mfi.index()-->proc
        //     ==> proc --> host
        // for partitions, reassign proc to mfi.index() 
        //     mfi.index()-->proc
        // so that so the mfi in same 3x2 partition have the same node.
        // Then you can remake a level.

        // Everyone knows now the mfi.index space --> node mapping
        
        // 1. Get the unique hosts
        std::set<Real> unique_hosts_set;
        for (auto h : (*hosts))
        {
            unique_hosts_set.insert(h);
        }

        std::unique_ptr< Vector<Real> > unique_hosts;
        unique_hosts.reset(new Vector<Real>);
        for (std::set<Real>::iterator it=unique_hosts_set.begin(); it!=unique_hosts_set.end(); ++it)
        {
            unique_hosts->push_back(*it);
        }

        
        // 2. make a dict of unique_hosts = {h1:[], h2:[], h3:[], ... } from unique hosts --> ranks
        std::map<amrex::Real, std::unique_ptr<Vector<amrex::Real> > > hostsToProcs;
        for (auto& u : (*unique_hosts))
        {
            std::unique_ptr<Vector<amrex::Real> > procs;
            hostsToProcs[u].reset(new Vector<amrex::Real>);
        }

        int blocking_factor = blockingFactor(lev)[0];

        int ind = 0;
        for (auto& h : (*hosts))
        {
            (*hostsToProcs[ h ]).push_back(currdm[ind]);
            ind+=1;
        }
        
        //amrex::AllPrint() << currdm;
        
        // for (MFIter mfi(*Ex, false); mfi.isValid(); ++mfi)
        // {
        //     const Box& tbx = mfi.tilebox();
        //     int i = tbx.loVect()[0]/blocking_factor;
        //     int j = tbx.loVect()[1]/blocking_factor;
        //     int k = tbx.loVect()[2]/blocking_factor;

        //     int i_rel = i%3;
        //     int j_rel = j%2;
            
        //     (*hostsToProcs[ (*hosts)[mfi.index()] ])[3*j_rel + i_rel] = 1.*currdm[mfi.index()];
        //     amrex::AllPrint() << "MyProc: " << ParallelDescriptor::MyProc()
        //                       << "; proc " << 1.*currdm[mfi.index()]
        //                       << " in loc "<< 3*j_rel + i_rel
        //                       << " (" << i << j << k << ") " << " (" << i_rel << j_rel << ") "
        //                       << " for host "
        //                       << (*hosts)[mfi.index()] << "\n";
        // }

        // for (auto& u : (*unique_hosts))
        // {
        //     amrex::AllPrint() << "MyProc: " << ParallelDescriptor::MyProc() << " ";
        //     for (auto x: (*hostsToProcs[u]))
        //     {
        //         amrex::AllPrint() << x << " ";
        //     }
        //     amrex::AllPrint() << "\n";
        // }
        
        
        // Reduce the hostToProce
        // for (auto& u : (*unique_hosts))
        // {
        //     //amrex::Print() << "It's a unique host; " << u << "\n";
        //     Vector<Real>::iterator it = (*hostsToProcs[u]).begin();
        //     Real* itAddr = &(*it);
        //     ParallelAllReduce::Sum(itAddr,
        //                            hostsToProcs[u]->size(),
        //                            ParallelContext::CommunicatorSub());
        // }
        
        // 3. Designate positions f(i,j,k) --> h1
        //    strategy: zigzag through the domain
        const auto dx = geom[lev].CellSizeArray();
        const RealBox& real_box = geom[lev].ProbDomain();
        int mx, my, mz;
        mx = int( (real_box.hi(0) - real_box.lo(0))/(dx[0]*blocking_factor) );
        my = int( (real_box.hi(1) - real_box.lo(1))/(dx[1]*blocking_factor) );
        mz = int( (real_box.hi(2) - real_box.lo(2))/(dx[2]*blocking_factor) );
        int kjiToHost[mz][my][mx] = {0};

        int uh_ind = 0;
        //amrex::Print() << "(mx, my, mz)" << "(" << mx << ", " << my << ", " << mz << ")\n";
        for (int k=0; k<mz; k++)
        {
            // Snake through the i j plane
            int j_max = 1;
            int toggle = 1;
            for (int j_mult=1; j_mult<=my/2; j_mult++)
            {
                j_max = 2*j_mult - 1;
                int j = j_max - 1;

                int i = 0;
                while (i<mx)
                {
                    //amrex::Print() << "Setting: " << (*unique_hosts)[uh_ind] << "\n";
                    
                    kjiToHost[k][j][i] = (*unique_hosts)[uh_ind];
                    //amrex::AllPrint() << "(k,j,i)" << "(" << k << ", " << j << ", " << i << ") -->" << kjiToHost[k][j][i] << "\n";
                    j+=toggle;
                    kjiToHost[k][j][i] = (*unique_hosts)[uh_ind];
                    //amrex::AllPrint() << "(k,j,i)" << "(" << k << ", " << j << ", " << i << ") -->" << kjiToHost[k][j][i] << "\n";
                    i+=1;
                    kjiToHost[k][j][i] = (*unique_hosts)[uh_ind];
                    //amrex::AllPrint() << "(k,j,i)" << "(" << k << ", " << j << ", " << i << ") -->" << kjiToHost[k][j][i] << "\n";
                    j-=toggle;
                    kjiToHost[k][j][i] = (*unique_hosts)[uh_ind];
                    //amrex::AllPrint() << "(k,j,i)" << "(" << k << ", " << j << ", " << i << ") -->" << kjiToHost[k][j][i] << "\n";
                    i+=1;
                    kjiToHost[k][j][i] = (*unique_hosts)[uh_ind];
                    //amrex::AllPrint() << "(k,j,i)" << "(" << k << ", " << j << ", " << i << ") -->" << kjiToHost[k][j][i] << "\n";
                    j+=toggle;
                    kjiToHost[k][j][i] = (*unique_hosts)[uh_ind];
                    //amrex::AllPrint() << "(k,j,i)" << "(" << k << ", " << j << ", " << i << ") -->" << kjiToHost[k][j][i] << "\n";
                    i+=1;

                    toggle *= -1;
                    uh_ind += 1;
                }
            }
        }

        // for (int k=0; k<mz; k++)
        // {
        //     for (int j=0; j<my; j++)
        //     {
        //         for (int i=0; i<mx; i++)
        //         {
        //             // amrex::AllPrint() << "MyProc: " << ParallelDescriptor::MyProc() << " "
        //             //                   << " ijk " << i << " " << j << " " << k << " "
        //             //                   << kjiToHost[k][j][i] << "\n";
        //         }
        //     }
        // }
        // 4. Loop over keys of the distribution mapping; get mfi position; pop from dict to get
        //        f(mfi.position) --> hX
        //        dm[mfi.index()] = unique_hosts[hX].pop()
        //amrex::Print() << "--------";
        int pmapArray[costs_heuristic[lev]->size()] = {0};
        int n = sizeof(pmapArray) / sizeof(pmapArray[0]);
        Vector<int> pmap(pmapArray, pmapArray + n);
        for (MFIter mfi(*Ex, false); mfi.isValid(); ++mfi)
        {
            const Box& tbx = mfi.tilebox();
            int i = tbx.loVect()[0]/blocking_factor;
            int j = tbx.loVect()[1]/blocking_factor;
            int k = tbx.loVect()[2]/blocking_factor;

            int i_rel = i%3;
            int j_rel = j%2;

            //amrex::AllPrint() << "(k,j,i)" << "(" << k << ", " << j << ", " << i << ") -->" << kjiToHost[k][j][i] << "\n";

            // Every box gets a different proc
            // for (auto x: (*hostsToProcs[kjiToHost[k][j][i]]))
            // {
            //     amrex::Print() << ParallelDescriptor::MyProc() << ": " << x <<"\n";
            // }
            pmap[mfi.index()] = (*hostsToProcs[kjiToHost[k][j][i]])[3*j_rel + i_rel];
            // amrex::AllPrint() << "Asign to pos " << mfi.index()
            //                << " the rank " << (*hostsToProcs[kjiToHost[k][j][i]])[3*j_rel + i_rel]
            //                << " my local position is " << 3*j_rel + i_rel << "\n";
            //hostsToProcs[kjiToHost[k][j][i]]->pop_back();            
        }
        Vector<int>::iterator itInt = pmap.begin();
        int* itIntAddr = &(*itInt);
        ParallelAllReduce::Sum(itIntAddr,
                               pmap.size(),
                               ParallelContext::CommunicatorSub());
        
        DistributionMapping newdm(pmap);
        //amrex::Print() << "Make the new !!!!!";
        //amrex::Print() << newdm;
        //RemakeLevel(lev, t_new[lev], boxArray(lev), newdm);

        //newdm = DistributionMapping::makeSFC(*costs_heuristic[lev], boxArray(lev), false);

        //amrex::Print() << newdm;

        RemakeLevel(lev, t_new[lev], boxArray(lev), newdm);

        
        // Useful printing
        // for (auto& x : hostsToProcs)
        // {
        //     for ( auto& y : (*x.second) )
        //     {
        //         amrex::AllPrint() << x.first << " has " << y << "\n";
        //     }
        // }
        
        // Parallel reduce the costs_heurisitc
        //amrex::Vector<Real>::iterator it = (*costs_heuristic[lev]).begin();
        //amrex::Real* itAddr = &(*it);
        //ParallelAllReduce::Sum(itAddr,
        //                      costs_heuristic[lev]->size(),
        //                       ParallelContext::CommunicatorSub());
#endif
        //const amrex::Real nboxes = costs_heuristic[lev]->size();
        //const amrex::Real nprocs = ParallelContext::NProcsSub();
        //const int nmax = static_cast<int>(std::ceil(nboxes/nprocs*load_balance_knapsack_factor));

        // const DistributionMapping newdm = (load_balance_with_sfc)
        //     ? DistributionMapping::makeSFC(*costs_heuristic[lev], boxArray(lev), false)
        //     : DistributionMapping::makeKnapSack(*costs_heuristic[lev], nmax);

        // RemakeLevel(lev, t_new[lev], boxArray(lev), newdm);
    }
    mypc->Redistribute();
}

void
WarpX::RemakeLevel (int lev, Real /*time*/, const BoxArray& ba, const DistributionMapping& dm)
{
    if (ba == boxArray(lev))
    {
        if (ParallelDescriptor::NProcs() == 1) return;

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

        if (WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            if (costs[lev] != nullptr)
            {
                costs[lev].reset(new MultiFab(costs[lev]->boxArray(), dm, 1, 0));
                costs[lev]->setVal(0.0);
            }
        } else if (WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Heuristic)
        {
            if (costs_heuristic[lev] != nullptr)
            {
                costs_heuristic[lev].reset(new amrex::Vector<Real>);
                const int nboxes = Efield_fp[lev][0].get()->size();
                costs_heuristic[lev]->resize(nboxes, 0.0); // Initializes to 0.0?
            }
        }

        SetDistributionMap(lev, dm);

    } else
    {
        amrex::Abort("RemakeLevel: to be implemented");
    }
}

void
WarpX::ComputeCostsHeuristic (amrex::Vector<std::unique_ptr<amrex::Vector<amrex::Real> > >& costs)
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        auto & mypc = WarpX::GetInstance().GetPartContainer();
        auto nSpecies = mypc.nSpecies();

        // Species loop
        for (int i_s = 0; i_s < nSpecies; ++i_s)
        {
            auto & myspc = mypc.GetParticleContainer(i_s);

            // Particle loop
            for (WarpXParIter pti(myspc, lev); pti.isValid(); ++pti)
            {
                (*costs[lev])[pti.index()] += costs_heuristic_particles_wt*pti.numParticles();
            }
        }

        //Cell loop
        MultiFab* Ex = Efield_fp[lev][0].get();
        for (MFIter mfi(*Ex, false); mfi.isValid(); ++mfi)
        {
            const Box& gbx = mfi.growntilebox();
            (*costs[lev])[mfi.index()] += costs_heuristic_cells_wt*gbx.numPts();
        }
    }
}

void
WarpX::ResetCosts ()
{
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        if (WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            costs[lev]->setVal(0.0);
        } else if (WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Heuristic)
        {
            costs_heuristic[lev]->assign((*costs_heuristic[lev]).size(), 0.0);
        }
    }
}
