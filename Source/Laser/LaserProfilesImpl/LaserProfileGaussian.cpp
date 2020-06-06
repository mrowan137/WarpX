/* Copyright 2019 Axel Huebl, Luca Fedeli, Maxence Thevenet
 * Weiqun Zhang
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "Laser/LaserProfiles.H"
#include "Utils/WarpX_Complex.H"
#include "Utils/WarpXConst.H"

#include <cmath>


using namespace amrex;

void
WarpXLaserProfiles::GaussianLaserProfile::init (
    const amrex::ParmParse& ppl,
    const amrex::ParmParse& /* ppc */,
    CommonLaserParameters params)
{
    //Copy common params
    m_common_params = params;

    // Parse the properties of the Gaussian profile
    ppl.get("profile_waist", m_params.waist);
    ppl.get("profile_duration", m_params.duration);
    ppl.get("profile_t_peak", m_params.t_peak);
    ppl.get("profile_focal_distance", m_params.focal_distance);
    ppl.query("zeta", m_params.zeta);
    ppl.query("beta", m_params.beta);
    ppl.query("phi2", m_params.phi2);

    m_params.stc_direction = m_common_params.p_X;
    ppl.queryarr("stc_direction", m_params.stc_direction);
    auto const s = 1.0_rt / std::sqrt(
        m_params.stc_direction[0]*m_params.stc_direction[0] +
        m_params.stc_direction[1]*m_params.stc_direction[1] +
        m_params.stc_direction[2]*m_params.stc_direction[2]);
    m_params.stc_direction = {
        m_params.stc_direction[0]*s,
        m_params.stc_direction[1]*s,
        m_params.stc_direction[2]*s };
    auto const dp2 =
        std::inner_product(
            m_common_params.nvec.begin(),
            m_common_params.nvec.end(),
            m_params.stc_direction.begin(), 0.0);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::abs(dp2) < 1.0e-14,
        "stc_direction is not perpendicular to the laser plane vector");

    // Get angle between p_X and stc_direction
    // in 2d, stcs are in the simulation plane
#if AMREX_SPACEDIM == 3
    m_params.theta_stc = acos(
        m_params.stc_direction[0]*m_common_params.p_X[0] +
        m_params.stc_direction[1]*m_common_params.p_X[1] +
        m_params.stc_direction[2]*m_common_params.p_X[2]);
#else
    m_params.theta_stc = 0.;
#endif

}

/* \brief compute field amplitude for a Gaussian laser, at particles' position
 *
 * Both Xp and Yp are given in laser plane coordinate.
 * For each particle with position Xp and Yp, this routine computes the
 * amplitude of the laser electric field, stored in array amplitude.
 *
 * \param np: number of laser particles
 * \param Xp: pointer to first component of positions of laser particles
 * \param Yp: pointer to second component of positions of laser particles
 * \param t: Current physical time
 * \param amplitude: pointer to array of field amplitude.
 */
void
WarpXLaserProfiles::GaussianLaserProfile::fill_amplitude (
    const int np, Real const * AMREX_RESTRICT const Xp, Real const * AMREX_RESTRICT const Yp,
    Real t, Real * AMREX_RESTRICT const amplitude) const
{
    //Complex I(0,1);
    std::complex<double> I(0, 1);
    // Calculate a few factors which are independent of the macroparticle
    //const Real k0 = 2.*MathConst::pi/m_common_params.wavelength;
    //const Real inv_tau2 = 1._rt /(m_params.duration * m_params.duration);
    //const Real oscillation_phase = k0 * PhysConst::c * ( t - m_params.t_peak );

    const double k0 = 2.*MathConst::pi/m_common_params.wavelength;
    const double inv_tau2 = 1. /(m_params.duration * m_params.duration);
    const double oscillation_phase = k0 * PhysConst::c * ( t - m_params.t_peak );
    // The coefficients below contain info about Gouy phase,
    // laser diffraction, and phase front curvature
    const std::complex<double> diffract_factor =
        1. + I * double(m_params.focal_distance) * 2./
        ( k0 * double(m_params.waist * m_params.waist) );
    const std::complex<double> inv_complex_waist_2 =
        1. /(double(m_params.waist*m_params.waist) * diffract_factor );

    // Time stretching due to STCs and phi2 complex envelope
    // (1 if zeta=0, beta=0, phi2=0)
    const std::complex<double> stretch_factor = 1. + 4. *
        (m_params.zeta+m_params.beta*m_params.focal_distance)
        * (m_params.zeta+m_params.beta*m_params.focal_distance)
        * (inv_tau2*inv_complex_waist_2) + 2. *I * (m_params.phi2
        - m_params.beta*m_params.beta*k0*m_params.focal_distance) * inv_tau2;

    // Amplitude and monochromatic oscillations
    std::complex<double> prefactor =
        std::complex<double>(m_common_params.e_max) * std::exp( I * oscillation_phase );

    // Because diffract_factor is a complex, the code below takes into
    // account the impact of the dimensionality on both the Gouy phase
    // and the amplitude of the laser
#if (AMREX_SPACEDIM == 3)
    prefactor = prefactor / diffract_factor;
#elif (AMREX_SPACEDIM == 2)
    prefactor = prefactor / std::sqrt(diffract_factor);
#endif

    // Copy member variables to tmp copies for GPU runs.
    const double tmp_profile_t_peak = m_params.t_peak;
    const double tmp_beta = m_params.beta;
    const double tmp_zeta = m_params.zeta;
    const double tmp_theta_stc = m_params.theta_stc;
    const double tmp_profile_focal_distance = m_params.focal_distance;
    // Loop through the macroparticle to calculate the proper amplitude
    amrex::ParallelFor(
        np,
        [=] AMREX_GPU_DEVICE (int i) {
            const std::complex<double> stc_exponent = 1. / stretch_factor * inv_tau2 *
                std::pow((t - tmp_profile_t_peak -
                    tmp_beta*k0*(Xp[i]*std::cos(tmp_theta_stc) + Yp[i]*std::sin(tmp_theta_stc)) -
                          2.*I*double(Xp[i]*std::cos(tmp_theta_stc) + Yp[i]*std::sin(tmp_theta_stc))
                          *double( tmp_zeta - tmp_beta*tmp_profile_focal_distance ) * inv_complex_waist_2),2);
            // stcfactor = everything but complex transverse envelope
            const std::complex<double> stcfactor = prefactor * std::exp( - stc_exponent );
            // Exp argument for transverse envelope
            const std::complex<double> exp_argument = - (double(Xp[i]*Xp[i] + Yp[i]*Yp[i])) * inv_complex_waist_2;
            // stcfactor + transverse envelope
            amplitude[i] = (Real)( stcfactor * std::exp( exp_argument ) ).real();
        }
        );
}
