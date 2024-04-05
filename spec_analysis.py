import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits 
import astropy.units as u
import specutils as spec 
from mpdaf.obj import Spectrum, WaveCoord
import pyneb as pn 
from dust_extinction.parameter_averages import G03_SMCBar
from astropy.cosmology import Planck18 as cosmo 
from astropy.nddata import StdDevUncertainty
from astropy.stats import sigma_clipped_stats
from specutils.manipulation import LinearInterpolatedResampler
from scipy.optimize import curve_fit

# Define functions

def exponential(x, a, b):
    """Exponential function for curve fitting"""
    return b * x**a

def read_spectrum(specfile, z=0, micron=False):
    """
    Read 1D spectrum using specutils.
    
    Inputs: 
        specfile: filename
        z (optional): redshift
        micron (optional): if True, wavelength in microns
    
    Returns:
        specutil_spec: specutils 1D spectrum object
    """
    f = fits.open(specfile)
    
    # Convert wavelength to desired unit
    if micron:
        lamb_m = f['WAVELENGTH'].data * u.m
        lamb = lamb_m.to(u.micron)
    else:
        lamb_m = f['WAVELENGTH'].data * u.m
        lamb = lamb_m.to(u.angstrom)

    # Remove NaN values from flux and uncertainty
    flux_val = np.nan_to_num(f['DATA'].data)
    flux_watt = flux_val * u.watt/u.m**2/u.m
    flux = flux_watt.to(u.erg/u.s/u.cm**2/u.AA)

    unc_val = np.nan_to_num(f['ERR'].data)
    unc_watt = unc_val * u.watt/u.m**2/u.m
    unc = unc_watt.to(u.erg/u.s/u.cm**2/u.AA)

    # Create specutils 1D spectrum
    if z == 0:
        specutil_spec = spec.Spectrum1D(spectral_axis=lamb, flux=flux, uncertainty=StdDevUncertainty(unc))  
    else:
        specutil_spec = spec.Spectrum1D(spectral_axis=lamb, flux=flux, uncertainty=StdDevUncertainty(unc), redshift=z)  
    
    return specutil_spec

class spectrum2d():
    def __init__(self, w, f, e):
        """2D spectrum class"""
        self.wavelength = np.array(w)
        self.flux = np.array(f)
        self.error = np.array(e)

def load_2D_spectrum(fname, micron=False):
    """Load 2D spectrum"""
    h = fits.open(fname)
    flx = h['DATA'].data
    err = h['ERR'].data

    if micron:
        wav = h['WAVELENGTH'].data*1e6
    else:
        wav = h['WAVELENGTH'].data*1e10

    spec2d = spectrum2d(wav, flx, err)
    return spec2d

def deredshift_spec(spectrum, z):
    """De-redshift spectrum"""
    rest_lamb = spectrum.spectral_axis/(1+z)
    rest_flux = spectrum.flux*(1+z)
    rest_unc = spectrum.uncertainty.array*(1+z) * u.erg/u.s/u.cm**2/u.AA
    
    rest_specutil_spec = spec.Spectrum1D(spectral_axis=rest_lamb, flux=rest_flux, 
                                         uncertainty=StdDevUncertainty(rest_unc))

    return rest_specutil_spec

def create_restspec(spectrum, z):
    """Create rest-frame spectrum"""
    rest_lamb = spectrum.spectral_axis/(1+z)
    rest_flux = spectrum.flux*(1+z)
    rest_unc = spectrum.uncertainty.array*(1+z) * u.erg/u.s/u.cm**2/u.AA
    
    rest_specutil_spec = spec.Spectrum1D(spectral_axis=rest_lamb, flux=rest_flux, 
                                         uncertainty=StdDevUncertainty(rest_unc))
    
    # Resample the rest spectrum
    lamb_start = np.min(rest_lamb.value)
    lamb_end = np.max(rest_lamb.value) 
    resampled_rest_wave = np.linspace(lamb_start, lamb_end, len(rest_lamb)) * u.AA

    linear = LinearInterpolatedResampler()
    resamp_rest_spec = linear(rest_specutil_spec, resampled_rest_wave)
    
    # Create mpdaf Spectrum object
    mp_wave = WaveCoord(crval=lamb_start, cdelt=resampled_rest_wave[1].value - resampled_rest_wave[0].value, 
                         cunit=u.angstrom) 
    rest_spectrum = Spectrum(wave=mp_wave, data=resamp_rest_spec.flux.value, 
                             var=resamp_rest_spec.uncertainty.array, unit=(u.erg/u.s/u.cm**2/u.angstrom))
    rest_err_spectrum = Spectrum(wave=mp_wave, data=resamp_rest_spec.uncertainty.array, 
                                 unit=(u.erg/u.s/u.cm**2/u.angstrom))

    return rest_spectrum, rest_err_spectrum

def measure_beta(restspec, redshift=6, lmin=1340, lmax=2400):
    """Measure UV slope beta"""
    if redshift <= 3.5:
        lmin = 2000
        lmax = 3500

    restspec_copy = restspec.copy()
    uvspec = restspec_copy.subspec(lmin, lmax, unit=u.AA)
    
    # Mask regions
    uvspec.mask_region(1440, 1590, unit=u.AA)
    uvspec.mask_region(1620, 1680, unit=u.AA)
    uvspec.mask_region(1866, 1980, unit=u.AA)
    
    p0 = [0., 1e-20]
    bestfit, cov_beta = curve_fit(exponential, uvspec.wave.coord(), uvspec.data, sigma=uvspec.var, p0=p0)
    beta = bestfit[0]
    beta_err = abs(beta * np.sqrt(np.mean(np.square(np.diag(cov_beta))))+
                    np.sqrt(np.mean(np.square(uvspec.var))))
    
    return beta, beta_err, bestfit, cov_beta

def measure_beta_bootstrap(restspec, specid, redshift=6, lmin=1340, lmax=2700, plot=True):
    """Measure UV slope beta with bootstrapping"""
    if redshift < 4.0:
        lmin = 2000
        lmax = 3500

    restspec_copy = restspec.copy()
    uvspec = restspec_copy.subspec(lmin, lmax, unit=u.AA)

    uvspec.mask_region(1440, 1590, unit=u.AA)
    uvspec.mask_region(1620, 1680, unit=u.AA)
    uvspec.mask_region(1880, 1940, unit=u.AA)
    
    boot_betas = []
    boot_errors = []
    norm_betas = []
    np.random.seed(42)
    
    for i in range(1000):
        # Generate resampled spectrum
        nspec = len(uvspec.flux)
        rand_idx = np.random.randint(0, nspec, nspec)
        resamp_spec = uvspec.flux[rand_idx]
        resamp_err = uvspec.uncertainty.array[rand_idx]

        # Fit the resampled spectrum
        p0 = [0., 1e-20]
        try:
            bestfit, cov_beta = curve_fit(exponential, uvspec.wave.coord(), resamp_spec, 
                                           sigma=resamp_err, p0=p0, absolute_sigma=True)
        except:
            continue
        
        beta = bestfit[0]
        beta_err = abs(beta * np.sqrt(np.mean(np.square(np.diag(cov_beta))))+
                       np.sqrt(np.mean(np.square(resamp_err))))

        boot_betas.append(beta)
        boot_errors.append(beta_err)
        norm_betas.append(beta*1e17)

    # Calculate mean and standard deviation of bootstrapped betas
    mean_beta = np.mean(boot_betas)
    std_beta = np.std(boot_betas)
    mean_err = np.mean(boot_errors)
    std_err = np.std(boot_errors)
    
    # Plot histogram of bootstrapped betas
    if plot:
        plt.hist(norm_betas, bins=30, color='blue', edgecolor='black', alpha=0.7)
        plt.title('Bootstrapped Beta Histogram for ' + specid)
        plt.xlabel(r'$\beta$ ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)')
        plt.ylabel('Frequency')
        plt.axvline(mean_beta*1e17, color='red', linestyle='dashed', linewidth=1)
        plt.show()

    return mean_beta, std_beta, mean_err, std_err

def find_nearest(array, value):
    """Find nearest value in an array"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def cal_1500_mag(rest_spectrum, z, lmin=1475, lmax=1525):
    """
    Calculate the 1500 magnitude.

    Input:
    - rest_spectrum: MPDAF Spectrum object
    - z: redshift
    - lmin, lmax: wavelength range (default: 1475-1525 Å)

    Output:
    - UV mag, upper bound, lower bound, F(1500), Error F(1500)

    Reference:
    https://astronomy.stackexchange.com/questions/35396/how-to-convert-luminosity-at-rest-frame-wavelength-of-1450-a-to-absolute-magnitu
    """
    
    # Calculate the flux at 1500 Å rest-frame
    f1500, median, std = sigma_clipped_stats(rest_spectrum.subspec(lmin, lmax, unit=u.AA).data, sigma=3)
    err_f1500 = np.std(rest_spectrum.subspec(lmin, lmax, unit=u.AA).data)

    f1500 = f1500 * u.erg/u.s/u.cm**2/u.AA
    err_f1500 = err_f1500 * u.erg/u.s/u.cm**2/u.AA
    lumdist = cosmo.luminosity_distance(z).to(u.parsec)
    
    # Flux density at a distance of 10 parsec (definition of UV mag)
    fnu_1500 = (f1500 * (lumdist/(10.*u.parsec))**2).to(u.erg/u.s/u.cm**2/u.Hertz, 
                                                        equivalencies=u.spectral_density(1500*u.AA))
    fnu_1500_u = ((f1500 + err_f1500) * (lumdist/(10.*u.parsec))**2).to(u.erg/u.s/u.cm**2/u.Hertz, 
                                                                        equivalencies=u.spectral_density(1500*u.AA))
    fnu_1500_l = ((f1500 - err_f1500) * (lumdist/(10.*u.parsec))**2).to(u.erg/u.s/u.cm**2/u.Hertz, 
                                                                        equivalencies=u.spectral_density(1500*u.AA))
    
    # UV mag calculation
    mab_1500 = (-2.5 * np.log10(fnu_1500.value)) - 48.60
    mab_1500_u = abs((-2.5 * np.log10(fnu_1500_u.value) - 48.60) - mab_1500)
    mab_1500_l = abs((-2.5 * np.log10(fnu_1500_l.value) - 48.60) - mab_1500)
    
    return mab_1500, mab_1500_u, mab_1500_l, f1500, err_f1500



import numpy as np
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from astropy.cosmology import Planck18 as cosmo
import pyneb as pn
from dust_extinction.parameter_averages import G03_SMCBar



# Calculate intrinsic ratios between H lines
H1 = pn.RecAtom('H', 1)
temperature = 1e4
density = 3e2

Halpha = H1.getEmissivity(tem=temperature, den=density, lev_i=3, lev_j=2)
Hbeta = H1.getEmissivity(tem=temperature, den=density, lev_i=4, lev_j=2)
Hgamma = H1.getEmissivity(tem=temperature, den=density, lev_i=5, lev_j=2)
Hdelta = H1.getEmissivity(tem=temperature, den=density, lev_i=6, lev_j=2)

Lyalpha = H1.getEmissivity(tem=temperature, den=density, lev_i=2, lev_j=1)

# Intrinsic ratios
halpha_hbeta = Halpha/Hbeta
hbeta_hgamma = Hbeta/Hgamma
hbeta_hdelta = Hbeta/Hdelta
lyalpha_halpha = Lyalpha/Halpha

# Calculate reddening curves at wavelengths of interest, assuming SMC
ext = G03_SMCBar()
k_halpha = ext.evaluate(1/(0.6563*u.micron))[0] * ext.Rv
k_hbeta = ext.evaluate(1/(0.4861*u.micron))[0] * ext.Rv
k_hgamma = ext.evaluate(1/(0.4340*u.micron))[0] * ext.Rv
k_hdelta = ext.evaluate(1/(0.4101*u.micron))[0] * ext.Rv
k_lyalpha = ext.evaluate(1/(0.1216*u.micron))[0] * ext.Rv
k_1500 = ext.evaluate(1/(0.1500*u.micron))[0] * ext.Rv

def cal_ebv_halpha(ha_flux, err_ha_flux, hb_flux, err_hb_flux):
    """
    Calculate E(B-V) based on H-alpha and H-beta fluxes.
    """
    ebv = 2.5/(k_hbeta - k_halpha) * np.log10((ha_flux/hb_flux)/halpha_hbeta)
    err_ebv = ebv * np.sqrt(np.mean(np.array([(err_ha_flux/ha_flux)**2,
                                              (err_hb_flux/hb_flux)**2])))
    return ebv, err_ebv


def cal_ebv_hgamma(hb_flux, hg_flux):
    """
    Calculate E(B-V) based on H-beta and H-gamma fluxes.
    """
    return 2.5/(k_hgamma - k_hbeta) * np.log10((hb_flux/hg_flux)/hbeta_hgamma)

def cal_ebv_hdelta(hb_flux, hd_flux):
    """
    Calculate E(B-V) based on H-beta and H-delta fluxes.
    """
    return 2.5/(k_hdelta - k_hbeta) * np.log10((hb_flux/hd_flux)/hbeta_hdelta)

def correct_EW(EW, line, z, ebv, cont, line_flux=None):
    """Correct equivalent width for dust extinction"""
    extinction_model = G03_SMCBar()
    extinction = extinction_model.extinguish(line, ebv)
    if line_flux is None:
        A_lambda = extinction.extinguish(cont)
    else:
        A_lambda = extinction.extinguish(line_flux)
    A_lambda_continuum = extinction.extinguish(cont)

    EW_corrected = EW * 10**(0.4 * A_lambda)
    EW_err = EW * 10**(0.4 * A_lambda) * (0.4 * np.log(10) * A_lambda_continuum)

    return EW_corrected, EW_err

def cal_lya_fesc(lya_flux, err_lya_flux, ha_flux, err_ha_flux, ebv):
    """
    Calculate Lyman-alpha escape fraction.
    """
    # Calculate Lya escape fraction compared to Halpha after correcting for dust
    dustcorr_lya = lya_flux * 10**(0.4 * k_lyalpha * ebv)
    dustcorr_ha = ha_flux * 10**(0.4 * k_halpha * ebv)
    
    lya_fesc = ((dustcorr_lya/dustcorr_ha) / lyalpha_halpha) # observed dust corrected ratio divided by instrinsic
    err_lya_fesc = lya_fesc * np.sqrt(np.mean(np.array([(err_lya_flux/lya_flux)**2,
                                                            (err_ha_flux/ha_flux)**2])))
    return lya_fesc, err_lya_fesc

def cal_xi_ion(ha_flux, err_ha_flux, f1500, err_f1500, ebv):
    """
    Calculate xi_ion.
    """
    # Correct fluxes for dust
    dustcorr_ha = ha_flux * 10**(0.4 * k_halpha * ebv)
    dustcorr_1500 = f1500 * 10**(0.4 * k_1500 * ebv) * (1/2) # 2x factor for continuum wrt nebular
    
    fnu_1500 = dustcorr_1500 * (u.erg/u.s/u.cm**2/u.AA).to(u.erg/u.s/u.cm**2/u.Hertz, 
                                                           equivalencies=u.spectral_density(1500*u.AA))
    
    xi_ion = (7.28e11 * dustcorr_ha) / fnu_1500
    err_xi_ion = xi_ion * np.sqrt(np.mean(np.array([(err_ha_flux/ha_flux)**2,
                                                (err_f1500/f1500)**2])))
    
    return xi_ion.value, err_xi_ion.value
