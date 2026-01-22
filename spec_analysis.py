### Import everything
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits 
import astropy.units as u
import specutils as spec 
from mpdaf.obj import Spectrum, WaveCoord
import h5py
import pyneb as pn 
from dust_extinction.parameter_averages import G03_SMCBar
from astropy.cosmology import Planck18 as cosmo 
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler
from scipy.optimize import curve_fit

### Define all functions here


def exponential(x, a, b):
    return(b * x**a)

def linear(x, a, b):
    return(b + a*x)


def read_spectrum(specfile, z=0, micron=False):
    """
    Inputs: filename, redshift (optional)
    If no redshift is supplied, it is not added to the specutils object

    Read 1D spectrum using specutils

    """
    f = fits.open(specfile)
    
    if micron:
        lamb_m = f['WAVELENGTH'].data * u.m
        lamb = lamb_m.to(u.micron)
    else:
        lamb_m = f['WAVELENGTH'].data * u.m
        lamb = lamb_m.to(u.angstrom)

    ### Remove nan
    flux_val = f['DATA'].data 
    flux_val = np.nan_to_num(flux_val)
    flux_watt = flux_val * u.watt/u.m**2/u.m
    flux = flux_watt.to(u.erg/u.s/u.cm**2/u.AA)

    unc_val = f['ERR'].data
    unc_val = np.nan_to_num(unc_val)
    unc_watt = unc_val * u.watt/u.m**2/u.m
    unc = unc_watt.to(u.erg/u.s/u.cm**2/u.AA)

    ### First create a specutils 1D spectrum
    if z==0:
        specutil_spec = spec.Spectrum1D(spectral_axis=lamb, flux=flux, uncertainty=StdDevUncertainty(unc))  

    else:
        specutil_spec = spec.Spectrum1D(spectral_axis=lamb, flux=flux, uncertainty=StdDevUncertainty(unc), 
                                        redshift=z)  
    
    return(specutil_spec)



class spectrum2d():
    def __init__(self, w, f, e):
        self.wavelength = np.array(w)
        self.flux = np.array(f)
        self.error = np.array(e)



def load_2D_spectrum(fname, micron=False):
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
    """
    Input: specutils spectrum object

    De-redshift the spectrum, with spectroscopic redshift info that we previously added to the object
    
    """

    rest_lamb = spectrum.spectral_axis/(1+z)
    rest_flux = spectrum.flux*(1+z)
    rest_unc = spectrum.uncertainty.array*(1+z) * u.erg/u.s/u.cm**2/u.AA
    
    rest_specutil_spec = spec.Spectrum1D(spectral_axis=rest_lamb, flux=rest_flux, 
                                         uncertainty=StdDevUncertainty(rest_unc))

    return rest_specutil_spec
    

def create_restspec(spectrum, z):
    """ 
    Input: specutils spectrum object
    Output: mpdaf Spectrum object, and error spectrum, which allows easier manipulation

    Create a new wavelength grid divided by (1+z) and multiply the fluxes and errors accordingly by (1+z), 
    keeping everything else the same

    """
    
    rest_lamb = spectrum.spectral_axis/(1+z)
    rest_flux = spectrum.flux*(1+z)
    rest_unc = spectrum.uncertainty.array*(1+z) * u.erg/u.s/u.cm**2/u.AA
    
    rest_specutil_spec = spec.Spectrum1D(spectral_axis=rest_lamb, flux=rest_flux, 
                                         uncertainty=StdDevUncertainty(rest_unc))
    
    ### Let us resample the rest spectrum to match the mean wavelength resolution, needed for mpdaf object
    lamb_start = np.min(rest_lamb.value)
    lamb_end = np.max(rest_lamb.value) 
    # median_step = int((lamb_end-lamb_start)/len(rest_lamb))
    
    resampled_rest_wave = np.linspace(lamb_start, lamb_end, len(rest_lamb)) * u.AA
    median_step = resampled_rest_wave[1].value - resampled_rest_wave[0].value

    linear = LinearInterpolatedResampler()
    resamp_rest_spec = linear(rest_specutil_spec, resampled_rest_wave)
    
    
    ### Now create the mpdaf object, with flux taken from the flux conserved resampling
    mp_wave = WaveCoord(crval=lamb_start, cdelt=median_step, cunit=u.angstrom) 

    ### Create a MPDAF 1D spectrum object that will be used
    rest_spectrum = Spectrum(wave=mp_wave, data=resamp_rest_spec.flux.value, 
                             var=resamp_rest_spec.uncertainty.array, unit=(u.erg/u.s/u.cm**2/u.angstrom))
    rest_err_spectrum = Spectrum(wave=mp_wave, data=resamp_rest_spec.uncertainty.array, 
                                 unit=(u.erg/u.s/u.cm**2/u.angstrom))

    return(rest_spectrum, rest_err_spectrum)



def measure_beta(restspec, redshift=6, lmin=1340, lmax=2400):
    """
    Input: mpdaf Spectrum object, redshift of the object to determine the wavelength fitting range
    Output: beta, error on beta, best fitting parameters and covariance matrix

    """
    ### Change the wavelength range if the redshift is low
    if redshift <= 3.5:
        lmin = 2000
        lmax = 3500

    restspec_copy = restspec.copy()
    uvspec = restspec_copy.subspec(lmin, lmax, unit=u.AA)
    
    uvspec.mask_region(1440, 1590, unit=u.AA)
    uvspec.mask_region(1620, 1680, unit=u.AA)
    uvspec.mask_region(1866, 1980, unit=u.AA)
    
    p0 = [0., 1e-20]
    
    bestfit, cov_beta = curve_fit(exponential, uvspec.wave.coord(), uvspec.data, sigma=uvspec.var, p0=p0)
    beta = bestfit[0]
    beta_err = abs(beta * np.sqrt(np.mean(np.square(np.diag(cov_beta))))+
                    np.sqrt(np.mean(np.square(uvspec.var))))
    
    return(beta, beta_err, bestfit, cov_beta)

def measure_beta_bootstrap(restspec, specid, redshift=6, lmin=1340, lmax=2700, plot=True):
    """
    Input: mpdaf Spectrum object, redshift of the object to determine the wavelength fitting range
    Output: beta, error on beta, best fitting parameters and covariance matrix

    This routine performs a monte-carlo (bootstrapping) based beta measurement with more realistic uncertainties

    """

    ### Change the wavelength range if the redshift is low
    if redshift < 4.0:
        lmin = 2000
        lmax = 3500

    restspec_copy = restspec.copy()
    uvspec = restspec_copy.subspec(lmin, lmax, unit=u.AA)

    uvspec.mask_region(1440, 1590, unit=u.AA)
    uvspec.mask_region(1620, 1680, unit=u.AA)
    uvspec.mask_region(1880, 1940, unit=u.AA)
    

    ### Implement a bootstrapping method to measure the beta slopes robustly
    ### Idea is to perturb each flux pixel with a random number x the error
    boot_betas = []
    boot_errors = []
    norm_betas = []

    np.random.seed(42)

    ### Mask bad pixels
    # Set a standard deviation threshold
    std_threshold = 10
    # Calculate mean and standard deviation
    mean, median, std = sigma_clipped_stats(uvspec.data, sigma=3)
    # Identify outliers
    outliers_mask = np.abs(uvspec.data - mean) > std_threshold * std
    # Mask pixels that tend to zero
    zero_mask = uvspec.data < 1e-23
    # Mask outliers by replacing them with a specific value (e.g., NaN)
    uvspec.data[outliers_mask] = median
    uvspec.data[zero_mask] = median

    # # Alternatively, mark outliers as invalid (e.g., using np.ma.masked_invalid)
    # masked_data = np.ma.masked_array(data, mask=outliers_mask)

    for i in range(500):
        rand_noise = [np.random.uniform(-1,1) for _ in range(len(uvspec.data))] * uvspec.var
        boot_spec = uvspec.data + rand_noise

        p0 = [-2., 1.]
        
        try:
            bestfit, cov_beta = curve_fit(exponential, uvspec.wave.coord(), boot_spec, sigma=uvspec.var, p0=p0)
            beta = bestfit[0]
            beta_err = abs(beta * np.sqrt(np.mean(np.square(np.diag(cov_beta))))+
                np.sqrt(np.mean(np.square(uvspec.var))))
            norm = bestfit[1]

        except RuntimeError:
            beta = np.nan
            beta_err = np.nan
            norm = np.nan

        boot_betas.append(beta)
        boot_errors.append(beta_err)
        norm_betas.append(norm)

    boot_betas = np.array(boot_betas)
    boot_errors = np.array(boot_errors)
    norm_betas = np.array(norm_betas)

    measured_beta = np.nanmedian(boot_betas)
    measured_beta_error = np.nanstd(boot_betas)
    measured_norm = np.nanmedian(norm_betas)

    ### Calculate the best-fitting curve
    bestfit_curve = exponential(restspec.wave.coord(), measured_beta, measured_norm)
    betfit_curve_err1 = exponential(restspec.wave.coord(), measured_beta+measured_beta_error, measured_norm)
    betfit_curve_err2 = exponential(restspec.wave.coord(), measured_beta-measured_beta_error, measured_norm)

    ### Residuals
    residuals = restspec.data - bestfit_curve

    ### Let us convert the best fit, errors and residuals into MPDAF objects so that we can manipulate them
    bestfit_spectrum = Spectrum(wave=restspec.wave, data=bestfit_curve, var=betfit_curve_err1, unit=u.erg/u.s/u.cm**2/u.AA)
    error_spectrum = Spectrum(wave=restspec.wave, data=betfit_curve_err1, unit=u.erg/u.s/u.cm**2/u.AA)
    residual_spectrum = Spectrum(wave=restspec.wave, data=residuals, var=restspec.var, unit=u.erg/u.s/u.cm**2/u.AA)

    if plot==True:
        ### Check if directory exists
        if not os.path.isdir("./UV_slopes/"):
            os.mkdir("./UV_slopes/")

        fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True, figsize=(6,4))

        ax1.grid(alpha=0.4)
        ax1.step(restspec.wave.coord(), restspec.data, lw=0.75, c='k', zorder=10)
        ax1.errorbar(restspec.wave.coord(), restspec.data,
            yerr=restspec.var, alpha=0.2, zorder=0, capsize=0)

        ### Show best-fit and uncertainties
        ax1.plot(restspec.wave.coord(), bestfit_curve, zorder=11)
        ax1.fill_between(restspec.wave.coord(), betfit_curve_err2, betfit_curve_err1,
            alpha=0.3, zorder=1)

        ax1.set_yscale('log')

        ax2.grid(alpha=0.4)
        ax2.step(restspec.wave.coord(), residuals, lw=0.75, c='k', zorder=10)
        ax2.fill_between(restspec.wave.coord(), 0+restspec.var, 0-restspec.var, alpha=0.3, zorder=1)

        ax2.axvline(x=1215.67, ls='--', c='k', alpha=0.5, zorder=2)
        ax2.axvline(x=3646.0, ls='--', c='k', alpha=0.5, zorder=2)

        ax2.set_ylim(np.nanmin(residuals)/2, np.nanmax(residuals)*1.1)

        # ax2.set_yscale('log')

        plt.figtext(0.74, 0.9, f"{specid}")
        plt.figtext(0.74, 0.86, f"z = {redshift}")
        plt.figtext(0.74, 0.82, r"$\beta = $ %.2f +/- %.2f"%(measured_beta, measured_beta_error))


        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        plt.savefig(f"./UV_slopes/{specid}_beta_fit.png", dpi=300)
        plt.close()

    return(measured_beta, measured_beta_error, bestfit_spectrum, error_spectrum, residual_spectrum)


def measure_lines_from_residuals(residual_spec, specid, redshift):
    halpha_wave = 6564.614
    hbeta_wave = 4862.721
    oiii4959_wave = 4960.295
    oiii5007_wave = 5008.239
    oii_wave = 3727.50

    # initialize all fluxes as nans
    halpha_flux = np.nan
    halpha_err = np.nan
    oiii5007_flux = np.nan
    oiii5007_err = np.nan
    oiii4959_flux = np.nan
    oiii4959_err = np.nan  
    hbeta_flux = np.nan
    hbeta_err = np.nan
    oii_flux = np.nan
    oii_err = np.nan  

    ### Check if directory exists
    if not os.path.isdir("./line_fits/"):
        os.mkdir("./line_fits/")

    fig = plt.figure(figsize=(8,3))
    plt.grid(alpha=0.4)
    # Plot the residual spectrum
    plt.step(residual_spec.wave.coord(), residual_spec.data, lw=0.75, c='k', zorder=10)
    plt.fill_between(residual_spec.wave.coord(), residual_spec.data+residual_spec.var, 
        residual_spec.data-residual_spec.var, alpha=0.3, zorder=1)


    if (halpha_wave*(1+redshift)) < 52000:
        halpha_fit = residual_spec.gauss_fit(lmin=(halpha_wave-100), lmax=(halpha_wave+100), lpeak=halpha_wave, unit=u.AA, cont=0., plot=True)
        halpha_flux = halpha_fit.flux
        halpha_err = halpha_fit.err_flux

        oiii_doublet_fit = residual_spec.gauss_dfit(lmin=(oiii5007_wave-180), lmax=(oiii5007_wave+120), 
            wratio=(oiii4959_wave/oiii5007_wave), lpeak_1=oiii4959_wave, fratio=(1/2.98), cont=0., unit=u.AA, plot=True)
        if oiii_doublet_fit[0].flux > oiii_doublet_fit[1].flux:
            oiii5007_flux = oiii_doublet_fit[0].flux
            oiii5007_err = oiii_doublet_fit[0].err_flux
            oiii4959_flux = oiii_doublet_fit[1].flux
            oiii4959_err = oiii_doublet_fit[1].err_flux
        else:
            oiii5007_flux = oiii_doublet_fit[1].flux
            oiii5007_err = oiii_doublet_fit[1].err_flux
            oiii4959_flux = oiii_doublet_fit[0].flux
            oiii4959_err = oiii_doublet_fit[0].err_flux

        hbeta_fit = residual_spec.gauss_fit(lmin=(hbeta_wave-100), lmax=(hbeta_wave+100), lpeak=hbeta_wave, unit=u.AA, cont=0., plot=True)
        hbeta_flux = hbeta_fit.flux
        hbeta_err = hbeta_fit.err_flux

        oii_fit = residual_spec.gauss_fit(lmin=(oii_wave-100), lmax=(oii_wave+100), lpeak=oii_wave, unit=u.AA, cont=0., plot=True)
        oii_flux = oii_fit.flux
        oii_err = oii_fit.err_flux

    elif (oiii5007_wave*(1+redshift)) < 52000:
        oiii_doublet_fit = residual_spec.gauss_dfit(lmin=(oiii5007_wave-180), lmax=(oiii5007_wave+120), 
            wratio=(oiii4959_wave/oiii5007_wave), lpeak_1=oiii4959_wave, fratio=(1/2.98), cont=0., unit=u.AA, plot=True)
        if oiii_doublet_fit[0].flux > oiii_doublet_fit[1].flux:
            oiii5007_flux = oiii_doublet_fit[0].flux
            oiii5007_err = oiii_doublet_fit[0].err_flux
            oiii4959_flux = oiii_doublet_fit[1].flux
            oiii4959_err = oiii_doublet_fit[1].err_flux
        else:
            oiii5007_flux = oiii_doublet_fit[1].flux
            oiii5007_err = oiii_doublet_fit[1].err_flux
            oiii4959_flux = oiii_doublet_fit[0].flux
            oiii4959_err = oiii_doublet_fit[0].err_flux

        hbeta_fit = residual_spec.gauss_fit(lmin=(hbeta_wave-100), lmax=(hbeta_wave+100), lpeak=hbeta_wave, unit=u.AA, cont=0., plot=True)
        hbeta_flux = hbeta_fit.flux
        hbeta_err = hbeta_fit.err_flux

        oii_fit = residual_spec.gauss_fit(lmin=(oii_wave-100), lmax=(oii_wave+100), lpeak=oii_wave, unit=u.AA, cont=0., plot=True)
        oii_flux = oii_fit.flux
        oii_err = oii_fit.err_flux

    elif (hbeta_wave*(1+redshift)) < 52000: 
        hbeta_fit = residual_spec.gauss_fit(lmin=(hbeta_wave-100), lmax=(hbeta_wave+100), lpeak=hbeta_wave, unit=u.AA, cont=0., plot=True)
        hbeta_flux = hbeta_fit.flux
        hbeta_err = hbeta_fit.err_flux

        oii_fit = residual_spec.gauss_fit(lmin=(oii_wave-100), lmax=(oii_wave+100), lpeak=oii_wave, unit=u.AA, cont=0., plot=True)
        oii_flux = oii_fit.flux
        oii_err = oii_fit.err_flux


        
    if (hbeta_wave*(1+redshift)) < 52000: 
        plt.axvline(x=hbeta_wave, ls='--', c='k', alpha=0.5, zorder=2)
        plt.axvline(x=oii_wave, ls='--', c='k', alpha=0.5, zorder=2)
        
    if (oiii5007_wave*(1+redshift)) < 52000:
        plt.axvline(x=oiii5007_wave, ls='--', c='k', alpha=0.5, zorder=2)
        
    if (halpha_wave*(1+redshift)) < 52000:
        plt.axvline(x=halpha_wave, ls='--', c='k', alpha=0.5, zorder=2)

    plt.ylim(np.nanmin(residual_spec.data)/2, np.nanmax(residual_spec.data)*1.1)
    
    plt.figtext(0.84, 0.8, f"{specid}")
    plt.figtext(0.84, 0.74, f"z = {redshift}")

    plt.tight_layout()

    plt.savefig(f"./line_fits/{specid}_linefit.png", dpi=300)
    plt.close()        

    return(oii_flux, oii_err, hbeta_flux, hbeta_err, oiii4959_flux, oiii4959_err, oiii5007_flux, oiii5007_err, halpha_flux, halpha_err)




def cal_1500_mag(rest_spectrum, z, lmin=1475, lmax=1525):
    """
    Input: MPDAF Spectrum object, redshift
    Output: UV mag, upper bound, lower bound, F(1500), Error F(1500)

    Calculate the 1500 magnitude. 
    Reference here: https://astronomy.stackexchange.com/questions/35396/how-to-convert-luminosity-at-rest-frame-wavelength-of-1450-a-to-absolute-magnitu
    
    """
    
    ### start by calculating the flux at 1500A rest-frame
    f1500, median, std = sigma_clipped_stats(rest_spectrum.subspec(lmin, lmax, unit=u.AA).data, sigma=3)
    err_f1500 = np.std(rest_spectrum.subspec(lmin, lmax, unit=u.AA).data)

    f1500 = f1500 * u.erg/u.s/u.cm**2/u.AA
    err_f1500 = err_f1500 * u.erg/u.s/u.cm**2/u.AA
    lumdist = cosmo.luminosity_distance(z).to(u.parsec)
    
    ### flux_density at a distance of 10 parsec (definition of UV mag)
    fnu_1500 = (f1500 * (lumdist/(10.*u.parsec))**2).to(u.erg/u.s/u.cm**2/u.Hertz, 
                                                        equivalencies=u.spectral_density(1500*u.AA))
    fnu_1500_u = ((f1500 + err_f1500) * (lumdist/(10.*u.parsec))**2).to(u.erg/u.s/u.cm**2/u.Hertz, 
                                                                        equivalencies=u.spectral_density(1500*u.AA))
    fnu_1500_l = ((f1500 - err_f1500) * (lumdist/(10.*u.parsec))**2).to(u.erg/u.s/u.cm**2/u.Hertz, 
                                                                        equivalencies=u.spectral_density(1500*u.AA))
    
    ### UV mag calculation
    mab_1500 = (-2.5 * np.log10(fnu_1500.value)) - 48.60
    mab_1500_u = abs((-2.5 * np.log10(fnu_1500_u.value) - 48.60) - mab_1500)
    mab_1500_l = abs((-2.5 * np.log10(fnu_1500_l.value) - 48.60) - mab_1500)
    
    return(mab_1500, mab_1500_u, mab_1500_l, f1500, err_f1500)

def cal_sfr_wilkins(ha_flux, err_ha_flux, redshift):
    """ 
    Use the Wilkins+19 BPASS based calibration for high-z galaxies, assuming 5% metallity
    return sfr and err_sfr
    """
    conversion_factor = 4.169e41 # solar mass/yr.erg/s
    ha_lum = ha_flux * (4*np.pi*(cosmo.luminosity_distance(redshift).to(u.cm))**2).value

    sfr = ha_lum/conversion_factor
    err_sfr = sfr * (err_ha_flux/ha_flux)

    return(sfr, err_sfr)

def cal_sfr_kennicutt(ha_flux, err_ha_flux, redshift):
    """
    Use the Kennicutt+94 SFR relation, that relies on a standard Salpeter IMF
    return sfr and err_sfr
    """
    conversion_factor = 1.26e41 # solar mass/yr.erg/s
    ha_lum = ha_flux * (4*np.pi*(cosmo.luminosity_distance(redshift).to(u.cm))**2).value

    sfr = ha_lum/conversion_factor
    err_sfr = sfr * (err_ha_flux/ha_flux)

    return(sfr, err_sfr)    

### ADD DUST PRESCRIPTIONS INTO THE NOTEBOOK

# Here we calculate the intrinsic ratios between H lines
H1 = pn.RecAtom('H', 1)
temperature = 1e4
density = 3e2

Halpha = H1.getEmissivity(tem=temperature, den=density, lev_i=3, lev_j=2)
Hbeta = H1.getEmissivity(tem=temperature, den=density, lev_i=4, lev_j=2)
Hgamma = H1.getEmissivity(tem=temperature, den=density, lev_i=5, lev_j=2)
Hdelta = H1.getEmissivity(tem=temperature, den=density, lev_i=6, lev_j=2)

Lyalpha = H1.getEmissivity(tem=temperature, den=density, lev_i=2, lev_j=1)

### Intrinsic ratios
halpha_hbeta = Halpha/Hbeta
hbeta_hgamma = Hbeta/Hgamma
hbeta_hdelta = Hbeta/Hdelta
lyalpha_halpha = Lyalpha/Halpha

# print("Intrinsic line ratios:")
# print("Lyalpha/Halpha = %.1f" %lyalpha_halpha)
# print("Halpha/Hbeta = %.1f" %halpha_hbeta)

### Calculate reddening curves at wavelengths of interest, assuming SMC
### The corrections are applied as f_corr(lam) = f_obs(lam) * 10^(0.4 * k(lam) * E(B-V)). 
### Here, k(lam) is also A_lam/A_V
ext = G03_SMCBar()
k_halpha = ext.evaluate(1/(0.6563*u.micron))[0] * ext.Rv
k_hbeta = ext.evaluate(1/(0.4861*u.micron))[0] * ext.Rv
k_hgamma = ext.evaluate(1/(0.4340*u.micron))[0] * ext.Rv
k_hdelta = ext.evaluate(1/(0.4101*u.micron))[0] * ext.Rv
k_lyalpha = ext.evaluate(1/(0.1216*u.micron))[0] * ext.Rv
k_1500 = ext.evaluate(1/(0.1500*u.micron))[0] * ext.Rv

def cal_ebv_halpha(ha_flux, err_ha_flux, hb_flux, err_hb_flux):
    ebv = 2.5/(k_hbeta - k_halpha) * np.log10((ha_flux/hb_flux)/halpha_hbeta)
    err_ebv = ebv * np.sqrt(np.mean(np.array([(err_ha_flux/ha_flux)**2,
                                              (err_hb_flux/hb_flux)**2])))
    return(ebv, err_ebv)


def cal_ebv_hgamma(hb_flux, hg_flux):
    return(2.5/(k_hgamma - k_hbeta) * np.log10((hb_flux/hg_flux)/hbeta_hgamma))

def cal_ebv_hdelta(hb_flux, hd_flux):
    return(2.5/(k_hdelta - k_hbeta) * np.log10((hb_flux/hd_flux)/hbeta_hdelta))

def cal_lya_fesc(lya_flux, err_lya_flux, ha_flux, err_ha_flux, ebv):
    ### Calculate Lya escape fraction compared to Halpha after correcting for dust
    dustcorr_lya = lya_flux * 10**(0.4 * k_lyalpha * ebv)
    dustcorr_ha = ha_flux * 10**(0.4 * k_halpha * ebv)
    
    lya_fesc = ((dustcorr_lya/dustcorr_ha) / lyalpha_halpha) # observed dust corrected ratio divided by instrinsic
    err_lya_fesc = lya_fesc * np.sqrt(np.mean(np.array([(err_lya_flux/lya_flux)**2,
                                                            (err_ha_flux/ha_flux)**2])))
    return(lya_fesc, err_lya_fesc)

def cal_xi_ion(ha_flux, err_ha_flux, f1500, err_f1500, ebv, hbeta=False):
    if hbeta:
        ha_flux = halpha_hbeta * ha_flux
        
    dustcorr_ha = ha_flux * 10**(0.4 * k_halpha * ebv)
    dustcorr_1500 = f1500 * 10**(0.4 * k_1500 * ebv) * (1/2) ### 2x factor for continuum wrt nebular
    
    fnu_1500 = dustcorr_1500 * (u.erg/u.s/u.cm**2/u.AA).to(u.erg/u.s/u.cm**2/u.Hertz, 
                                                           equivalencies=u.spectral_density(1500*u.AA))
    
    xi_ion = (7.28e11 * dustcorr_ha) / fnu_1500
    err_xi_ion = xi_ion * np.sqrt(np.mean(np.array([(err_ha_flux/ha_flux)**2,
                                                (err_f1500/f1500)**2])))
    
    return(xi_ion.value, err_xi_ion.value)


### Calculate Electron temperature from the [OIII] 4363 / [OIII] 5007 ratio
from scipy.interpolate import interp1d

def cal_Te_4363(oiii4363_flux, err_oiii4363_flux, 
    oiii5007_flux, err_oiii5007_flux, den=1000.):
    O3 = pn.Atom("O",3)

    tem = np.logspace(3.5,5,5000)

    O3_4363 = O3.getEmissivity(tem=tem,den=den,lev_i=5,lev_j=4)
    O3_5007 = O3.getEmissivity(tem=tem,den=den,lev_i=4,lev_j=3)

    temp_interp = interp1d(np.log10(O3_4363/O3_5007),np.log10(tem))

    O3_ratio = oiii4363_flux/oiii5007_flux

    Te = 10.**temp_interp(np.log10(O3_ratio))
    err_Te = Te * np.sqrt(np.mean(np.square(err_oiii4363_flux/oiii4363_flux)+np.square(err_oiii5007_flux/oiii5007_flux)))

    return(Te, err_Te)
    

### Calculate temperature of [OII] zone using relation with T([OIII])
def calculate_TOII(TOIII):
    t_OIII = TOIII/1e4
    return(2/((1/t_OIII) + 0.8)*1e4)

def calculate_TOII_Hagele(TOIII, ne=1000):
    t_OIII = TOIII/1e4
    t_OII = (1.2 + 0.002*ne + (4.2/ne))/((1/t_OIII) + 0.08 + 0.003*ne + (2.5/ne))
    return(t_OII*1e4)


### Calculate O/H metallicities

def calculate_OH(FOII, e_FOII, FOIII, e_FOIII, FHB, e_FHB, Te, e_Te, TOII):
    # assume a density of 1000
    den = 1e3
    ### Calculate emissivities
    O3 = pn.Atom("O",3)
    O2 = pn.Atom("O",2)
    
    O3_5007 = O3.getEmissivity(tem=Te, den=den, lev_i=4, lev_j=3)
    O2_3727 = O2.getEmissivity(tem=TOII, den=den, lev_i=2, lev_j=1) + O2.getEmissivity(tem=TOII, den=den, lev_i=3, lev_j=1)
    
    H1 = pn.RecAtom('H', 1)
    Hbeta = H1.getEmissivity(tem=Te, den=den, lev_i=4, lev_j=2)

    OH = (FOII*Hbeta)/(FHB*O2_3727) + (FOIII*Hbeta)/(FHB*O3_5007)
    OH_err = OH * (np.sqrt(np.mean([np.square(e_FOII/FOII), np.square(e_FOIII/FOIII), np.square(e_FHB/FHB), np.square(e_Te/Te), np.square(e_Te/Te)], axis=0)))
    return OH, OH_err


