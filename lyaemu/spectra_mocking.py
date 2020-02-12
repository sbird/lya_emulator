import numpy as np
import fake_spectra
from fake_spectra.plot_spectra import PlottingSpectra
from fake_spectra.spectra import Spectra
import matplotlib.pyplot as plt
import array
import h5py
import seaborn as sn

def plot_pdf_Wiener_filtered(map_file=['60_512', '60_1024', '120_1024'],bins=np.arange(-1.0, 1.0, 0.02)):
    
    plt.figure(figsize=(10,8))
    (mbin, hist) = get_pdf_Wiener_filtered(map_file = map_file, bins=bins)

    for (j,i) in enumerate(map_file):

        
        plt.title('Convergence check')
        plt.xlabel(r'$\delta_F$')
        plt.ylabel(r'$ pdf \ of \ voxels, \ \Delta \delta_F \ = \ 0.02 $')

        m = np.fromfile('./spectra/maps/map_'+i+'.dat')
        if i == '60_512':
            label = 'L = 60 h^-1 cMpc, Particles = 512'
            ls = 'dashed'
        if i == '60_1024':
            label = 'L = 60 h^-1 cMpc, Particles = 1024'
            ls = 'dotted'
        if i == '120_1024':
            label = 'L = 120 h^-1 cMpc, Particles = 1024'
            ls = 'dashdot'
        
        #sn.distplot(m, bins=bins, hist=False, kde=True, label=label
        plt.plot(mbin, hist[j], label=label, linestyle=ls)
    plt.legend()
    plt.savefig('hist_deltaF.png')


def get_pdf_Wiener_filtered(map_file=['60_512', '60_1024', '120_1024'],bins=np.arange(-1.0, 1.0, 0.02)):
    
    hist = []
    
    mbin = np.array([(bins[i]+bins[i+1])/2. for i in range(0,np.size(bins)-1)])
    for i in map_file:
        m = np.fromfile('./spectra/maps/map_'+i+'.dat')
        hist.append(np.histogram(m, bins=bins, density=True)[0])

    return mbin, hist



def plot_pdf_spectra(spectra_file=['randspectra_60_512_res_123.hdf5', 'randspectra_60_1024_res_123.hdf5', 'randspectra_120_1024_res_123.hdf5'], bins=np.arange(-8, 8, 0.2)):
    
    
    for i in spectra_file:
        plt.figure()
        f =h5py.File('./'+i,'r')
        lines = np.size(f['spectra']['axis'])
        f.close()
        deltaF = get_deltaF(savefile=i, lines=lines)
        (a, b) = np.shape(deltaF)
        deltaF = deltaF.reshape((a,b))
        plt.hist(deltaF, bins=bins, density=True)
        plt.title(i)
        plt.xlabel(r'$\delta_F$')
        plt.ylabel(r'$pdf$')
        plt.savefig('hist_deltaF_spectra'+i+'.png')


def plot_noisy_spectrum(flux=False, noise = True, spec_num=320, xlims=(-1500,1500), savefile='randspectra_120.hdf5', savefig='flux.png', color='blue', lines=10):

    
    if noise ==True :
        CNR = get_CNR(lines=lines)
        ## Get CE for eontinuum error
        CE = get_CE(CNR)
    else :

        CNR = None
        CE = None

    ps = PlottingSpectra(num = 1, base='./', savedir='./', savefile=savefile, res = 127,snr=CNR, CE=CE,spec_res = 145)

    ps.plot_spectrum(elem='H', ion=1, line=1215, spec_num=spec_num, xlims =xlims,flux=flux, color=color)
    #plt.title(savefile)
    #plt.savefig('spectrum'+str(spec_num)+'_flux_'+str(flux)+'_noise_'+str(noise)+'_spec_res_added_120.png')
    #plt.savefig(savefig)


def write_input_duchshund(savefile= 'ranspectra_120.hdf5', output_file='mock_deltaF',lines=10):
    """ Write a binary file of [X, y, z, deltaF, sigma_delta_F] of each pixel.
        Each row would be info for a single pixel

    """
    # A binary file file to write the result in
    out = open(output_file, 'wb')
   
    ps_no_noise = PlottingSpectra(num = 1, base='./', savedir='./', savefile=savefile, res = 127, snr = None, CE= None, spec_res = 145)

    deltaF = get_deltaF(savefile= savefile, lines=lines)

    ## Calculate total noise
    CNR = get_CNR(lines=lines)
    CE = get_CE(CNR)
    
    ## noise for pixels along each spectrum
    tot_noise = np.sqrt(CE**2 + (1/CNR)**2)

    ## find the position of the pixels along the spectrum
    dy = ps_no_noise.dvbin/ps_no_noise.velfac

    for i in range(lines):
        for j in range(np.size(deltaF[0])):
            array.array('d', [ps_no_noise.cofm[i][0]/1000., dy*j*1.0/1000., ps_no_noise.cofm[i][2]/1000., tot_noise[i], deltaF[i,j]]).tofile(out)
           
    out.close()




def get_deltaF(savefile='randspectra_120.hdf5', lines=10):
    """Calculates deltaF = (F/F_average) - 1   for each peixel 
        F is the flux including all noises
        F_average is the mean flux
    """

    CNR = get_CNR(lines=lines)
    CE = get_CE(CNR)
    
    #Spectra with noise 
    ps_with_noise = PlottingSpectra(num = 1, base='./', savedir='./', savefile=savefile, res = 127,snr=CNR, CE=CE,spec_res = 145)

    #Spectra without noise
    ps_no_noise = PlottingSpectra(num = 1, base='./', savedir='./', savefile=savefile, res = 127, snr = None, CE= None, spec_res = 145)

    

    flux_no_noise = ps_no_noise.get_tau(elem='H', ion=1, line=1215)
    flux_with_noise = ps_with_noise.get_tau(elem='H', ion=1, line=1215)
    

    #a, b = np.shape(flux_no_noise)

    #flux_no_noise = flux_no_noise.reshape((a*b,))
    #flux_with_noise = flux_with_noise.reshape((a*b,))
    # mean flux
    flux_mean = np.average(flux_with_noise)
    
    # over flux for each pixel
    deltaF = (flux_with_noise/(1.0*flux_mean)) - 1


    return deltaF
    

    



def get_CNR(lines):
   """ Calculate Continuum to noise ratio (signal to noise ratio) modeled in LATIS"""
   CNR = np.zeros(lines)
   np.random.seed(14)

   for ii in range(lines):

        CNR[ii] = np.exp(np.random.normal(.82, .43))

   return CNR

def get_CE(CNR) :
    """ Calculate Continuum noise for each spectra modeled in LATIS"""
    return 0.24*CNR**(-0.86)


