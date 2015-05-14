#!/usr/bin/env python
# vim: set fileencoding=UTF-8 :

"""Generic routines to smooth and rebin data"""
import numpy as np
import scipy.interpolate

def bounded_rebin(onek, oneP, twok, twoP, kbins):
    """Get the difference between two simulation power spectra, carefully rebinning"""
    onei = np.where(onek <= twok[-1])
    twoi= np.where (onek[onei] >= twok[0])
    relP=rebin(twoP, twok, onek[onei][twoi])
    relP=relP/rebin(oneP, onek, onek[onei][twoi])
    onek=onek[onei][twoi]
    relP_r=np.ones(np.size(kbins))
    ind = np.where(kbins > onek[0])
    relP_r[ind]=rebin(relP,onek,kbins[ind])
    return relP_r

def rebin(data, xaxis,newx):
    """Just rebins the data"""
    if newx[0] < xaxis[0] or newx[-1]> xaxis[-1]:
        raise ValueError("A value in newx is beyond the interpolation range")
    intp=scipy.interpolate.InterpolatedUnivariateSpline(np.log(xaxis),data)
    newdata=intp(np.log(newx))
    return newdata

def smooth_rebin(data, xaxis,newx=np.array([]), smoothing=11):
    """ Smooth and rebin data points """
    #Window_len of 9-13 seems about right; 7 is still pretty noisy, and higher than 13
    #tends to start changing the shape of the curve.
    smoothed=smooth(data,window_len=smoothing,window='kaiser')
    if np.size(newx) > 0:
        intp=scipy.interpolate.InterpolatedUnivariateSpline(np.log(xaxis),smoothed)
        newdata=intp(np.log(newx))
        return newdata
    else:
        return smoothed

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal

    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    np.hanning, np.hamming, np.bartlett, np.blackman, np.kaiser, np.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman','kaiser'"

    s=np.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    elif window =='kaiser':
        w=np.kaiser(window_len,14)
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]

