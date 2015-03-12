import numpy as np
import astropy.io.fits as fits
from spectral_cube import SpectralCube
import numpy.fft as fft
from scipy.interpolate import LSQUnivariateSpline,interp1d
from astropy.modeling import models, fitting
from scipy.signal import argrelmin
import pdb

def WidthEstimate2D(inList, method = 'fit', NoiseACF = 0):
    scales = np.zeros(len(inList))
    for idx,z in enumerate(inList):
        x = fft.fftfreq(z.shape[0])*z.shape[0]/2.0
        y = fft.fftfreq(z.shape[1])*z.shape[1]/2.0
        xmat,ymat = np.meshgrid(x,y,indexing='xy')
        rmat = (xmat**2+ymat**2)**0.5
        if method == 'fit':
            g = models.Gaussian2D(x_mean=[0],y_mean=[0],
                             x_stddev =[1],y_stddev = [1],
                             amplitude = z[0,0],
                             theta = [0],
                             fixed ={'amplitude':True,
                                     'x_mean':True,
                                     'y_mean':True})
            fit_g = fitting.LevMarLSQFitter()
            output = fit_g(g,xmat,ymat,z-NoiseACF)
            scales[idx]=2**0.5*np.sqrt(output.x_stddev.value[0]**2+
                                       output.y_stddev.value[0]**2)
        if method == 'interpolate':
            rvec = rmat.ravel()
            zvec = (z-NoiseACF).ravel()
            zvec /= zvec.max()
            sortidx = np.argsort(zvec)
            rvec = rvec[sortidx]
            zvec = zvec[sortidx]
            dz = len(zvec)/100.
            spl = LSQUnivariateSpline(zvec,rvec,zvec[dz::dz])
            scales[idx] = spl(np.exp(-1))

#        plt.imshow(z)
#        plt.contour(output(xmat,ymat))
#        plt.show()
    return scales


def WidthEstimate1D(inList, method = 'fit'):
    scales = np.zeros(len(inList))
    for idx,y in enumerate(inList):
        x = fft.fftfreq(len(y))*len(y)/2.0
        if method == 'interpolate':
            minima = (argrelmin(y))[0]
            if minima[0]>1:
                interpolator = interp1d(y[0:minima[0]],x[0:minima[0]])
                scales[idx] = interpolator(np.exp(-1))
        if method == 'fit':
            g = models.Gaussian1D(amplitude=y[0],mean=[0],stddev = [1],
                                  fixed={'amplitude':True,'mean':True})
            fit_g = fitting.LevMarLSQFitter()
            output = fit_g(g,x,y)
            scales[idx]=np.abs(output.stddev.value[0])*(2**0.5)
    return scales


def AutoCorrelateImages(imageList):
    acorList = []
    for image in imageList:
        fftx = fft.fft2(image)
        fftxs = np.conjugate(fftx)
        acor = fft.ifft2((fftx-fftx.mean())*(fftxs-fftxs.mean()))
        acorList.append(acor.real)
    return(acorList)

def AutoCorrelateSpectrum(evec,nScales = 10):
    acorList = []
    for idx in range(nScales):
        fftx = fft.fft(evec[:,idx])
        fftxs = np.conjugate(fftx)
        acor = fft.ifft((fftx-fftx.mean())*(fftxs-fftxs.mean()))
        acorList.append(acor.real)
    return(acorList)

def NoiseACF(evec, cube, nScales = 10):
    if nScales == 0:
        return 0
    imageList = []
    for idx in range(nScales):
        thisImage = np.zeros((cube.shape[1],cube.shape[2]))
        for channel in range(cube.shape[0]):
            thisImage +=np.nan_to_num(cube[channel,:,:].value*evec[channel,-(idx+1)])
        imageList.append(thisImage)
    acorList = []
    for image in imageList:
        fftx = fft.fft2(image)
        fftxs = np.conjugate(fftx)
        acor = fft.ifft2((fftx-fftx.mean())*(fftxs-fftxs.mean()))
        acorList.append(acor.real)
    
    NoiseACF = np.zeros((cube.shape[1],cube.shape[2]))
    for planeACF in acorList:
        NoiseACF += planeACF
    NoiseACF /= len(NoiseACF)
    return(NoiseACF)
    
def EigenImages(evec,cube,nScales = 10):
    imageList = []
    for idx in range(nScales):
        thisImage = np.zeros((cube.shape[1],cube.shape[2]))
        for channel in range(cube.shape[0]):
            thisImage +=np.nan_to_num(cube[channel,:,:].value*evec[channel,idx])
        imageList.append(thisImage)
    return imageList

def pca(cube, meanCorrection = False):
    PCAMatrix = np.zeros((cube.shape[0],cube.shape[0]))
    GoodCount  = np.zeros((cube.shape[0],cube.shape[0]),dtype=np.float)
    ChannelMeans = np.zeros((cube.shape[0]))
    if meanCorrection:
        for i in range(cube.shape[0]):
            ChannelMeans[i] = np.nanmean(cube[i,:,:].value)
    else:
        ChannelMeans = np.zeros(cube.shape[0])
        
    for i in range(cube.shape[0]):
        for j in range(i):
            PlaneProduct = (cube[i,:,:].value-ChannelMeans[i])*(cube[j,:,:].value-ChannelMeans[j])
            PCAMatrix[i,j] = np.nanmean(PlaneProduct)
            GoodCount[i,j] = np.sum(np.isfinite(PlaneProduct))
        PCAMatrix[i,i] = np.nanmean((cube[i,:,:].value-ChannelMeans[i])**2)
        GoodCount[i,i] = np.sum(np.isfinite(cube[i,:,:]))

    PCAMatrix = PCAMatrix + np.transpose(PCAMatrix)
    GoodCount = GoodCount + np.transpose(GoodCount)
    # Correct elements on the diagonal for the doubling in the transpose-and-add
    PCAMatrix[range(cube.shape[0]),range(cube.shape[0])] = \
        PCAMatrix[range(cube.shape[0]),range(cube.shape[0])]/2
    GoodCount[range(cube.shape[0]),range(cube.shape[0])] = \
        GoodCount[range(cube.shape[0]),range(cube.shape[0])]/2
    
    if meanCorrection:
        N = cube.shape[1]*cube.shape[2]
        PCAMatrix = PCAMatrix * GoodCount/(GoodCount-1)
    PCAMatrix[~np.isfinite(PCAMatrix)]=0.0
    evals,evec = np.linalg.eig(PCAMatrix)
    order = (np.argsort(evals))[::-1]
    evals = evals[order]
    evec = evec[:,order]
    return evals,evec,PCAMatrix
