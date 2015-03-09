import numpy as np
import astropy.io.fits as fits
from spectral_cube import SpectralCube
import numpy.fft as fft

from astropy.modeling import models, fitting


def WidthEstimate2D(inList):
    scales = np.zeros(len(inList))
    for idx,z in enumerate(inList):
        x = fft.fftfreq(z.shape[0])*z.shape[0]/2.0
        y = fft.fftfreq(z.shape[1])*z.shape[1]/2.0
        xmat,ymat = np.meshgrid(x,y,indexing='xy')
        g = models.Gaussian2D(x_mean=[0],y_mean=[0],
                              x_stddev =[1],y_stddev = [1],
                              amplitude = z[0,0],
                              theta = [0],
                              fixed ={'amplitude':True,
                                      'x_mean':True,
                                      'y_mean':True})
        fit_g = fitting.LevMarLSQFitter()
        output = fit_g(g,xmat,ymat,z)
#        plt.imshow(z)
#        plt.contour(output(xmat,ymat))
#        plt.show()
        scales[idx]=np.sqrt(output.x_stddev.value[0]**2+
                            output.y_stddev.value[0]**2)
    return scales


def WidthEstimate1D(inList):
    scales = np.zeros(len(inList))
    for idx,y in enumerate(inList):
        x = fft.fftfreq(len(y))*len(y)/2.0
        g = models.Gaussian1D(amplitude=y[0],mean=[0],stddev = [1],
                              fixed={'amplitude':True,'mean':True})
        fit_g = fitting.LevMarLSQFitter()
        output = fit_g(g,x,y)
        scales[idx]=np.abs(output.stddev.value[0])
#        plt.plot(x,y)
#        plt.plot(x,output(x))
#        plt.show()
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

def EigenImages(evec,cube,nScales = 10):
    imageList = []
    for idx in range(nScales):
        thisImage = np.zeros((cube.shape[1],cube.shape[2]))
        for channel in range(cube.shape[0]):
            thisImage +=np.nan_to_num(cube[channel,:,:].value*evec[channel,idx])
        imageList.append(thisImage)
    return imageList

def pca(cube):
    PCAMatrix = np.zeros((cube.shape[0],cube.shape[0]))
    ChannelMeans = np.zeros((cube.shape[0]))
    for i in range(cube.shape[0]):
        ChannelMeans[i] = np.nanmean(cube[i,:,:].value)
    for i in range(cube.shape[0]):
        for j in range(i):
            PCAMatrix[i,j] = np.nanmean(((cube[i,:,:].value-ChannelMeans[i])*(cube[j,:,:].value-ChannelMeans[j])))
        PCAMatrix[i,i] = np.nanmean((cube[i,:,:].value-ChannelMeans[i])**2)
    PCAMatrix = PCAMatrix + np.transpose(PCAMatrix)
    # Correct elements on the diagonal for the doubling in the transpose-and-add
    PCAMatrix[range(cube.shape[0]),range(cube.shape[0])] = \
        PCAMatrix[range(cube.shape[0]),range(cube.shape[0])]/2
    PCAMatrix[np.isnan(PCAMatrix)]=0.0
    evals,evec = np.linalg.eig(PCAMatrix)
    order = (np.argsort(evals))[::-1]
    evals = evals[order]
    evec = evec[:,order]
    return evals,evec,PCAMatrix
