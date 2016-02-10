import numpy as np
import astropy.io.fits as fits
from spectral_cube import SpectralCube
import numpy.fft as fft
from scipy.interpolate import LSQUnivariateSpline,interp1d
from astropy.modeling import models, fitting
from scipy.signal import argrelmin
import pdb
import matplotlib.pyplot as plt
def Exponential1D(x, amp, scale):
    return amp*np.exp(-x/scale)

def Exponential2D(x,y,x0,y0,amp,xscale,yscale,theta):
    xrot =  x*np.cos(theta) + y*np.sin(theta)
    yrot = -x*np.sin(theta) + y*np.cos(theta)
    dist = ((xrot/xscale)**2 + (yrot/yscale)**2)**0.5
    return (amp*np.exp(-dist)).flatten()

def WidthEstimate2D(inList, method = 'contour', NoiseACF = 0):
    scales = np.zeros(len(inList))

    # set up the x/y grid just once
    z = inList[0]
    x = fft.fftfreq(z.shape[0])*z.shape[0]/2.0
    y = fft.fftfreq(z.shape[1])*z.shape[1]/2.0
    xmat,ymat = np.meshgrid(x,y,indexing='ij')
    xmat = np.fft.fftshift(xmat)
    ymat = np.fft.fftshift(ymat)
    rmat = (xmat**2+ymat**2)**0.5

    for idx,zraw in enumerate(inList):
        z = zraw - NoiseACF
        z = np.fft.fftshift(z)

        if method == 'fit':
            g = models.Gaussian2D(x_mean=[0],y_mean=[0],
                                  x_stddev =[1],y_stddev = [1],
                                  amplitude = z.max(),
                                  theta = [0],
                                  fixed ={'amplitude':True,
                                          'x_mean':True,
                                          'y_mean':True})
            fit_g = fitting.LevMarLSQFitter()
            output = fit_g(g,np.abs(xmat)**0.5,np.abs(ymat)**0.5,z)
            scales[idx]=2**0.5*np.sqrt(output.x_stddev.value[0]**2+
                                       output.y_stddev.value[0]**2)
        if method == 'interpolate':
            rvec = rmat.ravel()
            zvec = z.ravel()
            zvec /= zvec.max()
            sortidx = np.argsort(zvec)
            rvec = rvec[sortidx]
            zvec = zvec[sortidx]
            dz = len(zvec)/100.
            spl = LSQUnivariateSpline(zvec,rvec,zvec[dz::dz])
            scales[idx] = spl(np.exp(-1))
        if method == 'xinterpolate':
            g = models.Gaussian2D(x_mean=[0],y_mean=[0],
                                  x_stddev =[1],y_stddev = [1],
                                  amplitude = z[0,0],
                                  theta = [0],
                                  fixed ={'amplitude':True,
                                          'x_mean':True,
                                          'y_mean':True})
            fit_g = fitting.LevMarLSQFitter()
            output = fit_g(g,xmat,ymat,z)
            aspect = 1/(output.x_stddev.value[0]/output.y_stddev.value[0])
            theta = output.theta.value[0]
            rmat = ((xmat * np.cos(theta) + ymat * np.sin(theta))**2+\
                (-xmat * np.sin(theta) + ymat * np.cos(theta))**2*\
                aspect**2)**0.5
            rvec = rmat.ravel()
            zvec = z.ravel()
            zvec /= zvec.max()
            sortidx = np.argsort(zvec)
            rvec = rvec[sortidx]
            zvec = zvec[sortidx]
            dz = len(zvec)/100.
            spl = LSQUnivariateSpline(zvec,rvec,zvec[dz::dz])
            scales[idx] = spl(np.exp(-1))
            plt.plot((((xmat**2)+(ymat**2))**0.5).ravel(),z.ravel(),'b,')
            plt.plot(rmat.ravel(),z.ravel(),'r,')
            plt.vlines(scales[idx],zvec.min(),zvec.max())
            plt.show()
            pdb.set_trace()
        if method == 'contour':
            znorm = z
            znorm /= znorm.max()
#            plt.imshow(znorm,vmin=0,vmax=1)
            cs = plt.contour(xmat,ymat,znorm,levels=[np.exp(-1)])
            paths = (cs.collections[0].get_paths())
#            plt.show()
            plt.clf()
           # Only points that contain the origin

            if isinstance(paths,list) and len(paths)>1:
                pidx = np.where([p.contains_point((0,0)) for p in paths])
                if len(pidx[0])>0:
                    paths = paths[pidx[0]]
                    scales[idx] = (np.min(np.array([np.max(p.vertices[:,0]**2+p.vertices[:,1]**2) for p in paths])))**0.5
                else:
                    scales[idx] = np.nan
            elif len(paths)>0:
                scales[idx] = (np.max(paths[0].vertices[:,0]**2+paths[0].vertices[:,1]**2))**0.5
            else:
                scales[idx] = np.nan
    return scales


def WidthEstimate1D(inList, method = 'interpolate'):
    scales = np.zeros(len(inList))
    for idx,y in enumerate(inList):
        x = fft.fftfreq(len(y))*len(y)/2.0
        if method == 'interpolate':
            minima = (argrelmin(y))[0]
            if minima[0]>1:
                interpolator = interp1d(y[0:minima[0]],x[0:minima[0]])
                scales[idx] = interpolator(np.exp(-1))
        if method == 'fit':
            g = models.Gaussian1D(amplitude=y[0],mean=[0],stddev = [10],
                                  fixed={'amplitude':True,'mean':True})
            fit_g = fitting.LevMarLSQFitter()
            minima = (argrelmin(y))[0]
            if minima[0]>1:
                xtrans = (np.abs(x)**0.5)[0:minima[0]]
                yfit = y[0:minima[0]]
            else:
                xtrans = np.abs(x)**0.5
                yfit = y
            output = fit_g(g,xtrans,yfit)
            scales[idx]=np.abs(output.stddev.value[0])*(2**0.5)
#             expmod = Model(Exponential1D)
#             pars = expmod.make_params(amp=y[0],scale=5.0)
#             pars['amp'].vary = False
#             result = expmod.fit(y,x=x,params = pars)
#             scales[idx] = result.params['scale'].value
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
