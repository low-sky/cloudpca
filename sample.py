from spectral_cube import SpectralCube
import cloudpca
from scipy.interpolate import UnivariateSpline
s = SpectralCube.read('/Users/erosolo/Dropbox/AstroStatistics/ngc1333.13co.fits')
evals,evec,matrix = cloudpca.pca(s)
ll = cloudpca.EigenImages(evec,s)
acorImg = cloudpca.AutoCorrelateImages(ll)
acorSpec = cloudpca.AutoCorrelateSpectrum(evec)
noiseACF = cloudpca.NoiseACF(evec,s)
scales = cloudpca.WidthEstimate1D(acorSpec)

# z = acorImg[0]
# x = fft.fftfreq(z.shape[0])*z.shape[0]/2.0
# y = fft.fftfreq(z.shape[1])*z.shape[1]/2.0

# xmat,ymat = np.meshgrid(x,y,indexing='xy')
# rmat = (xmat**2+ymat**2)**0.5

# rvec = rmat.ravel()
# zvec = z.ravel()/z.max()
# idx = np.argsort(zvec)
# rvec = rvec[idx]
# zvec = zvec[idx]
# dz = len(zvec)/100
# spl = LSQUnivariateSpline(rvec,zvec,zvec[1::dz])

