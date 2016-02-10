from spectral_cube import SpectralCube
import cloudpca
from scipy.interpolate import UnivariateSpline

# Read in a data cube
s = SpectralCube.read('/Users/erik/Dropbox/AstroStatistics/ngc1333.13co.fits')
s = s[100:,:,:]
evals,evec,matrix = cloudpca.pca(s)
ll = cloudpca.EigenImages(evec,s)
acorImg = cloudpca.AutoCorrelateImages(ll)
acorSpec = cloudpca.AutoCorrelateSpectrum(evec)
noiseACF = cloudpca.NoiseACF(evec,s)
scales = cloudpca.WidthEstimate1D(acorSpec)
sscales = cloudpca.WidthEstimate2D(acorImg,NoiseACF=noiseACF)
