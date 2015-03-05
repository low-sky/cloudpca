from spectral_cube import SpectralCube
from cloudpca import cloudpca
s = SpectralCube.read('/Users/erosolo/Dropbox/AstroStatistics/ngc1333.13co.fits')
evals,evec,matrix = cloudpca.pca(s)
ll = cloudpca.EigenImages(evec,s)
acorImg = cloudpca.AutoCorrelateImages(ll)
acorSpec = cloudpca.AutoCorrelateSpectrum(evec)
scales = WidthEstimate1D(acorSpec)
