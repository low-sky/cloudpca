from spectral_cube import SpectralCube
from cloudpca import cloudpca
s = SpectralCube.read('~/Dropbox/AstroStatistics/ngc1333.13co.fits')
evals,evec,matrix = cloudpca.pca(s)
ll = cloudpca.EigenImages(evec,s)
