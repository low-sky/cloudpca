from . import pca_utils as pca

from spectral_cube import SpectralCube

def structure_function(input,nScales = 10, noiseScales = 10, method = 'fit', meanCorrection = False):
    """
    Calculates structure function of molecular line emission cube using PCA

    Parameters
    ----------
    input : SpectralCube or string
       Either the SpectralCube object or load path for the same.
    nScales : int
       Number of size - line width scales to explore.  Defaults to 10.
    noiseScales : int
       Number of scales used for noise estimation.  Defaults to 10.  To suppress 
       noise correction, set to 0.
    method : 'fit' or 'interpolate'
       Choses method to estimate the 1/e widths of the ACFs.  Defaults to fitting 
       a Gaussian ('fit').  'interpolate' uses a spline interpolation

    Returns
    -------
    Size_scale : 1D `numpy` array 
        Measure of size in pixel units
    LineWidth_scale : 1D `numpy` array
        Measure of LineWidth returned in pixel units

    """


    if isinstance(input,SpectralCube):
        cube = input
    elif isinstance(input,str):
        try:
            cube = SpectralCube.read(input)
        except:
            raise
    else:
        raise NotImplementedError

    evals, evec, _ = pca.pca(cube, meanCorrection = meanCorrection)
    imgStack = pca.EigenImages(evec, cube, nScales = nScales)
    acorImg = pca.AutoCorrelateImages(imgStack)
    NoiseACF = pca.NoiseACF(evec, cube ,nScales = noiseScales)
    acorSpec = pca.AutoCorrelateSpectrum(evec, nScales = nScales)
    line_width = pca.WidthEstimate1D(acorSpec, method = method)
    size = pca.WidthEstimate2D(acorImg, NoiseACF = NoiseACF, method = method)
    return size,line_width
