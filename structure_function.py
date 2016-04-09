from . import pca_utils as pca

from spectral_cube import SpectralCube

def structure_function(input,nScales = 10, noiseScales = 10, spatialMethod = 'contour', spectralMethod = 'interpolate', meanCorrection = False):
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
    method : 'fit', 'interpolate', or 'contour'
       Choses method to estimate the 1/e widths of the ACFs.  Defaults to 'interpolate' 
       for 1D and 'contour' for 2D.
    meanCorrection : bool
       If True, calculates a proper covariance in PCA matrix.  If False (default), 
       no correctio nis applied, following the literature approach.

    Returns
    -------
    Size_scale : 1D `numpy` array 
        Measure of size in pixel units
    LineWidth_scale : 1D `numpy` array
        Measure of LineWidth returned in pixel units

    """


    if isinstance(input,SpectralCube):
        cube = input.filled_data[:].value
    elif isinstance(input,str):
        try:
            speccube = SpectralCube.read(input)
            cube = input.filled_data[:].value
        except:
            raise
    elif isinstance(input,np.array):
        cube = input
    else:
        raise NotImplementedError

    evals, evec, _ = pca.pca(cube, meanCorrection = meanCorrection)
    imgStack = pca.EigenImages(evec, cube, nScales = nScales)
    acorImg = pca.AutoCorrelateImages(imgStack)
    NoiseACF = pca.NoiseACF(evec, cube ,nScales = noiseScales)
    acorSpec = pca.AutoCorrelateSpectrum(evec, nScales = nScales)
    line_width = pca.WidthEstimate1D(acorSpec, method = spectralMethod)
    size = pca.WidthEstimate2D(acorImg, NoiseACF = NoiseACF, method = spatialMethod)
    return size,line_width
