from . import pca_utils as pca

from spectral_cube import SpectralCube

def structure_function(input,nScales = 10):
    """
    Calculates structure function of molecular line emission cube using PCA

    Parameters
    ----------
    input : SpectralCube or string
       Either the SpectralCube object or load path for the same.
    nScales : int
       Number of size - line width scales to explore.  Defaults to 10.

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

    evals, evec, _ = pca.pca(cube)
    imgStack = pca.EigenImages(evec, cube, nScales = nScales)
    acorImg = pca.AutoCorrelateImages(imgStack)
    acorSpec = pca.AutoCorrelateSpectrum(evec, nScales = nScales)
    line_width = pca.WidthEstimate1D(acorSpec)
    size = pca.WidthEstimate2D(acorImg)
    return size,line_width
