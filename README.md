# cloudpca
Implements Heyer &amp; Brunt PCA to measure turbulent structure function in clouds.

See [Heyer & Schloerb (1997)](http://adsabs.harvard.edu/abs/1997ApJ...475..173H), 
[Heyer & Brunt (2004)](http://adsabs.harvard.edu/abs/2004ApJ...615L..45H), 
[Brunt & Heyer (2013)](http://adsabs.harvard.edu/abs/2013MNRAS.433..117B)

Requires [astropy](http://astropy.readthedocs.org/en/stable/) and [spectral_cube](http://spectral-cube.readthedocs.org/en/stable/) as well as numpy.

Current non-installable.

Quick start:

"""
import cloudpca
size,linewidth = cloudpca.structure_function('my_3d_image_cube.fits')

"""