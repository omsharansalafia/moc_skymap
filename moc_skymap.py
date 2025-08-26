import numpy as np
import astropy_healpix as ahp
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from mocpy import MOC
from scipy.interpolate import LinearNDInterpolator

class skymap:
    """
    A class to load a MOC skymap from a FITS file and do useful computations with it.  
    """
    def __init__(s,filename):
        """
        filename (str) is the input FITS file that contains the MOC skymap. 
        """
        
        # load the FITS file as an astropy table
        s.table = Table.read(filename)
        
        # extract the unique healpix identifier and the resolution corresponding to
        # each pixel, and compute the solid angle ("area") subtended by each pixel 
        s.level, s.ipix = ahp.uniq_to_level_ipix(s.table['UNIQ'])
        s.nside = ahp.level_to_nside(s.level)
        s.pixarea = ahp.nside_to_pixel_area(s.nside)
        
        # compute the angular coordinates of each pixel
        s.ra,s.dec = ahp.healpix_to_lonlat(s.ipix,s.nside,order='nested')
        
        # create an array that contains the cumulative sum of the localization 
        # probabilities in pixels in increasing probability density order
        s.sorter = np.argsort(s.table['PROBDENSITY'])
        s.cumprob = np.zeros(len(s.sorter))
        s.cumprob[s.sorter] = np.cumsum((s.table['PROBDENSITY']*s.pixarea)[s.sorter])
        s.cumprob/=s.cumprob.max()
        
        # create interpolators that can be used to compute the value of cumprob
        # and of the localization probability density at any point in the sky
        s.cumprob_itp = LinearNDInterpolator((s.ra,s.dec),s.cumprob)
        s.prob_itp = LinearNDInterpolator((s.ra,s.dec),s.table['PROBDENSITY'])
        
    def sky_region_containing_X_probability(s,X,res=100):
        """
        X is the probability contained in the region (e.g. X=0.9 yields the 90% sky region).
        res is the ra and dec resolution.
        The function returns the tuple (ram,decm,cm):
        - ram, decm are a (res x res) meshgrid of sky coordinates
        - cm is a boolean array with True corresponding to points (ram,decm) that
          are contained within the X sky localization region. 
        """
        
        # select pixels located within the relevant sky region
        sel = s.cumprob>=(1.-X)
        ra0 = np.min(s.ra[sel])
        ra1 = np.max(s.ra[sel])
        dec0 = np.min(s.dec[sel])
        dec1 = np.max(s.dec[sel])
        
        # create a meshgrid of RA, DEC coordinates over which to evaluate
        # whether a point lies within the relevant sky region
        RA = np.linspace(ra0,ra1,res+1)
        DEC = np.linspace(dec0,dec1,res)
        ram,decm = np.meshgrid(RA,DEC)
        cm = s.cumprob_itp(ram,decm)>(1.-X)
        
        return ram,decm,cm
    
    def distance_posterior(s,position='allsky',res=1000,return_dist_params=False):
        """
        Parameters:
        - position: if 'allsky', return the marginalized distance posterior; 
                    otherwise must be either a SkyCoord instance, or 
                    a string with a sky position in hms,dms; 
        - res: resolution of the distance vector
        - return_dust_params: if True, return the DISTMU, DISTSIGMA and DISTNORM parameters
        
        Returns:
        - r, dp/dr, (DISTMU, DISTSIGMA, DISTNORM)
        """
        
        good = np.isfinite(s.table['DISTMU']) & np.isfinite(s.table['DISTSIGMA']) & np.isfinite(s.table['DISTNORM'])
        mu_max = s.table['DISTMU'][good].max()
        sigma_max = s.table['DISTSIGMA'][good].max()
        
        r = np.linspace(0.,mu_max+5.*sigma_max,res)*u.Mpc
        
        pri = s.table['PROBDENSITY']*s.pixarea*s.table['DISTNORM']*r[:,None]**2*np.exp(-0.5*((r[:,None]-s.table['DISTMU'])/s.table['DISTSIGMA'])**2)/np.sqrt(2.*np.pi*s.table['DISTSIGMA']**2)
        
        if position=='allsky':
            pr = np.mean(pri,axis=1)
            w = s.table['PROBDENSITY']*s.pixarea
            DISTMU = np.sum(w*s.table['DISTMU'])/np.sum(w)
            DISTSIGMA = np.sum(w*s.table['DISTSIGMA'])/np.sum(w)
            DISTNORM = np.sum(w*s.table['DISTNORM'])/np.sum(w)
              
        else:
            pos = SkyCoord(position)
            ipix = np.argmin(((pos.ra-s.ra)**2+(pos.dec-s.dec)**2).to('arcsec2').value)
            pr = pri[:,ipix]
            DISTMU = s.table['DISTMU'][ipix]
            DISTSIGMA = s.table['DISTSIGMA'][ipix]
            DISTNORM = s.table['DISTNORM'][ipix]
            
            
        pr/=np.trapz(pr,r)
        if return_dist_params:
            return r,pr,DISTMU,DISTSIGMA,DISTNORM
        else:
            return r,pr
    
    def intersect_with_MOC(s,moc,renormalize=True):
        """
        Erase the probabilities of pixels falling outside the given MOC (an instance of mocpy). 
        If renormalize=True, also renormalize the probability to one within the region.
        """
        
        skymap_moc = MOC.from_valued_healpix_cells(s.table['UNIQ'],s.table['PROBDENSITY'],values_are_densities=True,max_depth=10)
        
        intersection = moc.intersection(skymap_moc)
        
    def localization_region_extent(s,X):
        """
        Return the extent of the X% sky localization "area".
        """
        
        A = np.sum(s.pixarea[s.cumprob>=(1.-X)])
        
        return A   
        
    
    def write(s,filename,overwrite=True):
        
        s.table.write(filename,overwrite=overwrite)
        
        
        
        

class region:
    """
    A class to define simple sky regions, such as FoVs, to be combined with a skymap.
    """
    def __init__(s, ra0, dec0, size, shape='circle', PA=0.):
        """
        Initialize the class.
        
        Parameters:
         - ra0: the right ascension of the center of the region (to be used to initialize an astropy.coordinates.SkyCoord class), e.g. "14h00m00s"
         - dec0: the declination of the center of the region (to be used to initialize an astropy.coordinates.SkyCoord class), e.g. "+25d00m00s"
         - size: the size (radius or side length) of the region. If no units are attached, then assume it is degrees
        
        Keyword arguments:
         - shape: the shape of the region. Only 'circle' implemented so far.
         - PA: the position angle, measured clockwise from the N direction, in degrees. Default 0.
        """
        
        s.center = SkyCoord(ra=ra0,dec=dec0)
        s.shape = shape
        s.PA = PA*u.deg
        
        if type(size)==u.quantity.Quantity:
            s.size = size
        else:
            s.size = size*u.deg
        
        if shape=='circle':
            s.inside = s.inside_circle
        else:
            raise Exception('Not implemented')
        
    def inside_circle(s,ra,dec):
        """
        Check whether the ra,dec coordinates are inside the region, assumed circular.
        """
        
        pos = SkyCoord(ra=ra,dec=dec)
        dist = pos.separation(s.center)
        
        return dist<=s.size
            
            
        
        
        
