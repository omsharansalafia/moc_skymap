# moc_skymap
simple python code to read MOC GW skymaps and extract info


# Dependencies
numpy, astropy, mocpy, scipy

# Usage example 

```python

from moc_skymap import skymap
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

filename = 'bayestar.multiorder.fits'

sm = skymap(filename)

my_position = SkyCoord(237.97,30.9,unit='deg')

r,pr = sm.distance_posterior(my_position)

plt.plot(r,pr,'-r')
plt.xlabel('Distance (Mpc)')
plt.ylabel('Probability density at my position (Mpc-1)')
plt.show()
```
