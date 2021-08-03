# arpae
## Installation
See conda_env.yml. If you already have anaconda base environment, only lmfit needs to be installed.
## Classes
### HAPI_Molecule
A class represent a single molecule in HITRAN. See [hapi.py](https://hitran.org/static/hapi/hapi.py)
### HAPI_Molecules
A class based on list and containing multiple HAPI_Molecule objects.
### TILDAS_Spectra
A class based on list and containing multiple TILDAS_Spectrum objects.
### TILDAS_Spectrum
A class based on dictionary and representing one TILDAS spectrum.
### Fitting_Window
A fitting window within a TILDAS spectrum, usually contained by an item in the TILDAS_Spectrum object.
