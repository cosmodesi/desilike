# y1kp45_mock_challenge

After installing cosmosis, include the automatically-generated likelihood *_values.ini
file in you the CosmoSIS *_values.ini file, and add the path to the automatically-generated *.py
file to the CosmoSIS *.ini file.

Then you should be able to run e.g.:
```
cosmosis-configure
export CSL_DIR=/our/path/to/cosmosis-standard-library
cosmosis AbacusSummitLRGFullPowerSpectrumMultipoles.ini
```
