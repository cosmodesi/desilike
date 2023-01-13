# y1kp45_mock_challenge

After installing MontePython:
- copy-paste the relevant automatically-generated likelihood directory in the MontePython montepython/likelihoods folder
- copy-paste the content of the likelihood directory's *.param file in the nuisance parameter section of the MontePython *.param file.
Then, you can run e.g.:
```
python /path/to/montepython/MontePython.py run -p AbacusSummitLRGFullPowerSpectrumMultipoles.param -o _chains/
```
