# Appleseed

Contents:
 - `TRFExperiment` pipeline under `./appleseed`
 - Information on stimuli, like transcripts, under `./stimuli`

## Installing

This pipeline is designed as a module to make it importable independently of the current working directory.
Start with an environment set up for `TRFExperiment` ([installation instructions](https://github.com/christianbrodbeck/TRF-Tools#installing)).
Then, install `appleseed` into the environment with:

    $ pip install -e .

After this, the `[TRFExperiment](https://trf-tools.readthedocs.io)` object can be imported in any script or notebook using that environment with:

    >>> from appleseed import e
