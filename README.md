Python code to automate the generation of Gadget3 simulation config files.

The base class is Simulation, which creates the config files for a single simulation.

It is meant to be called from other classes as part of a suite,
More specialised simulation types can inherit from it.
For example LymanAlphaSimulation implements config files for simulating the Lyman alpha forest

Machine-specific data is implemented with a function which dynamically subclasses the base class.
