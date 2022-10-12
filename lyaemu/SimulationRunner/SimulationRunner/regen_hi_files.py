"""Script to regenerate all HI reionisation files for an emulator"""
import glob
import os.path
import json
import sys
import make_HI_reionization_table as hi

def regen(simdir, genic_param = "_genic_params.ini"):
    """Generate a new file for a specific Simulation"""
    with open(os.path.join(simdir,"SimulationICs.json")) as fd:
        icparams = json.load(fd)
        hireionz = icparams["hireionz"]
        uvffile = os.path.join(simdir, "UVFluctuationFile")
        #print(os.path.join(simdir, genic_param), uvffile, hireionz)
        hi.generate_zreion_file(paramfile = os.path.join(simdir, genic_param), output = uvffile, redshift = hireionz, resolution = 1.0)

if __name__ == "__main__":
    emudir = sys.argv[1]
    dirs = sorted(glob.glob(os.path.join(emudir, "ns*")))

    for sdir in dirs:
        try:
            regen(simdir=sdir)
        except IOError:
            pass
