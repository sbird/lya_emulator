"""Specialised module to contain functions to specialise the simulation run to different clusters"""
import os.path

class ClusterClass:
    """Generic class implementing some general defaults for cluster submissions."""
    def __init__(self, gadget="MP-Gadget", genic="MP-GenIC", param="mpgadget.param", genicparam="_genic_params.ini", nproc=256, timelimit=24):
        """CPU parameters (walltime, number of cpus, etc):
        these are specified to a default here, but should be over-ridden in a machine-specific decorator."""
        self.nproc = nproc
        self.email = "sbird@ucr.edu"
        self.timelimit = timelimit
        #Maximum memory available for an MPI task
        self.memory = 1800
        self.gadgetexe = gadget
        self.gadgetparam = param
        self.genicexe=genic
        self.genicparam=genicparam

    def generate_mpi_submit(self, outdir):
        """Generate a sample mpi_submit file.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        name = os.path.basename(os.path.normpath(outdir))
        with open(os.path.join(outdir, "mpi_submit"),'w') as mpis:
            mpis.write("#!/bin/bash\n")
            mpis.write(self._queue_directive(name, timelimit=self.timelimit, nproc=self.nproc))
            mpis.write(self._mpi_program(command=self.gadgetexe+" "+self.gadgetparam))

    def generate_mpi_submit_genic(self, outdir, extracommand=None):
        """Generate a sample mpi_submit file for MP-GenIC.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        name = os.path.basename(os.path.normpath(outdir))
        with open(os.path.join(outdir, "mpi_submit_genic"),'w') as mpis:
            mpis.write("#!/bin/bash\n")
            mpis.write(self._queue_directive(name, timelimit=0.5, nproc=self.nproc))
            mpis.write(self._mpi_program(command=self.genicexe+" "+self.genicparam))
            if extracommand is not None:
                mpis.write(extracommand+"\n")

    def _mpi_program(self, command):
        """String for MPI program to execute"""
        qstring = "mpirun -np "+str(self.nproc)+" "+command+"\n"
        return qstring

    def timestring(self, timelimit):
        """Convert a fractional timelimit into a string"""
        hr = int(timelimit)
        minute = int((timelimit - hr)*60)
        assert 0 <= minute < 60
        timestring = str(hr)+":"+str(minute)+":00"
        return timestring

    def _queue_directive(self, name, timelimit, nproc=16, prefix="#PBS"):
        """Write the part of the mpi_submit file that directs the queueing system.
        This is usually specific to a given cluster.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        _ = name
        _ = nproc
        qstring = prefix+" -j eo\n"
        qstring += prefix+" -m bae\n"
        qstring += prefix+" -M "+self.email+"\n"
        qstring += prefix+" -l walltime="+self.timestring(timelimit)+"\n"
        return qstring

    def cluster_runtime(self):
        """Runtime options for cluster. Applied to both MP-GenIC and MP-Gadget."""
        return {}

    def cluster_config_options(self,config, prefix=""):
        """Config options that might be specific to a particular cluster"""
        _ = (config, prefix)
        #isend/irecv is quite slow on some clusters because of the extra memory allocations.
        #Maybe test this on your specific system and see if it helps.
        #config.write(prefix+"NO_ISEND_IRECV_IN_DOMAIN\n")
        #config.write(prefix+"NO_ISEND_IRECV_IN_PM\n")
        #config.write(prefix+"NOTYPEPREFIX_FFTW\n")

    def cluster_optimize(self):
        """Compiler optimisation options for a specific cluster.
        Only MP-Gadget pays attention to this."""
        return "-fopenmp -O3 -g -Wall -ffast-math -march=native"

class HipatiaClass(ClusterClass):
    """Subclassed for specific properties of the Hipatia cluster in Barcelona.
    __init__ and _queue_directive are changed."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = 2500

    def _queue_directive(self, name, timelimit, nproc=16, prefix="#PBS"):
        """Generate mpi_submit with coma specific parts"""
        qstring = super()._queue_directive(name=name, prefix=prefix, timelimit=timelimit)
        qstring += prefix+" -l nodes="+str(int(nproc/16))+":ppn=16\n"
        qstring += prefix+" -l mem="+str(int(self.memory*nproc/1000))+"g\n"
        #Pass environment to child processes
        qstring += prefix+" -V\n"
        return qstring

    def _mpi_program(self, command):
        """String for MPI program to execute. Hipatia is weird because PBS_JOBID needs to be unset for the job to launch."""
        #Change to current directory
        qstring = "cd $PBS_O_WORKDIR\n"
        #Don't ask me why this works, but it is necessary.
        qstring += "unset PBS_JOBID\n"
        qstring += "mpirun -np "+str(self.nproc)+" "+command+"\n"
        return qstring

class MARCCClass(ClusterClass):
    """Subclassed for the MARCC cluster at JHU.
    This has 24 cores per node, shared memory of 128GB pr node.
    Ask for complete nodes.
    Uses SLURM."""
    def __init__(self, *args, nproc=48,timelimit=8,**kwargs):
        #Complete nodes!
        assert nproc % 24 == 0
        super().__init__(*args, nproc=nproc,timelimit=timelimit, **kwargs)
        self.memory = 5000

    def _queue_directive(self, name, timelimit, nproc=48, prefix="#SBATCH"):
        """Generate mpi_submit with coma specific parts"""
        _ = timelimit
        qstring = prefix+" --partition=parallel\n"
        qstring += prefix+" --job-name="+name+"\n"
        qstring += prefix+" --time="+self.timestring(timelimit)+"\n"
        qstring += prefix+" --nodes="+str(int(nproc/24))+"\n"
        #Number of tasks (processes) per node
        qstring += prefix+" --ntasks-per-node=24\n"
        #Number of cpus (threads) per task (process)
        qstring += prefix+" --cpus-per-task=1\n"
        #Max 128 GB per node (24 cores)
        qstring += prefix+" --mem-per-cpu="+str(self.memory)+"\n"
        qstring += prefix+" --mail-type=end\n"
        qstring += prefix+" --mail-user="+self.email+"\n"
        return qstring

    def _mpi_program(self, command):
        """String for MPI program to execute.
        Note that this assumes you aren't using threads!"""
        #Change to current directory
        qstring = "export OMP_NUM_THREADS=1\n"
        #This is for threads
        #qstring += "export OMP_NUM_THREADS = $SLURM_CPUS_PER_TASK\n"
        #Adjust for thread/proc balance per socket.
        #qstring += "mpirun --map-by ppr:3:socket:PE=4 "+self.gadgetexe+" "+self.gadgetparam+"\n"
        qstring += "mpirun --map-by core "+command+"\n"
        return qstring

    def cluster_optimize(self):
        """Compiler optimisation options for a specific cluster.
        Only MP-Gadget pays attention to this."""
        return "-fopenmp -O3 -g -Wall -march=native"

class BIOClass(ClusterClass):
    """Subclassed for the biocluster at UCR.
    This has 32 cores per node, shared memory of 128GB per node.
    Ask for complete nodes.
    Uses SLURM."""
    def __init__(self, *args, nproc=4,timelimit=2,**kwargs):
        #Complete nodes!
        super().__init__(*args, nproc=nproc,timelimit=timelimit, **kwargs)
        self.memory = 230

    def _queue_directive(self, name, timelimit, nproc=4, prefix="#SBATCH"):
        """Generate mpi_submit with coma specific parts"""
        _ = timelimit
        qstring = prefix+" --partition=short\n"
        qstring += prefix+" --job-name="+name+"\n"
        qstring += prefix+" --time="+self.timestring(timelimit)+"\n"
        qstring += prefix+" --nodes="+str(int(nproc))+"\n"
        #Number of tasks (processes) per node
        qstring += prefix+" --ntasks-per-node=2\n"
        #Number of cpus (threads) per task (process)
        qstring += prefix+" --cpus-per-task=32\n"
        #Max memory is 0.6 * 512 per node, but there is a job maximum of 1TB and 4 nodes.
        #So per node this value should be a little less than 250 GB.
        qstring += prefix+" --mem="+str(self.memory)+"G\n"
        qstring += prefix+" --mail-type=end\n"
        qstring += prefix+" --mail-user="+self.email+"\n"
        return qstring

    def _mpi_program(self, command):
        """String for MPI program to execute."""
        qstring = "export OMP_NUM_THREADS=32\n"
        qstring += "mpirun"+command+"\n"
        return qstring

    def cluster_runtime(self):
        """Runtime options for cluster. Here memory."""
        return {'MaxMemSizePerNode': self.memory * 1024}

    def cluster_optimize(self):
        """Compiler optimisation options for a specific cluster.
        Only MP-Gadget pays attention to this."""
        return "-fopenmp -O3 -g -Wall -ffast-math -march=corei7"

    def generate_spectra_submit(self, outdir):
        """Generate a sample spectra_submit file, which generates artificial spectra.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        name = os.path.basename(os.path.normpath(outdir))
        with open(os.path.join(outdir, "spectra_submit"),'w') as mpis:
            mpis.write("#!/bin/bash\n")
            mpis.write("""#SBATCH --partition=short\n#SBATCH --job-name="""+name+"\n")
            mpis.write("""#SBATCH --time=1:55:00\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=1\n#SBATCH --cpus-per-task=32\n#SBATCH --mem-per-cpu=4G\n""")
            mpis.write( """#SBATCH --mail-type=end\n#SBATCH --mail-user=sbird@ucr.edu\nexport OMP_NUM_THREADS=32\n""")
            mpis.write("python flux_power.py "+outdir+"/output\n")

class StampedeClass(ClusterClass):
    """Subclassed for Stampede2's Skylake nodes.
    This has 48 cores (96 threads) per node, each with two sockets, shared memory of 192GB per node, 96 GB per socket.
    Charged in node-hours, uses SLURM and icc."""
    def __init__(self, *args, nproc=8,timelimit=3,**kwargs):
        super().__init__(*args, nproc=nproc,timelimit=timelimit, **kwargs)

    def _queue_directive(self, name, timelimit, nproc=8, prefix="#SBATCH", ntasks=2):
        """Generate mpi_submit with stampede specific parts"""
        _ = timelimit
        qstring = prefix+" --partition=skx-normal\n"
        qstring += prefix+" --job-name="+name+"\n"
        qstring += prefix+" --time="+self.timestring(timelimit)+"\n"
        qstring += prefix+" --nodes=%d\n" % int(nproc)
        #Number of tasks (processes) per node:
        #currently optimal is 1 process per socket.
        qstring += prefix+" --ntasks-per-node=%d\n" % int(ntasks)
        qstring += prefix+" --mail-type=end\n"
        qstring += prefix+" --mail-user="+self.email+"\n"
        qstring += prefix+" -A TG-AST180058\n"
        return qstring

    def _mpi_program(self, command):
        """String for MPI program to execute."""
        #Should be 96/ntasks-per-node. This uses the hyperthreading,
        #which is perhaps an extra 10% performance.
        qstring = "export OMP_NUM_THREADS=48\n"
        qstring += "ibrun "+command+"\n"
        return qstring

    def generate_spectra_submit(self, outdir, threads=48):
        """Generate a sample spectra_submit file, which generates artificial spectra.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        name = os.path.basename(os.path.normpath(outdir))
        with open(os.path.join(outdir, "spectra_submit"),'w') as mpis:
            mpis.write("#!/bin/bash\n")
            #Nodes!
            mpis.write(self._queue_directive(name, timelimit=1, nproc=1, ntasks=1))
            mpis.write("export OMP_NUM_THREADS=%d\n" % threads)
            mpis.write("export PYTHONPATH=$HOME/.local/lib/python3.7/site-packages/:$PYTHONPATH\n")
            mpis.write("python3 flux_power.py output")

    def cluster_runtime(self):
        """Runtime options for cluster."""
        #Trying to print a backtrace causes the job to hang on exit
        return {'ShowBacktrace': 0}

    def cluster_optimize(self):
        """Compiler optimisation options for stampede.
        Only MP-Gadget pays attention to this."""
        #TACC_VEC_FLAGS generates one binary for knl, one for skx.
        return "-fopenmp -O3 -g -Wall -xCORE-AVX2 -fp-model fast=1 -simd -ipo"

class FronteraClass(StampedeClass):
    """Subclassed for Stampede2's Skylake nodes.
    This has 56 cores (56 threads) per node, each with two sockets, shared memory of 192GB per node, 96 GB per socket.
    Charged in node-hours, uses SLURM and icc. Hyperthreading is OFF"""
    def _mpi_program(self, command):
        """String for MPI program to execute."""
        #Should be 96/ntasks-per-node. This uses the hyperthreading,
        #which is perhaps an extra 10% performance.
        qstring = "export OMP_NUM_THREADS=28\n"
        qstring += "ibrun "+command+"\n"
        return qstring

    def _queue_directive(self, name, timelimit, nproc=16, prefix="#SBATCH",ntasks=2):
        """Generate mpi_submit with stampede specific parts"""
        _ = timelimit
        qstring = prefix+" --partition=normal\n"
        qstring += prefix+" --job-name="+name+"\n"
        qstring += prefix+" --time="+self.timestring(timelimit)+"\n"
        qstring += prefix+" --nodes=%d\n" % int(nproc)
        #Number of tasks (processes) per node:
        #currently optimal is 2 processes per socket.
        qstring += prefix+" --ntasks-per-node=%d\n" % int(ntasks)
        qstring += prefix+" --mail-type=ALL\n"
        qstring += prefix+" --mail-user="+self.email+"\n"
        qstring += prefix+" -A AST21005\n"
        return qstring

    def generate_spectra_submit(self, outdir, threads=56):
        """Generate a sample spectra_submit file, which generates artificial spectra.
        The prefix argument is a string at the start of each line.
        It separates queueing system directives from normal comments"""
        super().generate_spectra_submit(outdir, threads=threads)

    def cluster_optimize(self):
        """Compiler optimisation options for frontera.
        Only MP-Gadget pays attention to this."""
        return "-fopenmp -O3 -g -Wall -xCORE-AVX512 -fp-model fast=1 -ipo -qopt-mem-layout-trans=4"

class HypatiaClass(ClusterClass):
    """Subclass for Hypatia cluster in UCL"""
    def _queue_directive(self, name, timelimit, nproc=256, prefix="#PBS"):
        """Generate Hypatia-specific mpi_submit"""
        _ = timelimit
        qstring = prefix+" -m bae\n"
        qstring += prefix+" -r n\n"
        qstring += prefix+" -q smp\n"
        qstring += prefix+" -N "+name+"\n"
        qstring += prefix+" -M "+self.email+"\n"
        qstring += prefix+" -l nodes=1:ppn="+str(nproc)+"\n"
        #Pass environment to child processes
        qstring += prefix+" -V\n"
        return qstring

    def _mpi_program(self, command):
        """String for MPI program to execute. Hipatia is weird because PBS_JOBID needs to be unset for the job to launch."""
        #Change to current directory
        qstring = "cd $PBS_O_WORKDIR\n"
        #Don't ask me why this works, but it is necessary.
        qstring += ". /opt/torque/etc/openmpi-setup.sh\n"
        qstring += "mpirun -v -hostfile $PBS_NODEFILE -npernode "+str(self.nproc)+" "+command+"\n"
        return qstring
