"""A rate network for neutral hydrogen following
Katz, Weinberg & Hernquist 1996, eq. 28-32."""
import numpy as np
import scipy.interpolate as interp
import scipy.optimize

class RateNetwork(object):
    """A rate network for neutral hydrogen following
    Katz, Weinberg & Hernquist 1996, astro-ph/9509107, eq. 28-32."""
    def __init__(self,redshift, photo_factor = 1., f_bar = 0.17, converge = 1e-5, selfshield=True):
        self.recomb = RecombRatesVerner96()
        self.photo = PhotoRates()
        self.photo_factor = photo_factor
        self.f_bar = f_bar
        #proton mass in g
        self.protonmass = 1.67262178e-24
        self.redshift = redshift
        self.converge = converge
        self.selfshield = selfshield
        zz = [0, 1, 2, 3, 4, 5, 6, 7,8]
        #Tables for the self-shielding correction. Note these are not well-measured for z > 5!
        gray_opac = [2.59e-18,2.37e-18,2.27e-18, 2.15e-18, 2.02e-18, 1.94e-18, 1.82e-18, 1.71e-18, 1.60e-18]
        self.Gray_ss = interp.InterpolatedUnivariateSpline(zz, gray_opac)

    def get_temp(self, density, ienergy, helium=0.24):
        """Get the equilibrium temperature at given internal energy.
        density is gas density in protons/cm^3
        Internal energy is in J/kg == 10^-10 ergs/g.
        helium is a mass fraction"""
        ne = self.get_equilib_ne(density, ienergy, helium)
        nh = density * (1-helium)
        return self._get_temp(ne/nh, ienergy, helium)

    def get_equilib_ne(self, density, ienergy,helium=0.24):
        """Solve the system of equations for photo-ionisation equilibrium,
        starting with ne = nH and continuing until convergence.
        density is gas density in protons/cm^3
        Internal energy is in J/kg == 10^-10 ergs/g.
        helium is a mass fraction.
        """
        #Get hydrogen number density
        nh = density * (1-helium)
        rooted = lambda ne: self._ne(nh, self._get_temp(ne/nh, ienergy, helium=helium), ne, helium=helium)
        ne = scipy.optimize.fixed_point(rooted, nh,xtol=self.converge)
        assert np.all(np.abs(rooted(ne) - ne) < self.converge)
        return ne

    def get_neutral_fraction(self, density, ienergy, helium=0.24):
        """Get the neutral hydrogen fraction at a given temperature and density.
        density is gas density in protons/cm^3
        Internal energy is in J/kg == 10^-10 ergs/g.
        helium is a mass fraction.
        """
        ne = self.get_equilib_ne(density, ienergy, helium=helium)
        nh = density * (1-helium)
        temp = self._get_temp(ne/nh, ienergy, helium)
        return self._nH0(nh, temp, ne) / nh

    def _nH0(self, nh, temp, ne):
        """The neutral hydrogen number density. Eq. 33 of KWH."""
        alphaHp = self.recomb.alphaHp(temp)
        GammaeH0 = self.recomb.GammaeH0(temp)
        photorate = self.photo.gH0(self.redshift)/ne*self.photo_factor*self._self_shield_corr(nh, temp)
        return nh * alphaHp/ (alphaHp + GammaeH0 + photorate)

    def _nHp(self, nh, temp, ne):
        """The ionised hydrogen number density. Eq. 34 of KWH."""
        return nh - self._nH0(nh, temp, ne)

    def _nHep(self, nh, temp, ne):
        """The ionised helium number density, divided by the helium number fraction. Eq. 35 of KWH."""
        alphaHep = self.recomb.alphaHep(temp) + self.recomb.alphad(temp)
        alphaHepp = self.recomb.alphaHepp(temp)
        photofac = self.photo_factor*self._self_shield_corr(nh, temp)
        GammaHe0 = self.recomb.GammaeHe0(temp) + self.photo.gHe0(self.redshift)/ne*photofac
        GammaHep = self.recomb.GammaeHep(temp) + self.photo.gHep(self.redshift)/ne*photofac
        return nh / (1 + alphaHep / GammaHe0 + GammaHep/alphaHepp)

    def _nHe0(self, nh, temp, ne):
        """The neutral helium number density, divided by the helium number fraction. Eq. 36 of KWH."""
        alphaHep = self.recomb.alphaHep(temp) + self.recomb.alphad(temp)
        photofac = self.photo_factor*self._self_shield_corr(nh, temp)
        GammaHe0 = self.recomb.GammaeHe0(temp) + self.photo.gHe0(self.redshift)/ne*photofac
        return self._nHep(nh, temp, ne) * alphaHep / GammaHe0

    def _nHepp(self, nh, temp, ne):
        """The doubly ionised helium number density, divided by the helium number fraction. Eq. 37 of KWH."""
        photofac = self.photo_factor*self._self_shield_corr(nh, temp)
        GammaHep = self.recomb.GammaeHep(temp) + self.photo.gHep(self.redshift)/ne*photofac
        alphaHepp = self.recomb.alphaHepp(temp)
        return self._nHep(nh, temp, ne) * GammaHep / alphaHepp

    def _ne(self, nh, temp, ne, helium=0.24):
        """The electron number density. Eq. 38 of KWH."""
        yy = helium / 4 / (1 - helium)
        return self._nHp(nh, temp, ne) + yy * self._nHep(nh, temp, ne) + 2* yy * self._nHepp(nh, temp, ne)

    def _self_shield_corr(self, nh, temp):
        """Photoionisation rate as  a function of density from Rahmati 2012, eq. 14.
        Calculates Gamma_{Phot} / Gamma_{UVB}.
        Inputs: hydrogen density, temperature
            n_H
        The coefficients are their best-fit from appendix A."""
        if not self.selfshield:
            return np.ones_like(nh)
        nSSh = 1.003*self._self_shield_dens(self.redshift, temp)
        return 0.98*(1+(nh/nSSh)**1.64)**-2.28+0.02*(1+nh/nSSh)**-0.84

    def _self_shield_dens(self,redshift, temp):
        """Calculate the critical self-shielding density. Rahmati 202 eq. 13.
        gray_opac is a parameter of the UVB used.
        gray_opac is in cm^2 (2.49e-18 is HM01 at z=3)
        temp is particle temperature in K
        f_bar is the baryon fraction. 0.17 is roughly 0.045/0.265
        Returns density in atoms/cm^3"""
        T4 = temp/1e4
        G12 = self.photo.gH0(redshift)/1e-12
        return 6.73e-3 * (self.Gray_ss(redshift) / 2.49e-18)**(-2./3)*(T4)**0.17*(G12)**(2./3)*(self.f_bar/0.17)**(-1./3)

    def _get_temp(self, nebynh, ienergy, helium=0.24):
        """Compute temperature (in K) from internal energy and electron density.
           Uses: internal energy
                 electron abundance per H atom (ne/nH)
                 hydrogen mass fraction (0.76)
           Internal energy is in J/kg, internal gadget units, == 10^-10 ergs/g.
           Factor to convert U (J/kg) to T (K) : U = N k T / (γ - 1)
           T = U (γ-1) μ m_P / k_B
           where k_B is the Boltzmann constant
           γ is 5/3, the perfect gas constant
           m_P is the proton mass

           μ = 1 / (mean no. molecules per unit atomic weight)
             = 1 / (X + Y /4 + E)
             where E = Ne * X, and Y = (1-X).
             Can neglect metals as they are heavy.
             Leading contribution is from electrons, which is already included
             [+ Z / (12->16)] from metal species
             [+ Z/16*4 ] for OIV from electrons."""
        #convert U (J/kg) to T (K) : U = N k T / (γ - 1)
        #T = U (γ-1) μ m_P / k_B
        #where k_B is the Boltzmann constant
        #γ is 5/3, the perfect gas constant
        #m_P is the proton mass
        #μ is 1 / (mean no. molecules per unit atomic weight) calculated in loop.
        #Internal energy units are 10^-10 erg/g
        hy_mass = 1 - helium
        muienergy = 4 / (hy_mass * (3 + 4*nebynh) + 1)*ienergy*1e10
        #Boltzmann constant (cgs)
        boltzmann=1.38066e-16
        gamma=5./3
        #So for T in K, boltzmann in erg/K, internal energy has units of erg/g
        temp = (gamma-1) * self.protonmass / boltzmann * muienergy
        return temp

class RecombRatesCen92(object):
    """Recombination rates and collisional ionization rates, as a function of temperature.
    This is taken from KWH 06, astro-ph/9509107, Table 2, based on Cen 1992.
    Illustris uses these rates."""
    def alphaHp(self,temp):
        """Recombination rate for H+, ionized hydrogen, in cm^3/s.
        Temp in K."""
        return 8.4e-11 / np.sqrt(temp) / np.power(temp/1000, 0.2) / (1+ np.power(temp/1e6, 0.7))

    def alphaHep(self,temp):
        """Recombination rate for He+, ionized helium, in cm^3/s.
        Temp in K."""
        return 1.5e-10 / np.power(temp,0.6353)

    def alphad(self, temp):
        """Recombination rate for dielectronic recombination, in cm^3/s.
        Temp in K."""
        return 1.9e-3 / np.power(temp,1.5) * np.exp(-4.7e5/temp)*(1+0.3*np.exp(-9.4e4/temp))

    def alphaHepp(self, temp):
        """Recombination rate for doubly ionized helium, in cm^3/s.
        Temp in K."""
        return 3.36e-10 / np.sqrt(temp) / np.power(temp/1000, 0.2) / (1+ np.power(temp/1e6, 0.7))

    def GammaeH0(self,temp):
        """Collisional ionization rate for H0 in cm^3/s. Temp in K"""
        return 5.85e-11 * np.sqrt(temp) * np.exp(-157809.1/temp) / (1+ np.sqrt(temp/1e5))

    def GammaeHe0(self,temp):
        """Collisional ionization rate for H0 in cm^3/s. Temp in K"""
        return 2.38e-11 * np.sqrt(temp) * np.exp(-285335.4/temp) / (1+ np.sqrt(temp/1e5))

    def GammaeHep(self,temp):
        """Collisional ionization rate for H0 in cm^3/s. Temp in K"""
        return 5.68e-12 * np.sqrt(temp) * np.exp(-631515.0/temp) / (1+ np.sqrt(temp/1e5))

class RecombRatesVerner96(object):
    """Recombination rates and collisional ionization rates, as a function of temperature.
     Recombination rates are the fit from Verner & Ferland 1996 (astro-ph/9509083).
     Collisional rates are the fit from Voronov 1997 (http://www.sciencedirect.com/science/article/pii/S0092640X97907324).

     In a very photoionised medium this changes the neutral hydrogen abundance by approximately 10% compared to Cen 1992.
     These rates are those used by Nyx.
    """
    def _Verner96Fit(self, temp, aa, bb, temp0, temp1):
        """Formula used as a fitting function in Verner & Ferland 1996 (astro-ph/9509083)."""
        sqrttt0 = np.sqrt(temp/temp0)
        sqrttt1 = np.sqrt(temp/temp1)
        return aa / ( sqrttt0 * (1 + sqrttt0)**(1-bb)*(1+sqrttt1)**(1+bb) )

    def alphaHp(self,temp):
        """Recombination rate for H+, ionized hydrogen, in cm^3/s.
        The V&F 96 fitting formula is accurate to < 1% in the worst case.
        Temp in K."""
        #See line 1 of V&F96 table 1.
        return self._Verner96Fit(temp, aa=7.982e-11, bb=0.748, temp0=3.148, temp1=7.036e+05)

    def alphaHep(self,temp):
        """Recombination rate for He+, ionized helium, in cm^3/s.
        Accurate to ~2% for T < 10^6 and 5% for T< 10^10.
        Temp in K."""
        #VF96 give two rates. The first is more accurate for T < 10^6, the second is valid up to T = 10^10.
        #We use the most accurate allowed. See lines 2 and 3 of Table 1 of VF96.
        lowTfit = self._Verner96Fit(temp, aa=3.294e-11, bb=0.6910, temp0=1.554e+01, temp1=3.676e+07)
        highTfit = self._Verner96Fit(temp, aa=9.356e-10, bb=0.7892, temp0=4.266e-02, temp1=4.677e+06)
        #Note that at 10^6K the two fits differ by ~10%. This may lead one to disbelieve the quoted accuracies!
        #We thus switch over at a slightly lower temperature.
        #The two fits cross at T ~ 3e5K.
        swtmp = 7e5
        deltat = 1e5
        upper = swtmp + deltat
        lower = swtmp - deltat
        #In order to avoid a sharp feature at 10^6 K, we linearly interpolate between the two fits around 10^6 K.
        interpfit = (lowTfit * (upper - temp) + highTfit * (temp - lower))/(2*deltat)
        return (temp < lower)*lowTfit + (temp > upper)*highTfit + (upper > temp)*(temp > lower)*interpfit

    def alphad(self, temp):
        """Recombination rate for dielectronic recombination, in cm^3/s. This is the value from Cen 1992.
        An updated value should probably be sought.
        Temp in K."""
        return 1.9e-3 / np.power(temp,1.5) * np.exp(-4.7e5/temp)*(1+0.3*np.exp(-9.4e4/temp))

    def alphaHepp(self, temp):
        """Recombination rate for doubly ionized helium, in cm^3/s. Accurate to 2%.
        Temp in K."""
        #See line 4 of V&F96 table 1.
        return self._Verner96Fit(temp, aa=1.891e-10, bb=0.7524, temp0=9.370, temp1=2.774e6)

    def _Voronov96Fit(self, temp, dE, PP, AA, XX, KK):
        """Fitting function for collisional rates. Eq. 1 of Voronov 1997. Accurate to 10%,
        but data is only accurate to 50%."""
        bolevk = 8.61734e-5 # Boltzmann constant in units of eV/K
        UU = dE / (bolevk * temp)
        return AA * (1 + PP * np.sqrt(UU))/(XX+UU) * UU**KK * np.exp(-UU)

    def GammaeH0(self,temp):
        """Collisional ionization rate for H0 in cm^3/s. Temp in K. Voronov 97, Table 1."""
        return self._Voronov96Fit(temp, 13.6, 0, 0.291e-07, 0.232, 0.39)

    def GammaeHe0(self,temp):
        """Collisional ionization rate for He0 in cm^3/s. Temp in K. Voronov 97, Table 1."""
        return self._Voronov96Fit(temp, 24.6, 0, 0.175e-07, 0.180, 0.35)

    def GammaeHep(self,temp):
        """Collisional ionization rate for HeI in cm^3/s. Temp in K. Voronov 97, Table 1."""
        return self._Voronov96Fit(temp, 54.4, 1, 0.205e-08, 0.265, 0.25)

class PhotoRates(object):
    """The photoionization rates for a given species.
    Eq. 29 of KWH 96. This is loaded from a TREECOOL table."""
    def __init__(self, treecool_file="TREECOOL"):
        #Format of the treecool table:
        # log_10(1+z), Gamma_HI, Gamma_HeI, Gamma_HeII,  Qdot_HI, Qdot_HeI, Qdot_HeII,
        # where 'Gamma' is the photoionization rate and 'Qdot' is the photoheating rate.
        # The Gamma's are in units of s^-1, and the Qdot's are in units of erg s^-1.
        data = np.loadtxt(treecool_file)
        redshifts = data[:,0]
        photo_rates = data[:,1:4]
        assert np.shape(redshifts)[0] == np.shape(photo_rates)[0]
        self.Gamma_HI = interp.InterpolatedUnivariateSpline(redshifts, photo_rates[:,0])
        self.Gamma_HeI = interp.InterpolatedUnivariateSpline(redshifts, photo_rates[:,1])
        self.Gamma_HeII = interp.InterpolatedUnivariateSpline(redshifts, photo_rates[:,2])

    def gHe0(self,redshift):
        """Get photo rate for neutral Helium"""
        log1z = np.log10(1+redshift)
        return self.Gamma_HeI(log1z)

    def gHep(self,redshift):
        """Get photo rate for singly ionized Helium"""
        log1z = np.log10(1+redshift)
        return self.Gamma_HeII(log1z)

    def gH0(self,redshift):
        """Get photo rate for neutral Hydrogen"""
        log1z = np.log10(1+redshift)
        return self.Gamma_HI(log1z)
