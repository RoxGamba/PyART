
import numpy as np;

class EccentricityCalculator():
    def __init__(self, 
             h,
             pars,
             t    = None,
             omg  = None,
             kind = '3PN',
            ):
        """
        Class to compute eccentricity from a waveform
        from various definitions
        """
        self.h    = h
        self.t    = t
        self.omg  = omg
        self.kind = kind
        self.pars = pars
    
        self.e = self.compute_eccentricity(self.kind)
        pass

    def compute_eccentricity(self, kind):
        if kind[1:] == 'PN':
            return self._compute_eccentricity_PN_EJ(kind)
        elif kind == 'gwecc':
            return self._compute_eccentricity_gwecc(kind, tref_in=self.pars['tref_in'])
        else:
            raise NotImplementedError("Only PN/gwecc eccentricity is implemented for now")
        pass

    def _compute_eccentricity_PN_EJ(self, kind):
        """
        Compute eccentricity from the waveform using the 3PN formula
        Eq. 4.8 of https://arxiv.org/pdf/1507.07100.pdf
        e_t in Harmonic coordinates
        """
        q  = self.pars['q']
        nu = q/(1+q)**2 
        nu2 = nu*nu
        nu3 = nu2*nu 
        Pi  = np.pi
        Pi2 = Pi*Pi

        Eb, pph = self.h.Eb, self.h.j
        xi  = -Eb*pph**2
        e_0PN  = 1-2*xi
        e_1PN  = -4.-2*nu+ (-1 + 3*nu)*xi
        e_2PN  = (20.-23*nu)/xi -22. + 60*nu + 3*nu2 - (31*nu+4*nu2)*xi
        e_3PN  = ((-2016 + (5644 - 123*Pi2)*nu -252*nu2)/(12*xi*xi) + 
                 (4848 +(-21128 + 369*Pi2)*nu + 2988*nu2)/(24*xi) 
                 - 20 + 298*nu - 186*nu2 - 4*nu3 + 
                 (-1*30.*nu + 283./4*nu2 + 5*nu3)*xi)

        if kind == '0PN':
            return  np.sqrt(e_0PN)
        elif kind == '1PN':
            return np.sqrt(e_0PN + Eb*e_1PN)
        elif kind == '2PN':
            return np.sqrt(Eb*(Eb*e_2PN + e_1PN) + e_0PN)
        elif kind == '3PN':
            return np.sqrt(Eb*(Eb*(Eb*e_3PN + e_2PN) + e_1PN) + e_0PN)
        else:
            raise NotImplementedError("PN eccentricity is implemented up to 3PN for now")

    def _compute_eccentricity_gwecc(self, kind, tref_in=None, method='AmplitudeFits'):
        """
        Compute eccentricity from the waveform using gw_eccentricity
        """
        try:
            from gw_eccentricity import measure_eccentricity
        except ImportError:
            raise ImportError("To compute the eccentricity from the waveform you need to install `gw_eccentricity`")

        h22 = self.h.hlm['1']

        modeDict = {(2,2) : h22[0]*np.exp(-1j*h22[1])}
        dataDict = {'t': self.h.u, 'hlm': modeDict}
        res      = measure_eccentricity(tref_in=tref_in, dataDict=dataDict, method=method)
        self.res = res
        return res['eccentricity']