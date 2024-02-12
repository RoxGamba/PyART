
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
    
if __name__ == '__main__':

    # eccs = np.linspace(0.01, 0.7, 10)
    # kind = '3PN'

    # # Plot the diagonal
    # fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
    # ax0.plot(eccs, eccs, color='black', linestyle='--', label='Diagonal')
    
    # # PLot the PN eccentricity vs initial input eccentricity
    # for ecc in eccs:
    #     pars          = wf.CreateDict(q=1., chi1z=0., chi2z=0., l1=0,l2=0, ecc=ecc, f0=0.001)
    #     eob           = wf.Waveform_EOB(pars=pars)
    #     eob.compute_energetics()

    #     Calc = EccentricityCalculator(eob, pars, kind=kind)
    #     ax0.scatter(ecc, Calc.e[0], color='r', marker='o')
    #     ax1.scatter(ecc, abs(Calc.e[0]-ecc), color='r', marker='o')
    
    # ax0.set_ylabel(r'$e_{'+ kind+'}$')
    # ax1.set_xlabel(r'$e_{EOB}$')
    # ax1.set_ylabel(r'$|e_{EOB}-e_{'+ kind+'}|$')
    # ax1.set_yscale('log')
    # ax0.legend()
    # plt.show()

    # n      = 5
    # colors = plt.cm.magma(np.linspace(0,1,n))
    # fig, ax = plt.subplots()
    # for e,c in zip(np.linspace(0.8, 0.95, n), colors):
    #     print(e)
    #     pars            = wf.CreateDict(q=1., chi1z=0., chi2z=0., l1=0,l2=0, ecc=e, f0=0.0001, interp="no")
    #     eob             = wf.Waveform_EOB(pars=pars)
    #     plt.plot(eob.u, eob.hp)
    #     plt.show()
    # plt.show()
