"""
Pure python implementation of the (old) TEOBResumS ringdown.
Copy-paste of PyRing, with minor mods.
https://git.ligo.org/lscsoft/pyring/-/blob/master/pyRing/eob_utils.pyx?ref_type=heads
https://bitbucket.org/eob_ihes/teobresums/src/master/C/src/TEOBResumSFits.c

TODO: it would be better to have all fixed coefficients called/computed only once, and then stored
in e.g. a dictionary. Now, they are evaluated every time the function is called, which is suboptimal.
"""

import numpy as np

def Y(Y_0, b_1, b_2, b_3, c_1, c_2, c_3, af, af2, af3):
    return _Y(Y_0, b_1, b_2, b_3, c_1, c_2, c_3, af, af2, af3)

# eq. 5.21 of arXiv:1904.09550v2
def _Y(Y_0, b_1, b_2, b_3, c_1, c_2, c_3, af, af2, af3):
    """
    Function implementing eq. 5.21 of arXiv:1904.09550v2.
    It is the same fitting equation for the three coefficients alpha1, alpha21, omega1.
    """
    res = 0.

    num = 1 + b_1*af + b_2*af2 + b_3*af3
    den = 1 + c_1*af + c_2*af2 + c_3*af3
    res = Y_0 * num/den

    return res


def _alpha1(af, l, m):
    """
    Function returning the value of alpha1 for each mode, following appendix C.2 of arXiv:2001.09082v2.
    It is called in function EOBPM_SetupFitCoefficients of waveform.pyx.
    ---------------------------------------------------------------------------------------
    The fitting coefficients and equations are implemented from:
    (2,2) (2,1) (3,3) (3,2) (3,1) (4,4) (4,3) (4,2) (4,1) - arXiv:1904.09550v2, section V.D
    ---------------------------------------------------------------------------------------
    Note that, as outlined in appendix C.1 of arXiv:2001.09082v2, we use the non-spinning values (i.e.
    taken from arXiv:1904.09550v2) for the post-peak coefficients also for the spinning case.
    """

    res = 0.
    af2 = af*af; af3 = af2*af

    # -------------------------------------------------------------------------------- #
    # (2,2) - TEOBResumSFits.c, line 2803
    alp1_22_0 = 0.08896
    b_1_22    = -1.90036; b_2_22 = 0.86200; b_3_22 = 0.0384893
    c_1_22    = -1.87933; c_2_22 = 0.88062; c_3_22 = 0.

    # (2,1) - TEOBResumSFits.c, line 2798
    alp1_21_0 = 0.0889623
    b_1_21    = -1.31253; b_2_21 = -0.21033; b_3_21 = 0.52502
    c_1_21    = -1.30041; c_2_21 = -0.15660;  c_3_21 = 0.46204

    # (3,3) - TEOBResumSFits.c, line 2832
    alp1_33_0 = 0.0927030
    b_1_33    = -1.8310; b_2_33 = 0.7568; b_3_33 = 0.0745
    c_1_33    = -1.8098; c_2_33 = 0.7926; c_3_33 = 0.0196

    # (3,2) - TEOBResumSFits.c, line 2827
    alp1_32_0 = 0.0927030
    b_1_32    = -1.58277; b_2_32 = 0.2783; b_3_32 = 0.30503
    c_1_32    = -1.56797; c_2_32 = 0.3290; c_3_32 = 0.24155

    # (3,1) - TEOBResumSFits.c, line 2822
    alp1_31_0 = 0.0927030
    b_1_31    = -1.2345; b_2_31 = -0.30447; b_3_31 = 0.5446
    c_1_31    = -1.2263; c_2_31 = -0.24223; c_3_31 = 0.47738

    # (4,4) - TEOBResumSFits.c, line 2852
    alp1_44_0 = 0.0941640
    b_1_44    = -1.8662; b_2_44 = 0.8248; b_3_44 = 0.0417
    c_1_44    = -1.8514; c_2_44 = 0.8736; c_3_44 = -0.0198

    # (4,3) - TEOBResumSFits.c, line 2847
    alp1_43_0 = 0.0941640
    b_1_43    = -1.7177; b_2_43 = 0.5320; b_3_43 = 0.1860
    c_1_43    = -1.7065; c_2_43 = 0.5876; c_3_43 = 0.120939

    # (4,2) - TEOBResumSFits.c, line 2842
    alp1_42_0 = 0.0941640
    b_1_42    = -1.44152; b_2_42 = 0.0542; b_3_42 = 0.39020
    c_1_42    = -1.43312; c_2_42 = 0.1167; c_3_42 = 0.32253

    # (4,1) - TEOBResumSFits.c, line 2837
    alp1_41_0 = 0.0941640
    b_1_41    = 1.1018882; b_2_41 = -0.88643; b_3_41 = -0.78266
    c_1_41    = 1.1065495; c_2_41 = -0.80961; c_3_41 = -0.68905

    # (5,5) - TEOBResumSFits.c, line 2857
    alp1_55_0 = 0.0948705
    b_1_55    = -1.8845; b_2_55 = 0.8585; b_3_55 = 0.0263
    c_1_55    = -1.8740; c_2_55 = 0.9147; c_3_55 = -0.0384
    # -------------------------------------------------------------------------------- #

    if ((l==2) and (m==2)):
        alpha1_22 = _Y(alp1_22_0, b_1_22, b_2_22, b_3_22, c_1_22, c_2_22, c_3_22, af, af2, af3)
        res = alpha1_22

    elif ((l==2) and (m==1)):
        alpha1_21 = _Y(alp1_21_0, b_1_21, b_2_21, b_3_21, c_1_21, c_2_21, c_3_21, af, af2, af3)
        res = alpha1_21

    elif ((l==3) and (m==3)):
        alpha1_33 = _Y(alp1_33_0, b_1_33, b_2_33, b_3_33, c_1_33, c_2_33, c_3_33, af, af2, af3)
        res = alpha1_33

    elif ((l==3) and (m==2)):
        alpha1_32 = _Y(alp1_32_0, b_1_32, b_2_32, b_3_32, c_1_32, c_2_32, c_3_32, af, af2, af3)
        res = alpha1_32

    elif ((l==3) and (m==1)):
        alpha1_31 = _Y(alp1_31_0, b_1_31, b_2_31, b_3_31, c_1_31, c_2_31, c_3_31, af, af2, af3)
        res = alpha1_31

    elif ((l==4) and (m==4)):
        alpha1_44 = _Y(alp1_44_0, b_1_44, b_2_44, b_3_44, c_1_44, c_2_44, c_3_44, af, af2, af3)
        res = alpha1_44

    elif ((l==4) and (m==3)):
        alpha1_43 = _Y(alp1_43_0, b_1_43, b_2_43, b_3_43, c_1_43, c_2_43, c_3_43, af, af2, af3)
        res = alpha1_43

    elif ((l==4) and (m==2)):
        alpha1_42 = _Y(alp1_42_0, b_1_42, b_2_42, b_3_42, c_1_42, c_2_42, c_3_42, af, af2, af3)
        res = alpha1_42

    elif ((l==4) and (m==1)):
        alpha1_41 = _Y(alp1_41_0, b_1_41, b_2_41, b_3_41, c_1_41, c_2_41, c_3_41, af, af2, af3)
        res = alpha1_41

    elif ((l==5) and (m==5)):
        alpha1_55 = _Y(alp1_55_0, b_1_55, b_2_55, b_3_55, c_1_55, c_2_55, c_3_55, af, af2, af3)
        res = alpha1_55

    return res


def _alpha21(af, l, m):
    """
    Function returning the value of alpha21 for each mode, following appendix C.2 of arXiv:2001.09082v2.
    It is called in function EOBPM_SetupFitCoefficients of waveform.pyx.
    ---------------------------------------------------------------------------------------
    The fitting coefficients and equations are implemented from:
    (2,2) (2,1) (3,3) (3,2) (3,1) (4,4) (4,3) (4,2) (4,1) - arXiv:1904.09550v2, section V.D
    ---------------------------------------------------------------------------------------
    Note that, as outlined in appendix C.1 of arXiv:2001.09082v2, we use the non-spinning values (i.e.
    taken from arXiv:1904.09550v2) for the post-peak coefficients also for the spinning case.
    """

    res = 0.
    af2 = af*af; af3 = af2*af

    # -------------------------------------------------------------------------------- #
    # (2,2) - TEOBResumSFits.c, line 2804
    alp21_22_0 = 0.184953
    b_1_22     = -1.89397; b_2_22 = 0.88126; b_3_22 = 0.0130256
    c_1_22     = -1.83901; c_2_22 = 0.84162; c_3_22 = 0.

    # (2,1) - TEOBResumSFits.c, line 2799
    alp21_21_0 = 0.184952
    b_1_21     = -1.13290; b_2_21 = -0.3520; b_3_21 = 0.4924
    c_1_21     = -1.10334; c_2_21 = -0.3037; c_3_21 = 0.4262

    # (3,3) - TEOBResumSFits.c, line 2833
    alp21_33_0 = 0.188595
    b_1_33     = -1.8011; b_2_33 = 0.7046; b_3_33 = 0.0968
    c_1_33     = -1.7653; c_2_33 = 0.7176; c_3_33 = 0.0504

    # (3,2) - TEOBResumSFits.c, line 2828
    alp21_32_0 = 0.188595
    b_1_32     = -1.5212; b_2_32 = 0.1563; b_3_32 = 0.3652
    c_1_32     = -1.4968; c_2_32 = 0.1968; c_3_32 = 0.3021

    # (3,1) - TEOBResumSFits.c, line 2823
    alp21_31_0 = 0.188595
    b_1_31     = -1.035; b_2_31 = -0.3816; b_3_31 = 0.4486
    c_1_31     = -1.023; c_2_31 = -0.3170; c_3_31 = 0.3898

    # (4,4) - TEOBResumSFits.c, line 2853
    alp21_44_0 = 0.190170
    b_1_44     = -1.8546; b_2_44 = 0.8041; b_3_44 = 0.0507
    c_1_44     = -1.8315; c_2_44 = 0.8391; c_3_44 = -0.0051

    # (4,3) - TEOBResumSFits.c, line 2848
    alp21_43_0 = 0.190170
    b_1_43     = -1.6860; b_2_43 = 0.4724; b_3_43 = 0.2139
    c_1_43     = -1.6684; c_2_43 = 0.5198; c_3_43 = 0.1508

    # (4,2) - TEOBResumSFits.c, line 2843
    alp21_42_0 = 0.190170
    b_1_42     = -1.38840; b_2_42 =        0.; b_3_42 = 0.39333
    c_1_42     = -1.37584; c_2_42 = 0.0600017; c_3_42 = 0.32632

    # (4,1) - TEOBResumSFits.c, line 2838
    alp21_41_0 = 0.190170
    b_1_41     = 1.0590157; b_2_41 = -0.8650630; b_3_41 = -0.75222
    c_1_41     = 1.0654880; c_2_41 = -0.7830051; c_3_41 = -0.65814

    # (5,5) - TEOBResumSFits.c, line 2858
    alp21_55_0 = 0.190947
    b_1_55     = -1.8780; b_2_55 = 0.8467; b_3_55 = 0.0315
    c_1_55     = -1.8619; c_2_55 = 0.8936; c_3_55 = -0.0293
    # -------------------------------------------------------------------------------- #

    if ((l==2) and (m==2)):
        alpha21_22 = _Y(alp21_22_0, b_1_22, b_2_22, b_3_22, c_1_22, c_2_22, c_3_22, af, af2, af3)
        res = alpha21_22

    elif ((l==2) and (m==1)):
        alpha21_21 = _Y(alp21_21_0, b_1_21, b_2_21, b_3_21, c_1_21, c_2_21, c_3_21, af, af2, af3)
        res = alpha21_21

    elif ((l==3) and (m==3)):
        alpha21_33 = _Y(alp21_33_0, b_1_33, b_2_33, b_3_33, c_1_33, c_2_33, c_3_33, af, af2, af3)
        res = alpha21_33

    elif ((l==3) and (m==2)):
        alpha21_32 = _Y(alp21_32_0, b_1_32, b_2_32, b_3_32, c_1_32, c_2_32, c_3_32, af, af2, af3)
        res = alpha21_32

    elif ((l==3) and (m==1)):
        alpha21_31 = _Y(alp21_31_0, b_1_31, b_2_31, b_3_31, c_1_31, c_2_31, c_3_31, af, af2, af3)
        res = alpha21_31

    elif ((l==4) and (m==4)):
        alpha21_44 = _Y(alp21_44_0, b_1_44, b_2_44, b_3_44, c_1_44, c_2_44, c_3_44, af, af2, af3)
        res = alpha21_44

    elif ((l==4) and (m==3)):
        alpha21_43 = _Y(alp21_43_0, b_1_43, b_2_43, b_3_43, c_1_43, c_2_43, c_3_43, af, af2, af3)
        res = alpha21_43

    elif ((l==4) and (m==2)):
        alpha21_42 = _Y(alp21_42_0, b_1_42, b_2_42, b_3_42, c_1_42, c_2_42, c_3_42, af, af2, af3)
        res = alpha21_42

    elif ((l==4) and (m==1)):
        alpha21_41 = _Y(alp21_41_0, b_1_41, b_2_41, b_3_41, c_1_41, c_2_41, c_3_41, af, af2, af3)
        res = alpha21_41

    elif ((l==5) and (m==5)):
        alpha21_55 = _Y(alp21_55_0, b_1_55, b_2_55, b_3_55, c_1_55, c_2_55, c_3_55, af, af2, af3)
        res = alpha21_55

    return res


def _omega1(af, l, m):
    """
    Function returning the value of omega1 for each mode, following appendix C.2 of arXiv:2001.09082v2.
    It is called in function EOBPM_SetupFitCoefficients of waveform.pyx.
    ---------------------------------------------------------------------------------------
    The fitting coefficients and equations are implemented from:
    (2,2) (2,1) (3,3) (3,2) (3,1) (4,4) (4,3) (4,2) (4,1) - arXiv:1904.09550v2, section V.D
    ---------------------------------------------------------------------------------------
    Note that, as outlined in appendix C.1 of arXiv:2001.09082v2, we use the non-spinning values (i.e.
    taken from arXiv:1904.09550v2) for the post-peak coefficients also for the spinning case.
    """

    res = 0.
    af2 = af*af; af3 = af2*af

    # -------------------------------------------------------------------------------- #
    # (2,2) - TEOBResumSFits.c, line 2802
    omg1_22_0 = 0.373672
    b_1_22    = -1.5367; b_2_22 = 0.5503; b_3_22 = 0.
    c_1_22    = -1.8700; c_2_22 = 0.9848; c_3_22 = -0.10943

    # (2,1) - TEOBResumSFits.c, line 2797
    omg1_21_0 = 0.373672
    b_1_21    = -0.79546; b_2_21 = -0.1908; b_3_21 = 0.11460
    c_1_21    = -0.96337; c_2_21 = -0.1495; c_3_21 = 0.19522

    # (3,3) - TEOBResumSFits.c, line 2831
    omg1_33_0 = 0.599443
    b_1_33    = -1.84922; b_2_33 = 0.9294; b_3_33 = -0.07613
    c_1_33    = -2.18719; c_2_33 = 1.4903; c_3_33 = -0.3014

    # (3,2) - TEOBResumSFits.c, line 2826
    omg1_32_0 = 0.599443
    b_1_32    = -0.251; b_2_32 = -0.891; b_3_32 = 0.2706
    c_1_32    = -0.475; c_2_32 = -0.911; c_3_32 = 0.4609

    # (3,1) - TEOBResumSFits.c, line 2821
    omg1_31_0 = 0.599443
    b_1_31    = -0.70941; b_2_31 = -0.16975; b_3_31 = 0.08559
    c_1_31    = -0.82174; c_2_31 = -0.16792; c_3_31 = 0.14524

    # (4,4) - TEOBResumSFits.c, line 2851
    omg1_44_0 = 0.809178
    b_1_44    = -1.83156; b_2_44 = 0.9016; b_3_44 = -0.06579
    c_1_44    = -2.17745; c_2_44 = 1.4753; c_3_44 = -0.2961

    # (4,3) - TEOBResumSFits.c, line 2846
    omg1_43_0 = 0.809178
    b_1_43    = -1.8397; b_2_43 = 0.9616; b_3_43 = -0.11339
    c_1_43    = -2.0979; c_2_43 = 1.3701; c_3_43 = -0.2675

    # (4,2) - TEOBResumSFits.c, line 2841
    omg1_42_0 = 0.809178
    b_1_42    = -0.6644; b_2_42 = -0.3357; b_3_42 = 0.1425
    c_1_42    = -0.8366; c_2_42 = -0.2921; c_3_42 = 0.2254

    # (4,1) - TEOBResumSFits.c, line 2836
    omg1_41_0 = 0.809178
    b_1_41    = -0.68647; b_2_41 = -0.1852590; b_3_41 = 0.0934997
    c_1_41    = -0.77272; c_2_41 = -0.1986852; c_3_41 = 0.1485093

    # (5,5) - TEOBResumSFits.c, line 2856
    omg1_55_0 = 1.012295
    b_1_55    = -1.5659; b_2_55 = 0.5783; b_3_55 = 0.
    c_1_55    = -1.9149; c_2_55 = 1.0668; c_3_55 = 0.14663
    # -------------------------------------------------------------------------------- #

    if ((l==2) and (m==2)):
        omega1_22 = _Y(omg1_22_0, b_1_22, b_2_22, b_3_22, c_1_22, c_2_22, c_3_22, af, af2, af3)
        res = omega1_22

    elif ((l==2) and (m==1)):
        omega1_21 = _Y(omg1_21_0, b_1_21, b_2_21, b_3_21, c_1_21, c_2_21, c_3_21, af, af2, af3)
        res = omega1_21

    elif ((l==3) and (m==3)):
        omega1_33 = _Y(omg1_33_0, b_1_33, b_2_33, b_3_33, c_1_33, c_2_33, c_3_33, af, af2, af3)
        res = omega1_33

    elif ((l==3) and (m==2)):
        omega1_32 = _Y(omg1_32_0, b_1_32, b_2_32, b_3_32, c_1_32, c_2_32, c_3_32, af, af2, af3)
        res = omega1_32

    elif ((l==3) and (m==1)):
        omega1_31 = _Y(omg1_31_0, b_1_31, b_2_31, b_3_31, c_1_31, c_2_31, c_3_31, af, af2, af3)
        res = omega1_31

    elif ((l==4) and (m==4)):
        omega1_44 = _Y(omg1_44_0, b_1_44, b_2_44, b_3_44, c_1_44, c_2_44, c_3_44, af, af2, af3)
        res = omega1_44

    elif ((l==4) and (m==3)):
        omega1_43 = _Y(omg1_43_0, b_1_43, b_2_43, b_3_43, c_1_43, c_2_43, c_3_43, af, af2, af3)
        res = omega1_43

    elif ((l==4) and (m==2)):
        omega1_42 = _Y(omg1_42_0, b_1_42, b_2_42, b_3_42, c_1_42, c_2_42, c_3_42, af, af2, af3)
        res = omega1_42

    elif ((l==4) and (m==1)):
        omega1_41 = _Y(omg1_41_0, b_1_41, b_2_41, b_3_41, c_1_41, c_2_41, c_3_41, af, af2, af3)
        res = omega1_41

    elif ((l==5) and (m==5)):
        omega1_55 = _Y(omg1_55_0, b_1_55, b_2_55, b_3_55, c_1_55, c_2_55, c_3_55, af, af2, af3)
        res = omega1_55

    return res



#############################################################
# Utils Section 3: Amplitude and phase fitting coefficients #
# functions: c3_A, c3_phi, c4_phi                           #
#############################################################

def _c3_A(nu, X12, S_hat, a12, l, m):
    """
    Function returning the coefficient c3_A of the ringdown model in section V.A of arXiv:1904.09550v2.
    It is called in function EOBPM_SetupFitCoefficients of waveform.pyx.
    -----------------------------------------------------------------------------------
    The fitting coefficients and equations are implemented from:
    (2,2) (3,3) (4,4) (5,5) - arXiv:2001.09082v2, appendix C.3
    (2,1) (3,2) (3,1) (4,3) (4,2) (4,1) - arXiv:1904.09550v2, section V.D
    -----------------------------------------------------------------------------------
    Note that paper arXiv:1904.09550v2 refers to the non-spinning case, while arXiv:2001.09082v2 includes spin effects.
    """

    res = 0.,
    nu2 = nu*nu; nu3 = nu2*nu; S_hat2 = S_hat*S_hat; S_hat3 = S_hat2*S_hat; S_hat4 = S_hat3*S_hat; X12_2 = X12*X12; a12_2 = a12*a12

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    # (2,2) - TEOBResumSFits.c, line 2577-2580
    b_0_A22 = -0.5585 + 0.81196*nu; b_1_A22 = -0.398576 + 0.1659421*X12; b_2_A22 = 0.099805 -0.2560047*X12; b_3_A22 = 0.72125 -0.9418946*X12

    # (2,1) - TEOBResumSFits.c, line 2700
    c_3_A21_num = 0.23882 -2.2982*nu + 5.7022*nu2
    c_3_A21_den = 1 - 7.7463*nu + 27.266*nu2

    # (3,3) - TEOBResumSFits.c, line 2591-2594
    b_0_A33 = -0.41455 + 1.3225*nu; b_1_A33 = -0.3502608 + 1.587606*X12 -1.555325*X12_2

    # (3,2) - TEOBResumSFits.c, line 2716
    c_3_A32_num = 0.1877 -3.0017*nu + 19.501*nu2
    c_3_A32_den = 1 -1.8199*nu
    c_3_A32_exp = -703.67

    # (3,1) - TEOBResumSFits.c, line 2705
    c_3_A31_num = 3.5042 -55.171*nu + 217*nu2
    c_3_A31_den = 1 -15.749*nu + 605.17*nu3

    # (4,4) - TEOBResumSFits.c, line 2597-2599
    b_0_A44 = -0.41591 + 3.2099*nu; b_1_A44 = -9.614738*nu; b_2_A44 = 122.461125*nu

    # (4,3) - TEOBResumSFits.c, line 2741
    c_3_A43_num = -0.02833 + 2.8738*nu -31.503*nu2 + 93.513*nu3
    c_3_A43_den = 1 - 10.051*nu + 156.14*nu3

    # (4,2) - TEOBResumSFits.c, line 2731
    c_3_A42_num = 0.27143 -2.2629*nu + 4.6249*nu2
    c_3_A42_den = 1 -7.6762*nu + 15.117*nu2

    # (4,1) - TEOBResumSFits.c, line 2721
    c_3_A41 = 11.47 + 10.936*nu

    # (5,5) - TEOBResumSFits.c, line 2604-2609
    b_0_A55 = -0.5970347579708830 + 9.1187519178640084*nu; b_1_A55 = -2.055335 -0.585373*X12; b_2_A55 = -12.631409 + 19.271346*X12
    # ------------------------------------------------------------------------------------------------------------------------------------------------ #


    if ((l==2) and (m==2)):
        
        c_3_A22 = b_0_A22 + b_1_A22*S_hat + b_2_A22*S_hat2 + b_3_A22*S_hat3    # eq. C107 of arXiv:2001.09082v2
        res     = c_3_A22

    elif ((l==2) and (m==1)):

        c_3_A21 = c_3_A21_num/c_3_A21_den   # table IV of arXiv:1904.09550v2
        res     = c_3_A21

    elif ((l==3) and (m==3)):
        
        c_3_A33 = b_0_A33 + b_1_A33*a12    # eq. C108 of arXiv:2001.09082v2
        res     = c_3_A33

    elif ((l==3) and (m==2)):

        exp_A32 = np.exp(c_3_A32_exp * (nu-2./9.)*(nu-2./9.))   # table IV of arXiv:1904.09550v2
        c_3_A32 = c_3_A32_num/c_3_A32_den - exp_A32
        res     = c_3_A32

    elif ((l==3) and (m==1)):

        c_3_A31 = c_3_A31_num/c_3_A31_den   # table IV of arXiv:1904.09550v2
        res     = c_3_A31

    elif ((l==4) and (m==4)):

        c_3_A44 = b_0_A44 + b_1_A44*S_hat + b_2_A44*S_hat2    # eq. C110 of arXiv:2001.09082v2
        res     = c_3_A44

    elif ((l==4) and (m==3)):

        c_3_A43 = c_3_A43_num/c_3_A43_den   # table IV of arXiv:1904.09550v2
        res     = c_3_A43

    elif ((l==4) and (m==2)):

        c_3_A42 = c_3_A42_num/c_3_A42_den   # table IV of arXiv:1904.09550v2
        res     = c_3_A42

    elif ((l==4) and (m==1)):

        res = c_3_A41   # table IV of arXiv:1904.09550v2

    elif ((l==5) and (m==5)):
        
        c_3_A55 = b_0_A55 + b_1_A55*a12 + b_2_A55*a12_2    # eq. C111 of arXiv:2001.09082v2
        res     = c_3_A55

    return res


def _c3_phi(nu, X12, S_hat, l, m):
    """
    Function returning the coefficient c3_phi of the ringdown model in section V.A of arXiv:1904.09550v2.
    It is called in function EOBPM_SetupFitCoefficients of waveform.pyx.
    -----------------------------------------------------------------------------------
    The fitting coefficients and equations are implemented from:
    (2,2) (3,3) (4,4) (5,5) - arXiv:2001.09082v2, appendix C.3
    (2,1) (3,2) (3,1) (4,3) (4,2) (4,1) - arXiv:1904.09550v2, section V.D
    -----------------------------------------------------------------------------------
    Note that paper arXiv:1904.09550v2 refers to the non-spinning case, while arXiv:2001.09082v2 includes spin effects.
    """

    res = 0.
    nu2 = nu*nu; nu3 = nu2*nu; S_hat2 = S_hat*S_hat; S_hat3 = S_hat2*S_hat; S_hat4 = S_hat3*S_hat; X12_2 = X12*X12
    nu_d = 0.08271; nu_d2 = nu_d*nu_d   # TEOBResumSFits.c, line 2711

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    # (2,2) - TEOBResumSFits.c, line 2613-2617
    b_0_phi22 = 3.8436 + 0.71565*nu; b_1_phi22 = 5.12794 -1.323643*X12; b_2_phi22 = 9.9136 -3.555007*X12; b_3_phi22 = -4.1075 + 7.011267*X12; b_4_phi22 = -31.5562 + 32.737824*X12

    # (2,1) - TEOBResumSFits.c, line 2701
    c_3_phi21_num = 2.6269 - 37.677*nu + 181.61*nu2
    c_3_phi21_den = 1 - 16.082*nu + 89.836*nu2

    # (3,3) - TEOBResumSFits.c, line 2632
    b_0_phi33 = 3.0611 -6.1597*nu

    # (3,2) - TEOBResumSFits.c, line 2717
    c_3_phi32_num = 0.90944 - 1.8924*nu + 3.6848*nu2
    c_3_phi32_den = 1 - 8.9739*nu + 21.024*nu2

    # (3,1) - TEOBResumSFits.c, line 2708-2712
    c_3_phi31_num = -6.1719 + 29.617*nu + 254.24*nu2
    c_3_phi31_den = 1 - 1.5435*nu
    c_3_phi31_num_nu = -6.1719 + 29.617*nu_d + 254.24*nu_d2
    c_3_phi31_den_nu = 1 - 1.5435*nu_d

    # (4,4) - TEOBResumSFits.c, line 2637
    b_0_phi44 = (3.6662-30.072*nu + 76.371*nu2)/(1 - 3.5522*nu)

    # (4,3) - TEOBResumSFits.c, line 2742
    c_3_phi43_num = 2.284 - 23.817*nu + 70.952*nu2
    c_3_phi43_den = 1 - 10.909*nu + 30.723*nu2

    # (4,2) - TEOBResumSFits.c, line 2732
    c_3_phi42_num = 2.2065 - 17.629*nu + 65.372*nu2
    c_3_phi42_den = 1 - 4.7744*nu + 3.1876*nu2

    # (4,1) - TEOBResumSFits.c, line 2721-2727
    c_3_phi41    = -6.0286 + 46.632*nu
    c_3_phi41_nu = -2.1747

    # (5,5) - TEOBResumSFits.c, line 2642-2647
    b_0_phi55 = 4.226238 -59.69284*nu + 373.312597*nu2; b_1_phi55 = -2.687133 + 4.873750*X12; b_2_phi55 = -14.629684 + 19.696954*X12
    # ------------------------------------------------------------------------------------------------------------------------------------------------ #


    if ((l==2) and (m==2)):
        
        c_3_phi22 = b_0_phi22 + b_1_phi22*S_hat + b_2_phi22*S_hat2 + b_3_phi22*S_hat3 + b_4_phi22*S_hat4    # eq. C107 of arXiv:2001.09082v2
        res       = c_3_phi22

    elif ((l==2) and (m==1)):

        c_3_phi21 = c_3_phi21_num/c_3_phi21_den   # table IV of arXiv:1904.09550v2
        res       = c_3_phi21

    elif ((l==3) and (m==3)):

        c_3_phi33 = b_0_phi33    # TEOBResumSFits.c; line 2632
        res       = c_3_phi33

    elif ((l==3) and (m==2)):

        c_3_phi32 = c_3_phi32_num/c_3_phi32_den   # table IV of arXiv:1904.09550v2
        res       = c_3_phi32

    elif ((l==3) and (m==1)):

        if (nu>0.08271):
            c_3_phi31 = c_3_phi31_num/c_3_phi31_den   # table IV of arXiv:1904.09550v2
            res       = c_3_phi31

        else:
            c_3_phi31_nu = c_3_phi31_num_nu/c_3_phi31_den_nu
            res          = c_3_phi31_nu

    elif ((l==4) and (m==4)):

        c_3_phi44 = b_0_phi44    # TEOBResumSFits.c; line 2637
        res       = c_3_phi44

    elif ((l==4) and (m==3)):

        c_3_phi43 = c_3_phi43_num/c_3_phi43_den   # table IV of arXiv:1904.09550v2
        res       = c_3_phi43

    elif ((l==4) and (m==2)):

        c_3_phi42 = c_3_phi42_num/c_3_phi42_den   # table IV of arXiv:1904.09550v2
        res       = c_3_phi42

    elif ((l==4) and (m==1)):

        if (nu>=10./121.):
            res = c_3_phi41   # table IV of arXiv:1904.09550v2

        else:   res = c_3_phi41_nu

    elif ((l==5) and (m==5)):
        
        c_3_phi55 = b_0_phi55 + b_1_phi55*S_hat + b_2_phi55*S_hat2    # eq. C111 of arXiv:2001.09082v2
        res     = c_3_phi55

    return res


def _c4_phi(nu, X12, S_hat, l, m):
    """
    Function returning the coefficient c4_phi of the ringdown model in section V.A of arXiv:1904.09550v2.
    It is called in function EOBPM_SetupFitCoefficients of waveform.pyx.
    -----------------------------------------------------------------------------------
    The fitting coefficients and equations are implemented from:
    (2,2) (3,3) (4,4) (5,5) - arXiv:2001.09082v2, appendix C.3
    (2,1) (3,2) (3,1) (4,3) (4,2) (4,1) - arXiv:1904.09550v2, section V.D
    -----------------------------------------------------------------------------------
    Note that paper arXiv:1904.09550v2 refers to the non-spinning case, while arXiv:2001.09082v2 includes spin effects.
    """

    res = 0.
    nu2 = nu*nu; nu4 = nu2*nu2; S_hat2 = S_hat*S_hat; S_hat3 = S_hat2*S_hat; S_hat4 = S_hat3*S_hat; X12_2 = X12*X12

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    # (2,2) - TEOBResumSFits.c, line 2651-2653
    b_0_phi22 = 1.4736 + 2.2337*nu; b_1_phi22 = 8.26539 + 0.779683*X12; b_2_phi22 = 14.2053 -0.069638*X12

    # (2,1) - TEOBResumSFits.c, line 2702
    c_4_phi21_num = 4.355 - 53.763*nu + 188.06*nu2
    c_4_phi21_den = 1 - 18.427*nu + 147.16*nu2

    # (3,3) - TEOBResumSFits.c, line 2666
    b_0_phi33 = 1.789 -5.6684*nu

    # (3,2) - TEOBResumSFits.c, line 2718
    c_4_phi32_num = 2.3038 - 50.79*nu + 334.41*nu2
    c_4_phi32_den = 1 - 18.326*nu + 99.54*nu2

    # (3,1) - TEOBResumSFits.c, line 2706
    c_4_phi31 = 3.6485 + 5.4536*nu

    # (4,4) - TEOBResumSFits.c, line 2671
    b_0_phi44 = 0.21595 + 23.216*nu

    # (4,3) - TEOBResumSFits.c, line 2743
    c_4_phi43_num = 2.4966 - 6.2043*nu
    c_4_phi43_den = 1 - 252.47*nu4

    # (4,2) - TEOBResumSFits.c, line 2734-2737
    c_4_phi42_num_nu_a = 132.56 - 1155.5*nu + 2516.8*nu2
    c_4_phi42_den_nu_a = 1 - 3.8231*nu
    c_4_phi42_num_nu_b = -0.58736 + 16.401*nu
    c_4_phi42_den_nu_b = 1 - 4.5202*nu

    # (4,1) - TEOBResumSFits.c, line 2722
    c_4_phi41 = 1.6629 + 11.497*nu

    # (5,5) - TEOBResumSFits.c, line 2676-2681
    b_0_phi55 = 1.3639723340485870 + 14.9111373110275380*nu; b_1_phi55 = 7.198729 -3.870998*X12; b_2_phi55 = -25.992190 + 36.882645*X12
    # ------------------------------------------------------------------------------------------------------------------------------------------------ #


    if ((l==2) and (m==2)):
        
        c_4_phi22 = b_0_phi22 + b_1_phi22*S_hat + b_2_phi22*S_hat2    # eq. C107 of arXiv:2001.09082v2
        res       = c_4_phi22

    elif ((l==2) and (m==1)):

        c_4_phi21 = c_4_phi21_num/c_4_phi21_den   # table IV of arXiv:1904.09550v2
        res       = c_4_phi21

    elif ((l==3) and (m==3)):

        c_4_phi33 = b_0_phi33    # TEOBResumSFits.c, line 2666
        res       = c_4_phi33

    elif ((l==3) and (m==2)):

        c_4_phi32 = c_4_phi32_num/c_4_phi32_den   # table IV of arXiv:1904.09550v2
        res       = c_4_phi32

    elif ((l==3) and (m==1)):

        res = c_4_phi31   # table IV of arXiv:1904.09550v2

    elif ((l==4) and (m==4)):

        c_4_phi44 = b_0_phi44    # TEOBResumSFits.c, line 2671
        res       = c_4_phi44

    elif ((l==4) and (m==3)):

        c_4_phi43 = c_4_phi43_num/c_4_phi43_den   # table IV of arXiv:1904.09550v2
        res       = c_4_phi43

    elif ((l==4) and (m==2)):   # eq. 5.20 of arXiv:1904.09550v2

        if   (nu>=2.5/12.25):
            c_4_phi42 = c_4_phi42_num_nu_a/c_4_phi42_den_nu_a
            res       = c_4_phi42

        else:
            c_4_phi42 = c_4_phi42_num_nu_b/c_4_phi42_den_nu_b
            res       = c_4_phi42

    elif ((l==4) and (m==1)):

        res = c_4_phi41   # table IV of arXiv:1904.09550v2

    elif ((l==5) and (m==5)):

        c_4_phi55 = b_0_phi55 + b_1_phi55*S_hat + b_2_phi55*S_hat2    # eq. C111 of arXiv:2001.09082v2
        res     = c_4_phi55

    return res



#################################################
# Utils Section 4: Fits for peak quantities     #
# functions: dOmega, amplitude_peak, omega_peak #
#################################################

def _dOmega(omega1, Mf, omega_peak):
    """ In the papers is referred to as Delta_omega, e.g. see in section V.A of arXiv:1904.09550v2 """
    return omega1 - Mf * omega_peak


def _amplitude_peak(nu, X12, S_hat, a12, S_bar, a0, omega_peak, l, m):
    """
    Function returning the value of A at the peak, following appendix C.2 of arXiv:2001.09082v2.
    It is called in function EOBPM_SetupFitCoefficients of waveform.pyx.
    -----------------------------------------------------------------------------------
    The fitting coefficients and equations are implemented from:
    (2,2) (2,1) (3,3) (3,2) (4,4) (4,3) (4,2) (5,5) - arXiv:2001.09082v2, appendix C.3
    (3,1) (4,1) - arXiv:1904.09550v2, section V.D
    -----------------------------------------------------------------------------------
    Note that paper arXiv:1904.09550v2 refers to the non-spinning case, while arXiv:2001.09082v2 includes spin effects.
    -----------------------------------------------------------------------------------
    Note: To fit the amplitude peak, the waveform is scaled as in eqs. C6-C12 of arXiv:2001.09082v2,
          but the amplitude peak in eqs. 7-8 of arXiv:1406.0401v2 is just nu-rescaled (see eq. (1) of arXiv:1606.03952v4).
    """

    res = 0.
    nu2 = nu*nu; X12_2 = X12*X12; S_hat2 = S_hat*S_hat; S_hat3 = S_hat2*S_hat; S_bar2 = S_bar*S_bar; a0_2 = a0*a0; a12_2 = a12*a12; S_bar_21 = -abs(S_bar); a12_33 = abs(a12)

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    # (2,2) - TEOBResumSFits.c, line 2423-2431
    A22_0    = 1.44959     # TEOBResumSFits.c, line 1993
    a_1_A22  = -0.041285; a_2_A22  = 1.5971
    b_1_A22  = -0.741;    b_2_A22  = -0.0887;     b_3_A22  = -1.094
    c_11_A22 = 0.4446696; c_12_A22 = -0.3254310; c_31_A22 = 0.4582812; c_32_A22 = -0.2124477

    # (2;1) - TEOBResumSFits.c; line 2375-2389
    A21_0    = 0.5238781992     # TEOBResumSFits.c; line 1992
    a_1_A21  = 3.3362232268; a_2_A21  = 3.4708521429; a_3_A21  = 4.7623643259
    b_1_A21  = -0.4281863;   b_2_A21  = -0.335659;    b_3_A21  = 0.828923
    c_11_A21 = 0.891139;     c_12_A21 = -5.191702;    c_21_A21 = 3.480139;   c_22_A21 = 10.237782; c_31_A21 = -13.867475; c_32_A21 = 10.525510

    # (3,3) - TEOBResumSFits.c, line 2458-2468
    A33_0    = 0.5660165890     # TEOBResumSFits.c, line 1996
    a_1_A33  = -0.22523;   a_2_A33  = 3.0569;    a_3_A33  = -0.396851
    b_1_A33  = 0.100069;   b_2_A33  = -0.455859
    c_11_A33 = -0.401156;  c_12_A33 = -0.141551; c_13_A33 = -15.4949; c_21_A33 = 1.84962; c_22_A33 = -2.03512; c_23_A33 = -4.92334

    # (3;2) - TEOBResumSFits.c; line 2435-2447
    A32_0    = 0.1990192432     # TEOBResumSFits.c; line 1995
    a_1_A32  = -6.06831;   a_2_A32  = 10.7505;  a_3_A32  = -3.68883
    b_1_A32  = -0.258378;  b_2_A32  = 0.679163
    c_11_A32 = 4.36263;    c_12_A32 = -12.5897; c_13_A32 = -7.73233; c_14_A32 = 16.2082; c_21_A32 = 3.04724; c_22_A32 = 46.5711; c_23_A32 = 2.10475; c_24_A32 = 56.9136

    # (3,1) - TEOBResumSFits.c, line 2690
    A31_0   = 0.0623783     # TEOBResumSFits.c, line 1994
    n_1_A31 = -5.49;   n_2_A31 = 10.915

    # (4,4) - TEOBResumSFits.c, line 2536-2553
    A44_0    = 0.2766182761     # TEOBResumSFits.c, line 2000
    a_1_A44  = -3.7082;   a_2_A44  = 0.280906;  a_3_A44  = -3.71276
    b_1_A44  = -0.316647; b_2_A44  = -0.062423; b_3_A44  = -0.852876
    c_11_A44 = 1.2436;    c_12_A44 = -1.60555;  c_13_A44 = -4.05685; c_14_A44 = 1.59143; c_21_A44 = 0.837418; c_22_A44 = -2.93528; c_23_A44 = -11.5591; c_24_A44 = 34.1863; c_31_A44 = 0.950035; c_32_A44 = 7.95168; c_33_A44 = -1.26899; c_34_A44 = -9.72147

    # (4,3) - TEOBResumSFits.c, line 2510-2531
    A43_0      = 0.0941569508     # TEOBResumSFits.c, line 1999
    a_1_A43    = -5.74386;    a_2_A43    = 12.6016;     a_3_A43    = -3.27435
    b_1_A43    = -0.02132252; b_2_A43    = 0.02592749;  b_3_A43    = -0.826977
    b_1_A43_nu = 0.00452129;  b_2_A43_nu = -0.00471163; b_3_A43_nu = 0.0291409;  b_4_A43_nu = -0.351031
    c_11_A43   = 0.249099;    c_12_A43   = -7.345984;   c_13_A43   = 108.923746; c_21_A43 = -0.104206; c_22_A43 = 7.073534; c_23_A43 = -44.374738; c_31_A43 = 3.545134; c_32_A43 = 1.341375; c_33_A43 = -19.552083

    # (4;2) - TEOBResumSFits.c; line 2483-2498
    A42_0    = 0.0314363901     # TEOBResumSFits.c; line 1998
    a_1_A42  = -4.56243;   a_2_A42  = 6.4522
    b_1_A42  = -1.63682;   b_2_A42  = 0.854459; b_3_A42  = 0.120537;  b_4_A42  = -0.399718
    c_11_A42 = 6.53943;    c_12_A42 = -4.00073; c_21_A42 = -0.638688; c_22_A42 = -3.94066; c_31_A42 = -0.482148; c_32_A42 = -3.9999999923319502; c_41_A42 = 1.25617; c_42_A42 = -4.04848

    # (4;1) - TEOBResumSFits.c; line 2695
    A41_0   = 0.00925061     # TEOBResumSFits.c; line 1997
    n_1_A41 = -8.4449;  n_2_A41 = 26.825
    d_1_A41 = -1.2565

    # (5;5) - TEOBResumSFits.c; line 2565-2572
    A55_0    = 0.151492     # TEOBResumSFits.c; line 2001
    a_1_A55  = -0.9750925916632546; a_2_A55  = 11.2008774621215608
    b_1_A55  = 0.04360530;          b_2_A55  = -0.5769451
    c_11_A55 = 5.720690;            c_12_A55 = 44.868515;         c_21_A55 = 12.777090; c_22_A55 = -42.548247
    # ------------------------------------------------------------------------------------------------------------------------------------------------ #


    if ((l==2) and (m==2)): # eqs. C17-C20 of arXiv:2001.09082v2

        b1_A = (b_1_A22 + c_11_A22*X12)/(1 + c_12_A22*X12)
        b2_A =  b_2_A22
        b3_A = (b_3_A22 + c_31_A22*X12)/(1 + c_32_A22*X12)

        A22_orb  = 1 + a_1_A22*nu + a_2_A22*nu2
        num_spin = 1 + b1_A*S_hat + b2_A*S_hat2
        den_spin = 1 + b3_A*S_hat

        A22_spin = num_spin/den_spin

        res = A22_0 * A22_orb * A22_spin

        scale = 1 - S_hat*omega_peak # eq. C6 of arXiv:2001.09082v2
        res = res * scale


    elif ((l==2) and (m==1)):   # eqs. C21-C24 of arXiv:2001.09082v2

        b1_A = b_1_A21 + c_11_A21*nu + c_12_A21*nu2
        b2_A = b_2_A21 + c_21_A21*nu + c_22_A21*nu2
        b3_A = b_3_A21 + c_31_A21*nu + c_32_A21*nu2

        num_orb = 1 + a_1_A21*nu + a_2_A21*nu2
        den_orb = 1 + a_3_A21*nu

        if (nu==0.25):    # TEOBResumSFits.c, line 2382
            num_spin = b1_A*S_bar_21 + b2_A*S_bar_21*S_bar_21
            den_spin = 1 + b3_A*S_bar_21

        else:
            num_spin = b1_A*S_bar + b2_A*S_bar2
            den_spin = 1 + b3_A*S_bar

        A21_orb  = num_orb/den_orb
        A21_spin = num_spin/den_spin

        res = A21_0 * X12 * A21_orb + A21_spin


    elif ((l==3) and (m==3)):   # eqs. C33-C37 of arXiv:2001.09082v2

        b1_A = (b_1_A33 + c_11_A33*nu)/(1 + c_12_A33*nu + c_13_A33*nu2)
        b2_A = (b_2_A33 + c_21_A33*nu)/(1 + c_22_A33*nu + c_23_A33*nu2)

        num_orb  = 1 + a_1_A33*nu + a_2_A33*nu2
        den_orb  = 1 + a_3_A33*nu

        if (nu==0.25):    # TEOBResumSFits.c, line 2472
            num_spin = b1_A*a12_33
            den_spin = 1 + b2_A*a12_33

        else:
            num_spin = b1_A*a12
            den_spin = 1 + b2_A*a12

        A33_orb  = num_orb/den_orb
        A33_spin = num_spin/den_spin

        res = A33_0 * X12 * A33_orb + A33_spin


    elif ((l==3) and (m==2)):   # eqs. C46-C49 of arXiv:2001.09082v2

        b1_A = (b_1_A32 + c_11_A32*nu + c_12_A32*nu2)/(1 + c_13_A32*nu + c_14_A32*nu2)
        b2_A = (b_2_A32 + c_21_A32*nu + c_22_A32*nu2)/(1 + c_23_A32*nu + c_24_A32*nu2)

        num_orb  = 1 + a_1_A32*nu + a_2_A32*nu2
        den_orb  = 1 + a_3_A32*nu
        num_spin = 1 + b1_A*a0
        den_spin = 1 + b2_A*a0

        A32_orb  = num_orb/den_orb
        A32_spin = num_spin/den_spin

        res = A32_0 * (1-3*nu) * A32_orb * A32_spin

        scale = 1 + a0*(omega_peak/2)**(1./3)  # TEOBResumSFits.c, line 2451
        res = res * scale


    elif ((l==3) and (m==1)):   # eq. 5.19 of arXiv:1904.09550v2
        '''
        if (nu==0.25):    # TEOBResumSFits.c, line 2685
            res = 0.

        else:
            A31_orb = 1 + n_1_A31*nu + n_2_A31*nu2

            res = A31_0 * X12 * A31_orb'''

        res = 0     # not implemented in TEOBResumSFits.c, line 2452

    elif ((l==4) and (m==4)):   # eqs. C61-C64 of arXiv:2001.09082v2

        b1_A = (b_1_A44 + c_11_A44*nu + c_12_A44*nu2)/(1 + c_13_A44*nu + c_14_A44*nu2)
        b2_A = (b_2_A44 + c_21_A44*nu + c_22_A44*nu2)/(1 + c_23_A44*nu + c_24_A44*nu2)
        b3_A = (b_3_A44 + c_31_A44*nu + c_32_A44*nu2)/(1 + c_33_A44*nu + c_34_A44*nu2)

        num_orb  = 1 + a_1_A44*nu + a_2_A44*nu2
        den_orb  = 1 + a_3_A44*nu
        num_spin = 1 + b1_A*S_hat + b2_A*S_hat2
        den_spin = 1 + b3_A*S_hat

        A44_orb  = num_orb/den_orb
        A44_spin = num_spin/den_spin

        res = A44_0 * (1-3*nu) * A44_orb * A44_spin

        scale = 1 - 0.5*S_hat*omega_peak  # eq. C10 of arXiv:2001.09082v2
        res = res * scale


    elif ((l==4) and (m==3)):   # eqs. C73-C77 of arXiv:2001.09082v2

        if (nu==0.25):  # TEOBResumSFits.c, line 2515
            num_peak = b_1_A43_nu + b_2_A43_nu*a12 + b_3_A43_nu*a12_2
            den_peak = 1 + b_4_A43_nu*a12
            A43_peak = num_peak/den_peak
            
            res = A43_peak
        
        else:
            b1_A = (b_1_A43 + c_11_A43*nu)/(1 + c_12_A43*nu + c_13_A43*nu2)
            b2_A = (b_2_A43 + c_21_A43*nu)/(1 + c_22_A43*nu + c_23_A43*nu2)
            b3_A = (b_3_A43 + c_31_A43*nu)/(1 + c_32_A43*nu + c_33_A43*nu2)

            num_orb  = 1 + a_1_A43*nu + a_2_A43*nu2
            den_orb  = 1 + a_3_A43*nu
            num_spin = b1_A*a0 + b2_A*a0_2
            den_spin = 1 + b3_A*a0

            A43_orb  = num_orb/den_orb
            A43_spin = num_spin/den_spin

            res = A43_0 * X12 * (1-2*nu) * A43_orb + A43_spin


    elif ((l==4) and (m==2)):   # eqs. C86-C89 of arXiv:2001.09082v2

        b1_A = (b_1_A42 + c_11_A42*nu)/(1 + c_12_A42*nu)
        b2_A = (b_2_A42 + c_21_A42*nu)/(1 + c_22_A42*nu)
        b3_A = (b_3_A42 + c_31_A42*nu)/(1 + c_32_A42*nu)
        b4_A = (b_4_A42 + c_41_A42*nu)/(1 + c_42_A42*nu)

        A42_orb  = 1 + a_1_A42*nu + a_2_A42*nu2
        num_spin = 1 + b1_A*S_hat + b2_A*S_hat2
        den_spin = 1 + b3_A*S_hat + b4_A*S_hat2

        A42_spin = num_spin/den_spin

        res = A42_0 * (1-3*nu) * A42_orb * A42_spin

        scale = 1 + a0*(omega_peak/2)**(1./3)  # TEOBResumSFits.c, line 2506
        res = res * scale


    elif ((l==4) and (m==1)):   # eq. 5.19 of arXiv:1904.09550v2

        '''num_orb = 1 + n_1_A41*nu + n_2_A41*nu2
        den_orb = 1 + d_1_A41*nu

        A41_orb = num_orb/den_orb

        res = A41_0 * A41_orb'''

        res = 0     # TEOBResumSFits.c, line 2695


    elif ((l==5) and (m==5)):   # eqs. C98-C102 of arXiv:2001.09082v2

        b1_A = b_1_A55/(1 + c_11_A55*nu + c_12_A55*nu2)
        b2_A = b_2_A55/(1 + c_21_A55*nu + c_22_A55*nu2)

        A55_orb  = 1 + a_1_A55*nu + a_2_A55*nu2
        num_spin = b1_A*a12
        den_spin = 1 + b2_A*a12

        A55_spin = num_spin/den_spin

        res = A55_0 * X12 * (1-2*nu) * A55_orb + A55_spin


    return res


def _omega_peak(nu, X12, S_hat, a0, l, m):
    """
    Function returning the value of Omega at the peak of the strain, following appendix C.2 of arXiv:2001.09082v2.
    It is called in function EOBPM_SetupFitCoefficients of waveform.pyx.
    -----------------------------------------------------------------------------------
    The fitting coefficients and equations are implemented from:
    (2,2) (2,1) (3,3) (3,2) (4,4) (4,3) (4,2) (5,5) - arXiv:2001.09082v2, appendix C.3
    (3,1) (4,1) - arXiv:1904.09550v2, section V.D
    -----------------------------------------------------------------------------------
    Note that paper arXiv:1904.09550v2 refers to the non-spinning case, while arXiv:2001.09082v2 includes spin effects.
    """

    res = 0.
    nu2 = nu*nu; X12_2 = X12*X12; S_hat2 = S_hat*S_hat; S_hat3 = S_hat2*S_hat; a0_2 = a0*a0

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    # (2,2) - TEOBResumSFits.c, line 2226-2236
    omg22_0    = 0.273356     # TEOBResumSFits.c, line 2003
    a_1_omg22  = 0.84074;  a_2_omg22  = 1.6976
    b_1_omg22  = -0.42311; b_2_omg22  = -0.066699; b_3_omg22  = -0.83053
    c_11_omg22 = 0.066045; c_12_omg22 = -0.23876;  c_31_omg22 = 0.76819;  c_32_omg22 = -0.9201

    # (2;1) - TEOBResumSFits.c; line 2184-2189
    omg21_0   = 0.2906425497     # TEOBResumSFits.c; line 2002
    a_1_omg21 = -0.563075;  a_2_omg21 = 3.28677
    b_1_omg21 = 0.179639;   b_2_omg21 = -0.302122
    c_1_omg21 = -1.20684;   c_2_omg21 = 0.425645

    # (3;3) - TEOBResumSFits.c; line 2258-2266
    omg33_0    = 0.4541278937     # TEOBResumSFits.c; line 2006
    a_1_omg33  = 1.08224;    a_2_omg33  = 2.59333
    b_1_omg33  = -0.406161;  b_2_omg33  = -0.0647944; b_3_omg33  = -0.748126
    c_11_omg33 = 0.85777;    c_12_omg33 = -0.70066;   c_31_omg33 = 2.97025;   c_32_omg33 = -3.96242

    # (3;2) - TEOBResumSFits.c; line 2240-2248
    omg32_0    = 0.4516072248     # TEOBResumSFits.c; line 2005
    a_1_omg32  = -9.13525;   a_2_omg32  = 21.488;    a_3_omg32  = -8.81384; a_4_omg32  = 20.0595
    b_1_omg32  = -0.458126;  b_2_omg32  = 0.0474616; b_3_omg32  = -0.486049
    c_11_omg32 = 3.25319;    c_12_omg32 = 0.535555;  c_13_omg32 = -8.07905; c_21_omg32 = 1.00066; c_22_omg32 = -1.1333; c_23_omg32 = 0.601572

    # (3;1) - TEOBResumSFits.c; line 2691
    omg31_0   = 0.411755     # TEOBResumSFits.c; line 2004
    n_2_omg31 = 7.5362
    d_1_omg31 = -2.7555; d_2_omg31 = 38.572

    # (4;4) - TEOBResumSFits.c; line 2330-2353
    omg44_0    = 0.6356586393     # TEOBResumSFits.c; line 2010
    a_1_omg44  = -0.964614;  a_2_omg44  = -11.1828;   a_3_omg44  = -2.08471;   a_4_omg44  = -6.89287
    b_1_omg44  = -0.445192;  b_2_omg44  = -0.0985658; b_3_omg44  = -0.0307812; b_4_omg44  = -0.801552
    c_11_omg44 = -0.92902;   c_12_omg44 = 10.86310;   c_13_omg44 = -4.44930;   c_14_omg44 = 3.01808;   c_21_omg44 = 0; c_22_omg44 = 1.62523; c_23_omg44 = -7.70486; c_24_omg44 = 15.06517; c_41_omg44 = 0.93790; c_42_omg44 = 8.36038; c_43_omg44 = -4.85774; c_44_omg44 = 4.80446

    # (4;3) - TEOBResumSFits.c; line 2305-2320
    omg43_0    = 0.6361300619     # TEOBResumSFits.c; line 2009
    a_1_omg43  = -9.02463;   a_2_omg43  = 21.9802;   a_3_omg43  = -8.75892; a_4_omg43 = 20.5624
    b_1_omg43  = -0.973324;  b_2_omg43  = -0.109921; b_3_omg43  = -1.08036
    c_11_omg43 = 11.5224;    c_12_omg43 = -26.8421;  c_13_omg43 = -2.84285; c_21_omg43 = 3.51943; c_22_omg43 = -12.1688; c_23_omg43 = -3.96385; c_31_omg43 = 5.53433; c_32_omg43 = 3.73988; c_33_omg43 = 4.219

    # (4;2) - TEOBResumSFits.c; line 2275-2294
    omg42_0    = 0.6175331548     # TEOBResumSFits.c; line 2008
    a_1_omg42  = -7.44121;   a_2_omg42  = 14.233;   a_3_omg42  = -6.61754; a_4_omg42  = 11.4329
    b_1_omg42  = -2.37589;   b_2_omg42  = 1.97249;  b_3_omg42  = -2.36107; b_4_omg42  = 2.16383
    c_11_omg42 = 10.1045;    c_12_omg42 = -6.94127; c_13_omg42 = 12.1857;  c_21_omg42 = -1.62866; c_22_omg42 = -2.6756; c_23_omg42 = -4.7536; c_31_omg42 = 10.071; c_32_omg42 = -6.7299; c_33_omg42 = 12.0377; c_41_omg42 = -8.56139; c_42_omg42 = -5.27136; c_43_omg42 = 5.10653

    # (4;1) - TEOBResumSFits.c; line 2696
    omg41_0   = 0.552201     # TEOBResumSFits.c; line 2007
    n_1_omg41 = -10.876; n_2_omg41 = 37.904
    d_1_omg41 = -11.194; d_2_omg41 = 42.77

    # (5;5) - TEOBResumSFits.c; line 2364-2369
    omg55_0    = 0.818117     # TEOBResumSFits.c; line 2011
    a_1_omg55  = -2.8918;   a_2_omg55  = -3.2012;   a_3_omg55  = -3.773
    b_1_omg55  = -0.332703; b_2_omg55  = -0.675738
    c_11_omg55 = 1.487294;  c_12_omg55 = -2.058537; c_21_omg55 = 1.454248; c_22_omg55 = -1.301284
    # ------------------------------------------------------------------------------------------------------------------------------------------------ #


    if ((l==2) and (m==2)): # eqs. C17-C20 of arXiv:2001.09082v2

        b1_omg = (b_1_omg22 + c_11_omg22*X12)/(1 + c_12_omg22*X12)
        b2_omg =  b_2_omg22
        b3_omg = (b_3_omg22 + c_31_omg22*X12)/(1 + c_32_omg22*X12)

        omg22_orb = 1 + a_1_omg22*nu + a_2_omg22*nu2
        num_spin  = 1 + b1_omg*S_hat + b2_omg*S_hat2
        den_spin  = 1 + b3_omg*S_hat

        omg22_spin = num_spin/den_spin

        res = omg22_0 * omg22_orb * omg22_spin


    elif ((l==2) and (m==1)):   # eqs. C25-C28 of arXiv:2001.09082v2

        b1_omg = b_1_omg21 + c_1_omg21*nu
        b2_omg = b_2_omg21 + c_2_omg21*nu

        omg21_orb  = 1 + a_1_omg21*nu + a_2_omg21*nu2
        omg21_spin = 1 + b1_omg*S_hat + b2_omg*S_hat2

        res = omg21_0 * omg21_orb * omg21_spin


    elif ((l==3) and (m==3)):   # eqs. C38-C41 of arXiv:2001.09082v2

        b1_omg = (b_1_omg33 + c_11_omg33*nu)/(1 + c_12_omg33*nu)
        b2_omg =  b_2_omg33
        b3_omg = (b_3_omg33 + c_31_omg33*nu)/(1 + c_32_omg33*nu)

        omg33_orb = 1 + a_1_omg33*nu + a_2_omg33*nu2
        num_spin  = 1 + b1_omg*S_hat + b2_omg*S_hat2
        den_spin  = 1 + b3_omg*S_hat

        omg33_spin = num_spin/den_spin

        res = omg33_0 * omg33_orb * omg33_spin


    elif ((l==3) and (m==2)):   # eqs. C50-C53 of arXiv:2001.09082v2

        b1_omg = (b_1_omg32 + c_11_omg32*X12 + c_12_omg32*X12_2)/(1 + c_13_omg32*X12)
        b2_omg = (b_2_omg32 + c_21_omg32*X12 + c_22_omg32*X12_2)/(1 + c_23_omg32*X12)
        b3_omg =  b_3_omg32

        num_orb  = 1 + a_1_omg32*nu + a_2_omg32*nu2
        den_orb  = 1 + a_3_omg32*nu + a_4_omg32*nu2
        num_spin = 1 + b1_omg*a0 + b2_omg*a0_2
        den_spin = 1 + b3_omg*a0

        omg32_orb  = num_orb/den_orb
        omg32_spin = num_spin/den_spin

        res = omg32_0 * omg32_orb * omg32_spin


    elif ((l==3) and (m==1)):   # eq. 5.19 of arXiv:1904.09550v2

        if (nu==0.25):    # TEOBResumSFits.c, line 2685
            res = 0.
        
        else:
            num_orb = 1 + n_2_omg31*nu2
            den_orb = 1 + d_1_omg31*nu + d_2_omg31*nu2

            omg31_orb = num_orb/den_orb

            res = omg31_0 * omg31_orb


    elif ((l==4) and (m==4)):   # eqs. C65-C68 of arXiv:2001.09082v2

        b1_omg = (b_1_omg44 + c_11_omg44*nu + c_12_omg44*nu2)/(1 + c_13_omg44*nu + c_14_omg44*nu2)
        b2_omg = (b_2_omg44 + c_21_omg44*nu + c_22_omg44*nu2)/(1 + c_23_omg44*nu + c_24_omg44*nu2)
        b3_omg =  b_3_omg44
        b4_omg = (b_4_omg44 + c_41_omg44*nu + c_42_omg44*nu2)/(1 + c_43_omg44*nu + c_44_omg44*nu2)

        num_orb  = 1 + a_1_omg44*nu + a_2_omg44*nu2
        den_orb  = 1 + a_3_omg44*nu + a_4_omg44*nu2
        num_spin = 1 + b1_omg*S_hat + b2_omg*S_hat2 + b3_omg*S_hat3
        den_spin = 1 + b4_omg*S_hat

        omg44_orb  = num_orb/den_orb
        omg44_spin = num_spin/den_spin

        res = omg44_0 * omg44_orb * omg44_spin


    elif ((l==4) and (m==3)):   # eqs. C78-C81 of arXiv:2001.09082v2

        b1_omg = (b_1_omg43 + c_11_omg43*nu + c_12_omg43*nu2)/(1 + c_13_omg43*nu)
        b2_omg = (b_2_omg43 + c_21_omg43*nu + c_22_omg43*nu2)/(1 + c_23_omg43*nu)
        b3_omg = (b_3_omg43 + c_31_omg43*nu + c_32_omg43*nu2)/(1 + c_33_omg43*nu)

        num_orb  = 1 + a_1_omg43*nu + a_2_omg43*nu2
        den_orb  = 1 + a_3_omg43*nu + a_4_omg43*nu2
        num_spin = 1 + b1_omg*S_hat + b2_omg*S_hat2
        den_spin = 1 + b3_omg*S_hat

        omg43_orb  = num_orb/den_orb
        omg43_spin = num_spin/den_spin

        res = omg43_0 * omg43_orb * omg43_spin


    elif ((l==4) and (m==2)):   # eqs. C90-C93 of arXiv:2001.09082v2

        b1_omg = (b_1_omg42 + c_11_omg42*nu)/(1 + c_12_omg42*nu + c_13_omg42*nu2)
        b2_omg = (b_2_omg42 + c_21_omg42*nu)/(1 + c_22_omg42*nu + c_23_omg42*nu2)
        b3_omg = (b_3_omg42 + c_31_omg42*nu)/(1 + c_32_omg42*nu + c_33_omg42*nu2)
        b4_omg = (b_4_omg42 + c_41_omg42*nu)/(1 + c_42_omg42*nu + c_43_omg42*nu2)

        num_orb  = 1 + a_1_omg42*nu + a_2_omg42*nu2
        den_orb  = 1 + a_3_omg42*nu + a_4_omg42*nu2
        num_spin = 1 + b1_omg*S_hat + b2_omg*S_hat2
        den_spin = 1 + b3_omg*S_hat + b4_omg*S_hat2

        omg42_orb  = num_orb/den_orb
        omg42_spin = num_spin/den_spin

        res = omg42_0 * omg42_orb * omg42_spin


    elif ((l==4) and (m==1)):   # eq. 5.19 of arXiv:1904.09550v2

        '''num_orb = 1 + n_1_omg41*nu + n_2_omg41*nu2
        den_orb = 1 + d_1_omg41*nu + d_2_omg41*nu2

        omg41_orb = num_orb/den_orb

        res = omg41_0 * omg41_orb'''

        res = 0     # TEOBResumSFits.c, line 2696


    elif ((l==5) and (m==5)):   # eqs. C103-C106 of arXiv:2001.09082v2

        b1_omg = (b_1_omg55 + c_11_omg55*nu)/(1 + c_12_omg55*nu)
        b2_omg = (b_2_omg55 + c_21_omg55*nu)/(1 + c_22_omg55*nu)

        num_orb  = 1 + a_1_omg55*nu + a_2_omg55*nu2
        den_orb  = 1 + a_3_omg55*nu
        num_spin = 1 + b1_omg*S_hat
        den_spin = 1 + b2_omg*S_hat

        omg55_orb  = num_orb/den_orb
        omg55_spin = num_spin/den_spin

        res = omg55_0 * omg55_orb * omg55_spin


    return res



###################################################################
# Utils Section 5: Fits for remnant mass and spin                 #
# functions: JimenezFortezaRemnantMass, JimenezFortezaRemnantSpin #
###################################################################

def _JimenezFortezaRemnantMass(nu, X1, X2, chi1, chi2, M):
    """
    Compute the BH final mass with the aligned-spin NR fit by Xisco Jimenez Forteza,
    David Keitel, Sascha Husa et al.: see arXiv:1611.00332v2
    -----------------------------------------------------------------------------------
    m1, m2: component masses. chi1, chi2: dimensionless spins
    Note that it is assumed m1>m2
    -----------------------------------------------------------------------------------
    Note: In arXiv:1611.00332v2 the symmetric mass ratio is called 'eta', contrary to
          the rest of our implementation where is 'nu'.
          For clarity, we continue to use the symbol 'nu' also for the
          functions JimenezFortezaRemnantMass and JimenezFortezaRemnantSpin
    Note: In arXiv:1611.00332v2, S_hat has a slighly different definition wrt the
          rest of the implementation
    """

    X12  = X1 - X2
    X1_2 = X1*X1; X2_2 = X2*X2
    S    = (X1_2*chi1 + X2_2*chi2)/(X1_2 + X2_2)
    Dchi = chi1-chi2

    res = 0.
    nu2 = nu*nu; nu3 = nu2*nu; nu4 = nu3*nu; S_2 = S*S; S_3 = S_2*S; Dchi_2 = Dchi*Dchi

    # 1D fits - Table VII from arXiv:1611.00332v2
    a2 = 0.5610; a3 = -0.847; a4 = 3.145

    # 1D fits - Table VIII from arXiv:1611.00332v2
    b1 = -0.209; b2 = -0.197; b3 = -0.159; b5 = 2.985

    # 2D fits - Table IX from arXiv:1611.00332v2
    f20 = 4.27; f30 = 31.09; f50 = 1.56735
    f10 = -0.574752*f20 - 0.280958*f30 + 64.6408*f50 - 88.3165   # eq. 26 of arXiv:1611.00332v2

    # 3D fits - Table X from arXiv:1611.00332v2
    d10 = -0.098; d11 = -3.23; d20 = 0.0112; d30 = -0.0198; d31 = -4.92
    f11 = 15.7; f21 = 0.; f31 = -243.6; f51 = -0.58

    """
    Fit coefficients at increased numerical precision, taken from:
    https://git.ligo.org/uib-papers/finalstate2016/blob/master/LALInference/EradUIB2016v2_pyform_coeffs.txt
    git commit f490774d3593adff5bb09ae26b7efc6deab76a42
    -----------------------------------------------------------------------------------
    # 1D fits - Table VII from arXiv:1611.00332v2
    a2 = 0.5609904135313374, a3 = -0.84667563764404, a4 = 3.145145224278187

    # 1D fits - Table VIII from arXiv:1611.00332v2
    b1 = -0.2091189048177395, b2 = -0.19709136361080587, b3 = -0.1588185739358418, b5 = 2.9852925538232014

    # 2D fits - Table IX from arXiv:1611.00332v2
    f20 = 4.271313308472851, f30 = 31.08987570280556, f50 = 1.5673498395263061
    f10 = -0.574752*f20 - 0.280958*f30 + 64.6408*f50 - 88.3165   # eq. 26 of arXiv:1611.00332v2

    # 3D fits - Table X from arXiv:1611.00332v2
    d10 = -0.09803730445895877, d11 = -3.2283713377939134, d20 = 0.01118530335431078, d30 = -0.01978238971523653, d31 = -4.91667749015812
    f11 = 15.738082204419655, f21 = 0., f31 = -243.6299258830685, f51 = -0.5808669012986468
    -----------------------------------------------------------------------------------
    FIXME: URL is not still available: ask Greg where to find these coefficients.
    """

    # 1D fits
    Erad_nu_0 = a4*nu4 + a3*nu3 + a2*nu2 + (1-2.*np.sqrt(2)/3.)*nu    # eq. 21 of arXiv:1611.00332v2

    # 2D fits
    f12 = 16. - 16.*f10 - 4.*f11    # eq. 24 of arXiv:1611.00332v2
    f22 = 16. - 16.*f20 - 4.*f21
    f32 = 16. - 16.*f30 - 4.*f31
    f52 = 16. - 16.*f50 - 4.*f51

    b1_nu = b1 * (f10 + f11*nu + f12*nu2)    # eq. 9 of arXiv:1611.00332v2
    b2_nu = b2 * (f20 + f21*nu + f22*nu2)
    b3_nu = b3 * (f30 + f31*nu + f32*nu2)
    b5_nu = b5 * (f50 + f51*nu + f52*nu2)

    Erad_nu25_S = 0.0484161*(0.128*b3_nu*S_3 + 0.211*b2_nu*S_2 + 0.346*b1_nu*S + 1)/(1 - 0.212*b5_nu*S)    # eq. 22 of arXiv:1611.00332v2
    Erad_nu25_0 = 0.0484161     # see discussion at the beginning od sec. IV.B

    Erad_nu_S = Erad_nu_0 * Erad_nu25_S/Erad_nu25_0    # eq. 23 of arXiv:1611.00332v2

    # 3D fits
    A1 = d10 * X12 * nu2 * (d11*nu+1)    # eqs. 27(a-c) of arXiv:1611.00332v2
    A2 = d20 * nu3
    A3 = d30 * X12 * nu * (d31*nu+1)
        
    DErad_nu_S_Dchi = A1*Dchi + A2*Dchi_2 + A3*S*Dchi    # eq. 15 of arXiv:1611.00332v2
    Erad = Erad_nu_S + DErad_nu_S_Dchi     # eq. 28 of arXiv:1611.00332v2
    res = M * (1 - Erad)      # see discussion at the beginning of sec. IV

    return res


def _JimenezFortezaRemnantSpin(nu, X1, X2, chi1, chi2):
    """
    Compute the BH final spin with the aligned-spin NR fit by Xisco Jimenez Forteza,
    David Keitel, Sascha Husa et al.: see arXiv:1611.00332v2
    -----------------------------------------------------------------------------------
    m1, m2: component masses. chi1, chi2: dimensionless spins
    Note that it is assumed m1>m2
    -----------------------------------------------------------------------------------
    Note: In arXiv:1611.00332v2 the symmetric mass ratio is called 'eta', contrary to
          the rest of the implementation where is 'nu'.
          For clarity, we continue to use the symbol 'nu' also for the
          functions JimenezFortezaRemnantMass and JimenezFortezaRemnantSpin
    Note: In arXiv:1611.00332v2, S_hat has a slighly different definition wrt the
        rest of the implementation
    """

    X12  = X1 - X2
    X1_2 = X1*X1; X2_2 = X2*X2
    S    = (X1_2*chi1 + X2_2*chi2)/(X1_2 + X2_2)
    Dchi = chi1-chi2

    res = 0.
    nu2 = nu*nu; nu3 = nu2*nu; S_2 = S*S; S_3 = S_2*S; Dchi_2 = Dchi*Dchi

    # 1D fits - Table I from arXiv:1611.00332v2
    a2 = 3.833; a3 = -9.49; a5 = 2.513

    # 1D fits - Table II from arXiv:1611.00332v2
    b1 = 1.00096; b2 = 0.788; b3 = 0.654; b5 = 0.840

    # 2D fits - Table III from arXiv:1611.00332v2
    f21 = 8.774; f31 = 22.83; f50 = 1.8805
    f11 = 0.345225*f21 + 0.0321306*f31 - 3.66556*f50 + 7.5397   # eq. 13 of arXiv:1611.00332v2

    # 3D fits - Table IV from arXiv:1611.00332v2
    d10 = 0.322; d11 = 9.33; d20 = -0.0598; d30 = 2.32; d31 = -3.26
    f12 = 0.512; f22 = -32.1; f32 = -154; f51 = -4.77


    # 1D fits
    Lorb_nu_0 = (1.3*a3*nu3 + 5.24*a2*nu2 + 2.*np.sqrt(3)*nu)/(2.88*a5*nu + 1)    # eq. 7 of arXiv:1611.00332v2

    # 2D fits
    f13 = 64 - 16.*f11 - 4.*f12    # eq. 11 of arXiv:1611.00332v2
    f23 = 64 - 16.*f21 - 4.*f22
    f33 = 64 - 16.*f31 - 4.*f32
    f53 = 64 - 64.*f50 - 16.*f51

    b1_nu = b1 * (f11*nu + f12*nu2 + f13*nu3)    # eq. 9 of arXiv:1611.00332v2
    b2_nu = b2 * (f21*nu + f22*nu2 + f23*nu3)
    b3_nu = b3 * (f31*nu + f32*nu2 + f33*nu3)
    b5_nu = b5 * (f50 + f51*nu + f53*nu3)

    Lorb_nu25_S = (0.00954*b3_nu*S_3 + 0.0851*b2_nu*S_2 - 0.194*b1_nu*S)/(1 - 0.579*b5_nu*S) + 0.68637    # eq. 8 of arXiv:1611.00332v2
    Lorb_nu25_0 = 0.68637   # follow from eq. 8 by imposing S=0

    Lorb_nu_S = Lorb_nu_0 + Lorb_nu25_S - Lorb_nu25_0   # eq. 10 of arXiv:1611.00332v2

    # 3D fits
    A1 = d10 * X12 * nu2 * (d11*nu+1)    # eqs. 19(a-c) of arXiv:1611.00332v2
    A2 = d20 * nu3
    A3 = d30 * X12 * nu3 * (d31*nu+1)

    DLorb_nu_S_Dchi = A1*Dchi + A2*Dchi_2 + A3*S*Dchi    # eq. 15 of arXiv:1611.00332v2
    Lorb = Lorb_nu_S + DLorb_nu_S_Dchi     # eq. 16 of arXiv:1611.00332v2
    res = Lorb + X1_2*chi1 + X2_2*chi2      # see discussion in sec. III.A

    return res



###################################################
# Utils Section 6: Fits for time and phase delays #
# functions: DeltaT, DeltaPhi                     #
###################################################

def _DeltaT(nu, X12, S_hat, a0, l, m):
    """
    Function returning the value of Delta_t, following appendix C.2 of arXiv:2001.09082v2.
    It is called in function EOBPM_SetupFitCoefficients of waveform.pyx.
    -----------------------------------------------------------------------------------
    The fitting coefficients and equations are implemented from:
    (2,2) (2,1) (3,3) (3,2) (4,4) (4,3) (4,2) - arXiv:2001.09082v2, appendix C.3
    (3,1) (4,1) (5,5) - arXiv:1904.09550v2, section V.D
    -----------------------------------------------------------------------------------
    Note that paper arXiv:1904.09550v2 refers to the non-spinning case, while arXiv:2001.09082v2 includes spin effects.
    """

    res = 0.
    nu2    = nu*nu; 
    nu3    = nu2*nu
    nu4    = nu3*nu 
    X12_2  = X12*X12
    X12_3  = X12_2*X12
    S_hat2 = S_hat*S_hat 
    S_hat3 = S_hat2*S_hat 
    a0_2   = a0*a0

    # ------------------------------------------------------------------------------------------------------------------------------------------------ #
    # (2,1) - TEOBResumSFits.c, line 673-680
    Dt21_0    = 11.75925;  Dt21_0_nu = 6.6264     # TEOBResumSFits.c; line 640
    a_1_Dt21  = -2.0728
    b_1_Dt21  = 0.0472289; b_2_Dt21  = 0.115583
    c_11_Dt21 = -1976.13;  c_12_Dt21 = 3719.88; c_21_Dt21 = -2545.41; c_22_Dt21 = 5277.62

    # (3;3) - TEOBResumSFits.c; line 736-738
    Dt33_0    = 3.42593     # TEOBResumSFits.c; line 643
    a_1_Dt33  = 0.183349; a_2_Dt33  = 4.22361
    b_1_Dt33  = -0.49791; b_2_Dt33  = -0.18754; b_3_Dt33  = -1.07291
    c_11_Dt33 = -1.9478;  c_12_Dt33 = 13.9828;  c_21_Dt33 = 1.25084; c_22_Dt33 = -3.41811; c_31_Dt33 = -1043.15; c_32_Dt33 = 1033.85

    # (3;2) - TEOBResumSFits.c; line 686-729
    Dt32_0      = 9.16665     # TEOBResumSFits.c; line 642
    a_1_Dt32    = -11.3497;  a_2_Dt32    = 32.9144;   a_3_Dt32    = -8.36579; a_4_Dt32    = 20.1017
    b_1_Dt32    = -0.34161;  b_2_Dt32    = -0.46107;  b_3_Dt32    = 0.34744
    b_1_Dt32_nu = 0.15477;   b_2_Dt32_nu = -0.755639; b_3_Dt32_nu = 0.21816
    c_11_Dt32   = -0.037634; c_12_Dt32   = 12.456704; c_13_Dt32   = 2.670868; c_14_Dt32   = -12.255859; c_15_Dt32   = 37.843505; c_21_Dt32   = -25.058475; c_22_Dt32   = 449.470722;  c_23_Dt32   = -1413.508735; c_24_Dt32   = -11.852596; c_25_Dt32   = 41.348059; c_31_Dt32   = -5.650710; c_32_Dt32   = -9.567484; c_33_Dt32   = 173.182999; c_34_Dt32   = -10.938605; c_35_Dt32   = 35.670656
    c_11_Dt32_X = 2.497188;  c_12_Dt32_X = -7.532596; c_13_Dt32_X = 4.645986; c_14_Dt32_X = -3.652524;  c_15_Dt32_X = 3.398687;  c_21_Dt32_X = 7.054185;   c_22_Dt32_X = -12.260185;  c_23_Dt32_X = 5.724802;     c_24_Dt32_X = -3.242611;  c_25_Dt32_X = 2.714232;  c_31_Dt32_X = 2.614565;  c_32_Dt32_X = -9.507583; c_33_Dt32_X = 7.321586;   c_34_Dt32_X = -3.937568;  c_35_Dt32_X = 4.584970

    # (3;1) - TEOBResumSFits.c; line 793
    Dt31_0   = 12.9338     # TEOBResumSFits.c; line 641
    n_2_Dt31 = -25.615
    d_1_Dt31 = 0.88803; d_2_Dt31 = 16.292

    # (4;4) - TEOBResumSFits.c; line 781-787
    Dt44_0       = 5.27778     # TEOBResumSFits.c; line 647
    a_1_Dt44     = -8.35574;   a_2_Dt44     = 17.5288; a_3_Dt44  = -6.50259;  a_4_Dt44  = 10.1575
    b_1_Dt44_nu  = 0.00159701; b_2_Dt44_nu  = -1.14134
    c_11_Dt44    = -2.28656;   c_12_Dt44    = 1.66532; c_21_Dt44 = -0.589331; c_22_Dt44 = 0.708784

    # (4;3) - TEOBResumSFits.c; line 759-777
    Dt43_0    = 9.53705     # TEOBResumSFits.c; line 646
    a_1_Dt43  = -11.2377;  a_2_Dt43  = 38.3177;   a_3_Dt43  = -7.29734;  a_4_Dt43  = 21.4267
    b_1_Dt43  = -1.371832; b_2_Dt43  = 0.362375;  b_3_Dt43  = -1.0808402
    c_11_Dt43 = 3.215984;  c_12_Dt43 = 42.133767; c_13_Dt43 = -9.440398; c_14_Dt43 = 35.160776; c_21_Dt43 = 1.133942; c_22_Dt43 = -10.356311; c_23_Dt43 = -6.701429; c_24_Dt43 = 10.726960; c_31_Dt43 = -6.036207; c_32_Dt43 = 67.730599; c_33_Dt43 = -3.082275; c_34_Dt43 = 11.547917

    # (4;2) - TEOBResumSFits.c; line 742-755
    Dt42_0    = 11.66665     # TEOBResumSFits.c; line 645
    a_1_Dt42  = -9.8446172795; a_2_Dt42  = 23.3229430582; a_3_Dt42  = -5.7604819848; a_4_Dt42  = 7.1217930024
    b_1_Dt42  = -1.3002045;    b_2_Dt42  = -0.9494348
    c_11_Dt42 = 24.604717;     c_12_Dt42 = -0.808279;     c_21_Dt42 = 62.471781;     c_22_Dt42 = 48.340961

    # (4;1) - TEOBResumSFits.c; line 795
    Dt41_0   = 13.1116     # TEOBResumSFits.c; line 644
    n_1_Dt41 = -9.6225; n_2_Dt41 = 38.451
    d_1_Dt41 = -7.7998; d_2_Dt41 = 32.405

    # (5;5) - TEOBResumSFits.c; line 800
    Dt55_0    = 6.561811     # TEOBResumSFits.c; line 648
    n_1_Dt55 = -91.4401; n_2_Dt55 = 2548.5975; n_3_Dt55 = -11086.4884; n_4_Dt55 = 27137.0063
    d_1_Dt55 = -67.156;  d_2_Dt55 = 1773.5942
    # ------------------------------------------------------------------------------------------------------------------------------------------------ #


    if ((l==2) and (m==2)):

        res = 0.0


    elif ((l==2) and (m==1)):   # eqs. C29-C32 of arXiv:2001.09082v2

        b1_Dt = (b_1_Dt21 + c_11_Dt21*X12)/(1 + c_12_Dt21*X12)
        b2_Dt = (b_2_Dt21 + c_21_Dt21*X12)/(1 + c_22_Dt21*X12)

        Dt21_orb  = (Dt21_0*(1-4*nu) + Dt21_0_nu*4*nu)*(1 + a_1_Dt21*nu*np.sqrt(1-4*nu))
        Dt21_spin = 1 + b1_Dt*a0 + b2_Dt*a0_2

        res = Dt21_orb * Dt21_spin


    elif ((l==3) and (m==3)):   # eqs. C42-C45 of arXiv:2001.09082v2

        b1_Dt = (b_1_Dt33 + c_11_Dt33*nu)/(1 + c_12_Dt33*nu)
        b2_Dt = (b_2_Dt33 + c_21_Dt33*nu)/(1 + c_22_Dt33*nu)
        b3_Dt = (b_3_Dt33 + c_31_Dt33*nu)/(1 + c_32_Dt33*nu)

        Dt33_orb = 1 + a_1_Dt33*nu + a_2_Dt33*nu2
        num_spin = 1 + b1_Dt*S_hat + b2_Dt*S_hat2
        den_spin = 1 + b3_Dt*S_hat

        Dt33_spin = num_spin/den_spin

        res = Dt33_0 * Dt33_orb * Dt33_spin


    elif ((l==3) and (m==2)):   # eqs. C54-C60 of arXiv:2001.09082v2

        if (nu>=0.2):

            b1_Dt_nu = (b_1_Dt32_nu + c_11_Dt32_X*X12 + c_12_Dt32_X*X12_2 + c_13_Dt32_X*X12_3)/(1 + c_14_Dt32_X*X12 + c_15_Dt32_X*X12_2)
            b2_Dt_nu = (b_2_Dt32_nu + c_21_Dt32_X*X12 + c_22_Dt32_X*X12_2 + c_23_Dt32_X*X12_3)/(1 + c_24_Dt32_X*X12 + c_25_Dt32_X*X12_2)
            b3_Dt_nu = (b_3_Dt32_nu + c_31_Dt32_X*X12 + c_32_Dt32_X*X12_2 + c_33_Dt32_X*X12_3)/(1 + c_34_Dt32_X*X12 + c_35_Dt32_X*X12_2)

            num_spin = 1 + b1_Dt_nu*S_hat + b2_Dt_nu*S_hat2
            den_spin = 1 + b3_Dt_nu*S_hat

        else:

            b1_Dt = (b_1_Dt32 + c_11_Dt32*nu + c_12_Dt32*nu2 + c_13_Dt32*nu3)/(1 + c_14_Dt32*nu + c_15_Dt32*nu2)
            b2_Dt = (b_2_Dt32 + c_21_Dt32*nu + c_22_Dt32*nu2 + c_23_Dt32*nu3)/(1 + c_24_Dt32*nu + c_25_Dt32*nu2)
            b3_Dt = (b_3_Dt32 + c_31_Dt32*nu + c_32_Dt32*nu2 + c_33_Dt32*nu3)/(1 + c_34_Dt32*nu + c_35_Dt32*nu2)

            num_spin = 1 + b1_Dt*S_hat + b2_Dt*S_hat2
            den_spin = 1 + b3_Dt*S_hat

        num_orb  = 1 + a_1_Dt32*nu + a_2_Dt32*nu2
        den_orb  = 1 + a_3_Dt32*nu + a_4_Dt32*nu2

        Dt32_orb  = num_orb/den_orb
        Dt32_spin = num_spin/den_spin

        res = Dt32_0 * Dt32_orb * Dt32_spin


    elif ((l==3) and (m==1)):   # eq. 5.23-24 of arXiv:1904.09550v2

        num_orb = 1 + n_2_Dt31*nu2
        den_orb = 1 + d_1_Dt31*nu + d_2_Dt31*nu2

        Dt31_orb = num_orb/den_orb

        res = Dt31_0 * Dt31_orb


    elif ((l==4) and (m==4)):   # eqs. C69-C72 of arXiv:2001.09082v2

        b1_Dt = b_1_Dt44_nu + c_11_Dt44*X12 + c_12_Dt44*X12_2
        b2_Dt = b_2_Dt44_nu + c_21_Dt44*X12 + c_22_Dt44*X12_2

        num_orb  = 1 + a_1_Dt44*nu + a_2_Dt44*nu2
        den_orb  = 1 + a_3_Dt44*nu + a_4_Dt44*nu2
        num_spin = 1 + b1_Dt*S_hat
        den_spin = 1 + b2_Dt*S_hat

        Dt44_orb  = num_orb/den_orb
        Dt44_spin = num_spin/den_spin

        res = Dt44_0 * Dt44_orb * Dt44_spin


    elif ((l==4) and (m==3)):   # eqs. C82-C85 of arXiv:2001.09082v2

        b1_Dt = (b_1_Dt43 + c_11_Dt43*nu + c_12_Dt43*nu2)/(1 + c_13_Dt43*nu + c_14_Dt43*nu2)
        b2_Dt = (b_2_Dt43 + c_21_Dt43*nu + c_22_Dt43*nu2)/(1 + c_23_Dt43*nu + c_24_Dt43*nu2)
        b3_Dt = (b_3_Dt43 + c_31_Dt43*nu + c_32_Dt43*nu2)/(1 + c_33_Dt43*nu + c_34_Dt43*nu2)

        num_orb  = 1 + a_1_Dt43*nu + a_2_Dt43*nu2
        den_orb  = 1 + a_3_Dt43*nu + a_4_Dt43*nu2
        num_spin = 1 + b1_Dt*S_hat + b2_Dt*S_hat2
        den_spin = 1 + b3_Dt*S_hat

        Dt43_orb  = num_orb/den_orb
        Dt43_spin = num_spin/den_spin

        res = Dt43_0 * Dt43_orb * Dt43_spin


    elif ((l==4) and (m==2)):   # eqs. C94-C97 of arXiv:2001.09082v2

        if (nu<6./25.):
            b1_Dt = b_1_Dt42
            b2_Dt = b_2_Dt42

        else:
            b1_Dt = (b_1_Dt42 + c_11_Dt42*nu)/(1 + c_12_Dt42*nu)
            b2_Dt = (b_2_Dt42 + c_21_Dt42*nu)/(1 + c_22_Dt42*nu)

        num_orb  = 1 + a_1_Dt42*nu + a_2_Dt42*nu2
        den_orb  = 1 + a_3_Dt42*nu + a_4_Dt42*nu2
        num_spin = 1 + b1_Dt*S_hat
        den_spin = 1 + b2_Dt*S_hat

        Dt42_orb  = num_orb/den_orb
        Dt42_spin = num_spin/den_spin

        res = Dt42_0 * Dt42_orb * Dt42_spin


    elif ((l==4) and (m==1)):   # eq. 5.23-24 of arXiv:1904.09550v2
        
        num_orb = 1 + n_1_Dt41*nu + n_2_Dt41*nu2
        den_orb = 1 + d_1_Dt41*nu + d_2_Dt41*nu2

        Dt41_orb = num_orb/den_orb

        res = Dt41_0 * Dt41_orb


    elif ((l==5) and (m==5)):   # TEOBResumSFits.c, line 800

        num_orb = 1 + n_1_Dt55*nu + n_2_Dt55*nu2 + n_3_Dt55*nu3 + n_4_Dt55*nu4
        den_orb = 1 + d_1_Dt55*nu + d_2_Dt55*nu2

        Dt55_orb = num_orb/den_orb

        res = Dt55_0 * Dt55_orb


    return res


def _DeltaPhi(nu, X12, S_hat, l, m):
    """
    Not implemented
    """

    res = 0.

    if(l==2 and m==2):

        res = 0.0

    elif(l==2 and m==1):

        res = 0.0

    elif(l==3 and m==3):

        res = 0.0

    elif(l==3 and m==2):

        res = 0.0

    elif(l==3 and m==1):

        res = 0.0

    elif(l==4 and m==4):

        res = 0.0

    elif(l==4 and m==3):

        res = 0.0

    elif(l==4 and m==2):

        res = 0.0

    elif(l==4 and m==1):

        res = 0.0

    elif(l==5 and m==5):

        res = 0.0

    return res
