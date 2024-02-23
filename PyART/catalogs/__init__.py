#!/usr/bin/env python
from __future__ import absolute_import
__import__("pkg_resources").declare_namespace(__name__)

from . import analysis
from . import analytic
from . import catalogs
from . import models
from . import utils
from .. import waveform
from .. import simulations

# ---------------------
# Basic unit conversion
# --------------------- 

ufact= {
    'Msun_sec': 4.925794970773135e-06,
    'Msun_meter': 1.476625061404649406193430731479084713e3,
}

class geom:
    """
    Geometric units + Msun = 1
    """
    grav_constant       = 1.0
    light_speed         = 1.0
    solar_mass          = 1.0
    MeV                 = 1.0

class cgs:
    """
    CGS units
    """
    grav_constant       = 6.673e-8
    light_speed         = 29979245800.0
    solar_mass          = 1.988409902147041637325262574352366540e33
    MeV                 = 1.1604505e10

class metric:
    """
    Standard SI units
    """
    grav_constant       = 6.673e-11
    light_speed         = 299792458.0
    solar_mass          = 1.988409902147041637325262574352366540e30
    MeV                 = 1.1604505e10
