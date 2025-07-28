'''
Tests the ICC public catalog.
'''
from PyART.catalogs import icc_public
import os, numpy
def test_icc_public_catalog():
    '''
    Test the ICC public catalog waveform class.
    '''
    waveform = icc_public.Waveform_ICC(path='./', 
                                       ID='0004', 
                                       download=True, load=['hlm', 'metadata'],
                                       ellmax=4,
                                       extraction='extrap',
                                       nu_rescale=False)
    # --- File and structure checks ---
    assert waveform.ID == '0004'
    assert os.path.exists(waveform.sim_path)
    
    waveform.load_hlm()
    assert waveform.hlm is not None and len(waveform.hlm) > 0
    assert waveform.u is not None and len(waveform.u) > 0
    metadata_keys = ['m1', 'm2', 'Mf', 'q', 'S1', 'S2',]
    for key in metadata_keys:
        assert key in waveform.metadata, f"Metadata key {key} not found."
    
    # --- Mode content checks ---
    mode_keys = ['A', 'p', 'real', 'imag', 'z']
    hlm = waveform.hlm
    t = waveform.u
    for mode in waveform.hlm:
        ell, m = mode
        assert ell >= abs(m)

        mode_data = waveform.hlm[mode]

        # Check all expected keys are present
        for key in mode_keys:
            assert key in mode_data
        
        # Check array lengths are consistent
        n = len(waveform.u)
        for key in mode_keys:
            assert len(mode_data[key]) == n

        # Check A and z consistency
        A, z = mode_data['A'], mode_data['z']
        assert numpy.allclose(numpy.abs(z), A, rtol=1e-4, atol=1e-6)

    # Merger Sanity
    amplitude = waveform.hlm[(2,2)]['A']
    i_max = numpy.argmax(amplitude)
    assert i_max > 0 and i_max < len(amplitude)

    # Ringdown should decay
    assert numpy.mean(amplitude[i_max:]) < numpy.max(amplitude)

    # Phase continuity test
    phase = waveform.hlm[(2,2)]['p']
    delta_phase = numpy.unwrap(phase)
    delta = numpy.max(numpy.abs(numpy.diff(delta_phase)))
    assert delta < numpy.pi # no big jumps

test_icc_public_catalog()