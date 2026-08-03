"""
Plugins that expose PyART waveforms to external inference libraries.

Nothing is imported eagerly here: ``bilby_plugin`` pulls in lal, lalsimulation
and bilby, which are not required to use the rest of PyART. Import it
explicitly instead:

    from PyART.plugin.bilby_plugin import nr_frequency_domain_source_model
"""
