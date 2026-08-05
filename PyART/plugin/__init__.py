"""
Plugins that expose PyART waveforms to external inference libraries.

Nothing is imported eagerly here: these modules exist to be used with an
external library, and the rest of PyART should not pay their import cost.
Import the one you want explicitly instead:

    from PyART.plugin.bilby_plugin import nr_frequency_domain_source_model
"""
