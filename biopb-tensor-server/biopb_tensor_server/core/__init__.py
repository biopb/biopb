"""Foundational layer: shared primitives, adapter contracts, configuration,
discovery, and the low-level source/metadata stores the rest of the server
builds on.

Modules here depend only on each other (and the ``adapters`` / ``cache``
subpackages); ``serving`` and ``sources`` depend on this layer, not vice versa.
"""
