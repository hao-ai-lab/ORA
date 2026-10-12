"""Isolated resident model-service implementations used by perception tools.

Import concrete workers from their owning module so ``python -m`` service
startup does not preload the executable module through this package.
"""
