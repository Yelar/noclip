"""The world model: autoencoder wrapper, model, trainer, sampler, play server."""
try:    # a deployment's own defaults (where downloads go and the like): a file of yours beside this one, optional
    from wm import site_local  # noqa: F401
except ImportError:
    pass
