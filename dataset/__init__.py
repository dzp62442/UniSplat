"""Keep data-only OmniScene tools independent of Waymo/CUDA dependencies."""


def __getattr__(name):
    if name == 'WaymoDataset':
        from .waymo import WaymoDataset
        return WaymoDataset
    if name == 'OmniSceneDataset':
        from .omniscene import OmniSceneDataset
        return OmniSceneDataset
    raise AttributeError(name)
