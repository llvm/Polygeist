from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']
__version__ = '2.6.0a0+df5bbc09d1.nv24.12'
debug = False
cuda: Optional[str] = '12.6'
git_version = 'Unknown'
hip: Optional[str] = None
