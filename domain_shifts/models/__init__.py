from .amazon import Model as amazon
from .camelyon import Model as camelyon
from .civil import Model as civil
from .fmow import Model as fmow
from .rxrx import Model as rxrx

from .amazon import NUM_CLASSES as amazon_n_class
from .camelyon import NUM_CLASSES as camelyon_n_class
from .civil import NUM_CLASSES as civil_n_class
from .fmow import NUM_CLASSES as fmow_n_class
from .rxrx import NUM_CLASSES as rxrx_n_class

__all__ = [camelyon, amazon, civil, fmow, rxrx]