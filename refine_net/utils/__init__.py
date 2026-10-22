def dense_crf(*args, **kwargs):
    """Import the optional CRF dependency only when used."""
    from .crf import dense_crf as implementation
    return implementation(*args, **kwargs)

from .data_load import *
from .data_crop import *
from .utils import *
from .data_vis import *
from .mask2label import *
