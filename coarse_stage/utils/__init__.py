def dense_crf(*args, **kwargs):
    """Load the optional CRF dependency only when postprocessing is requested."""
    from .crf import dense_crf as implementation
    return implementation(*args, **kwargs)

from .load import *
from .utils import *
from .data_vis import *
