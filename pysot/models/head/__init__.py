from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.head.IRCA import MultiIRCA


IRCAS = {
        'MultiIRCA': MultiIRCA,
       }


def get_irca_head(name, **kwargs):
    return IRCAS[name](**kwargs)

