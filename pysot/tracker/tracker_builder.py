from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamirca_tracker import SiamIRCATracker

TRACKS = {
          'SiamIRCATracker': SiamIRCATracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
