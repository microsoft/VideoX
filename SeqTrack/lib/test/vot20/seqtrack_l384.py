import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.test.vot20.seqtrack_vot20 import run_vot_exp

run_vot_exp('seqtrack', 'seqtrack_l384', vis=False)
