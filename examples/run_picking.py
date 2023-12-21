import os
from multiprocessing import freeze_support
from surfquakecore import model_dir
from surfquakecore.phasenet.phasenet_handler import PhasenetUtils
from surfquakecore.phasenet.phasenet_handler import PhasenetISP
from surfquakecore.utils.obspy_utils import MseedUtil

### LOAD PROJECT ###
path_to_project = "/Volumes/LaCie/test_surfquake_core/project"
project_name = 'surfquake_project_test2.pkl'
output_picks = '/Volumes/LaCie/test_surfquake_core/test_picking2'
project_file = os.path.join(path_to_project, project_name)

if __name__ == '__main__':
    freeze_support()

    project = MseedUtil.load_project(file=project_file)
    # conservative mode
    #phISP = PhasenetISP(project, modelpath=model_dir, amplitude=True, min_p_prob=0.90, min_s_prob=0.65)

    # Full mode
    phISP = PhasenetISP(project, amplitude=True, min_p_prob=0.30, min_s_prob=0.30)

    # Running Stage
    picks = phISP.phasenet()

    """ PHASENET OUTPUT TO REAL INPUT """

    picks_results = PhasenetUtils.split_picks(picks)
    PhasenetUtils.convert2real(picks_results, output_picks)
    PhasenetUtils.save_original_picks(picks_results, output_picks)




