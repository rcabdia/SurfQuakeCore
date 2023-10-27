import os
from obspy import UTCDateTime, Stream, Inventory
from surfquakecore.moment_tensor.sq_isola_tools import BayesISOLA


class bayesian_isola_db:
    def __init__(self, metadata: Inventory, project: dict, parameters: dict, working_directory: str,
                 ouput_directory: str):

        """
        ----------
        Parameters
        ----------
        metadata ObsPy Inventory obsj: information of stations coordinates and instrument description
        project dict: information of seismogram data files available
        parameters: dictionary generated from the dataclass
        """

        self.metadata = metadata
        self.project = project
        self.parameters = parameters
        self.cpuCount = os.cpu_count() - 1
        self.working_directory_local = None
