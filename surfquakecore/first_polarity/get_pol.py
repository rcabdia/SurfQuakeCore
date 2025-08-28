#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
get_pol
"""

import os

from surfquakecore import POLARITY_NETWORK
from surfquakecore.first_polarity.PolarCAP.cnnFirstPolarity import Polarity
from surfquakecore.project.surf_project import SurfProject


class RunPolarity:

    def __init__(self, project, pick_file:str, output_path:str, threshold=0.9):

        self.project = project
        if isinstance(self.project, SurfProject):
            pass
        elif isinstance(self.project, str):
            self.project = SurfProject.load_project(self.project)

        self.pick_file = pick_file
        self.output_path = output_path
        self.threshold = threshold

    def send_polarities(self):

        pol = Polarity(project=self.project, model_path=POLARITY_NETWORK, arrivals_path=self.pick_file,
                       threshold=self.threshold,
                       output_path=self.output_path)

        pol.optimized_project_processing_pol()