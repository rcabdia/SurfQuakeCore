import re
import setuptools

with open('requirements.txt') as f:
    # clean from comments empty lines and -r local_requirements.txt
    INSTALL_REQUIRES = list(filter(
        lambda x: x != "" and not re.match("-r local_requirements.txt|#", x), f.read().splitlines()))

setuptools.setup(
    name="surfquakecore",
    version="0.0.1",
    author="Roberto Cabieces Díaz",
    description="Package for earquake monitoring",
    long_description="Package for earquake monitoring",
    long_description_content_type="text/m")