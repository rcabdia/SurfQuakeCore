import re
import setuptools

MIN_PYTHON_VERSION = (3, 8)

# add extra requirements here
EXTRAS_REQUIRES = {
    'tests': [],
}
EXTRAS_REQUIRES['all'] = [dep for depl in EXTRAS_REQUIRES.values() for dep in depl]


def filter_requirements(str_line: str):
    return str_line != "" and not re.match("-r local_requirements.txt|#", str_line)


with open('requirements.txt') as f:
    # clean from comments empty lines and -r local_requirements.txt
    INSTALL_REQUIRES = list(filter(filter_requirements, f.read().splitlines()))

setuptools.setup(
    name="surfquakecore",
    version="0.0.1",
    url="https://github.com/rcabdia/SurfQuakeCore",
    author="The SurfQuakeCore Development Team",
    description="Package for earthquake monitoring",
    long_description="Package for earthquake monitoring",
    license='GNU Lesser General Public License, Version 3 (LGPLv3)',
    long_description_content_type="text/m",
    install_requires=INSTALL_REQUIRES,
    tests_require=EXTRAS_REQUIRES['tests'],
    extras_require=EXTRAS_REQUIRES,
    platforms='OS Independent',
    python_requires=f'>={MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}',
)
