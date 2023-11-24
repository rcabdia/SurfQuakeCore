from sourcespec.ssp_parse_arguments import parse_args
from surfquakecore.magnitudes.structures import SoursceSpecOptions

options = parse_args(progname='source_spec')
print(options.config_file)
qml_file_path = ('/Users/roberto/Documents/SurfQuakeCore/examples/source_estimations/'
                     'locations/location.20220201.020301.grid0.loc.hyp')
trace_path = ['/Users/roberto/Documents/SurfQuakeCore/examples/source_estimations/data']
options = SoursceSpecOptions(config_file='source_spec.conf', evid=None, evname=None, hypo_file=None,
        outdir='sspec_out', pick_file=None, qml_file=qml_file_path, run_id="", sampleconf=False, station=None,
                              station_metadata=None ,trace_path=trace_path, updateconf=None)

# Setup stage
from sourcespec.ssp_setup import (
    configure, move_outdir, remove_old_outdir, setup_logging,
    save_config, ssp_exit)

config = configure(options, progname='source_spec')
setup_logging(config)