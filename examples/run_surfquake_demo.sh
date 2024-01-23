source ../venv/bin/activate
surfquake project -d /Volumes/LaCie/surfquake_test/testing_data -s /Volumes/LaCie/surfquake_test/project -n surfquake_project.pkl -v
surfquake pick -f /Volumes/LaCie/surfquake_test/project/surfquake_project.pkl -d /Volumes/LaCie/surfquake_test/test_picking_final -p 0.3 -s 0.3 -v
surfquake associate -i /Volumes/LaCie/surfquake_test/metadata/inv_all.xml -p /Volumes/LaCie/surfquake_test/test_picking_final -c /Volumes/LaCie/surfquake_test/config_files/real_config.ini -w /Volumes/LaCie/surfquake_test/test_real_final/working_directory -s /Volumes/LaCie/surfquake_test/test_real_final -v
surfquake locate -i /Volumes/LaCie/surfquake_test/metadata/inv_all.xml -c /Volumes/LaCie/surfquake_test/config_files/nll_config.ini -o /Volumes/LaCie/surfquake_test/test_nll_final -g -s
surfquake source -i /Volumes/LaCie/surfquake_test/metadata/inv_all.xml -p /Volumes/LaCie/surfquake_test/project/surfquake_project.pkl -c /Volumes/LaCie/surfquake_test/config_files/source_spec.conf -l /Volumes/LaCie/surfquake_test/test_nll_final/loc -o /Volumes/LaCie/surfquake_test/test_source_final
surfquake mti -i /Users/robertocabiecesdiaz/Documents/SurfQuakeCore/tests/test_resources/mti/mti_run_inversion_resources/inv_surfquakecore.xml -p /Volumes/LaCie/surfquake_test/project/surfquake_project_mti.pkl -o /Volumes/LaCie/surfquake_test/test_mti_final -c /Volumes/LaCie/surfquake_test/mti_configs -s