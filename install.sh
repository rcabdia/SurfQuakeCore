python3 -m pip install --upgrade pip
pip install build
python3 -m build
python3 -m venv surfquake_env
source surfquake_env/bin/activate
cd dist
pip install surfquakecore-0.0.1-py3-none-any.whl