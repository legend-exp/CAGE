BASEDIR=/home/pi/controls

cd $BASEDIR
if [ -d latest ]; then
  rm -r latest
fi
python3 -m venv --system-site-packages latest
source latest/bin/activate

rm -rf cage
git clone https://github.com/wcpettus/cage.git
cd cage
git submodule update --init --recursive

python3 -m pip install -e dragonfly/dripline-python
python3 -m pip install -e dragonfly/dragonfly[colorlog,gpio,ads1x15]
python3 -m pip install adafruit-circuitpython-max31865
python3 -m pip install serial
