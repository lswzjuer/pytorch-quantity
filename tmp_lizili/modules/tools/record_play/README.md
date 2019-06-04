# Record and Play Tool

### Prerequisite
Run the following command from your Roadstar root dir:
```bash
bash roadstar.sh build
source modules/tools/setup_pythonpath.sh
```

### Recorder
This tool records trajectory information from gateway into a csv file, the file name is defined in
filename_path.
```bash
python modules/tools/record_play/recorder_path.py
```

### Player
This tool reads information from a csv file and publishes planning trajectory in the same format as
real planning node.

Argument: Speed multiplier (default: 100)

```bash
python modules/tools/record_play/player_path.py [-s speed_multiplier]
```
