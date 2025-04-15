DATA
=========
Train and test dataset is a table in CSV file. </br>
| audio | start | end |  label  |
|-----|-----|-----|---------|
| /path/to/audio1.wav | 0 | 3 | guitar |
| /path/to/audio1.wav | 3 | 6 | voice |
| /path/to/audio2.wav | 0 | 3 | dog |
* **audio** - abs path 
* **start** **end** - start and end of *speech* segment in seconds (may be float or int).
* **label** - target class

#### RULE 1
Audio must be converted to mono `audio.wav` with sample rate `SR` kGhz and codec `pcm_alaw` or `pcm_s16le`

#### RULE 2
duration of segment must be equal in whole dataset for successful batch generation

#### Recomendation
Each CSV file should have a **version** reference, like `train.v1.csv`.

## Data Processing
There are 3 types of data processing
### **Manifest transformation**
Take one version of manifest as input, make transformation and save with another version. Only manifest is changed here, audios stay unchangeable.
1. Drop duplicated rows
```bash
python3 -m noisecls.data.make_dataset -t drop-duplicates -i /path/to/input/manifest.csv
```
* **-t, --task** - transformation task. See `noisecls.data.make_dataset:TASKS`
* **-i, --input** - /path/to/input/manifest.csv
* **-o, --output** - /path/to/output/manifest.csv. If not defined, set next version `v1.csv -> v2.csv`, `v2.3.csv -> v2.4.csv`
* **--clearml** - whether log output manifest to ClearML manager
* **-d,--description** - description of output version of manifest. If not defined, it will be task name
* **-v,--version** - version of output manifest to ClearML manager. If not defined, it will try to parse from manifest name
* **--tags** - tags for ClearML manager. Note that there are default tags

2. Drop given classes
```bash
python3 -m noisecls.data.make_dataset -t drop-classes -i /path/to/input/manifest.csv --drop_classes "classes.txt"
```

3. Remain only given classes
```bash
python3 -m noisecls.data.make_dataset -t drop-classes -i /path/to/input/manifest.csv --remain_classes "classes.txt"
```

4. Join manifests
```bash
python3 -m noisecls.data.make_dataset -t join-manifests -i "/path/to/input/manifest1.csv,/path/to/input/manifest2.csv" -o joined.csv
```

#### Add new transformation
You need to create class from `noisecls.data.transforms:DataTransform`.


### **Other processors**
1. Create manifest from audios dir with given class label
```bash
python3 -m noisecls.data.make_dataset -t form-manifest -i "/path/to/audios/dir/*.wav" -o /path/to/input/manifest.csv --label "class_name"
```
* **-i,--input** - `/path/to/audios/dir` or `/path/to/audios/dir/regex*.wav` with quotes
* **-o,--output** - /path/to/input/manifest.csv
* **--label** - name of class in manifest

2. Convert audios in input dir and save new ones to another dir. With defined sample rate and codec
```bash
python3 -m noisecls.data.make_dataset -t convert -i "/path/to/audios/dir/*.wav" -o /path/to/input/manifest.csv --label "class_name"
```
* **-i,--input** - `/path/to/audios/dir` or `/path/to/audios/dir/regex*.wav` with quotes
* **-o,--output** - /path/to/save/dir/
* **--label** - name of class in manifest

3. Save list of classes to another file
```bash
python3 -m noisecls.data.make_dataset -t get-classes -i /path/to/input/manifest.csv -o /path/to/save/classes.txt
```
* **-i,--input** - /path/to/manifest.csv
* **-o,--output** - /where/to/save/classes.txt

4. Split train and test
```bash
python3 -m noisecls.data.make_dataset -t split -i /path/to/input/manifest.csv -o /path/to/save/dir/ --test_ratio 0.2
```

5. Just add dataset to clearml (no task)
```bash
python3 -m noisecls.data.make_dataset -i /path/to/input/manifest.csv --clearml
```

#### Add new processor
You need to create class from `noisecls.data.transforms:DataProcess`.