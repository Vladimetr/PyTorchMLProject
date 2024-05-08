DATA
=========
Train and test dataset is a table in CSV file. </br>
| audio | start | end |  class  |
|-----|-----|-----|---------|
| /path/to/audio1.wav | 0 | 3 | real |
| /path/to/audio1.wav | 3 | 6 | real |
| /path/to/audio2.wav | 0 | 3 | fake |
* **audio** - abs path 
* **start** **end** - start and end of *speech* segment in seconds (may be float or int).
* **class** - target class `{real, fake}`

> See another example in `data/processed/train.v1.csv`

#### RULE 1
Audio must be converted to mono `audio.wav` with sample rate 8 kGhz and codec `pcm_alaw` or `pcm_s16le`

#### RULE 2
duration of speech segment must be equal in whole dataset for successful batch generation

#### Recomendation
Each CSV file should have a **version** reference, like train.v1.csv.
