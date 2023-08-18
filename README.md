# NovoB: A Bidirectional De Novo Peptide Sequencing Using Transformer Model

We recommand conda environment for runing NovoB.<p>
- NovoB requires :
  - python
  - tensorflow (tensorflow-gpu for using GPUs) & keras
  - pyteomics

- Our environment (on Linux CentOS 7.9) :
  - python = 3.10
  - tensorflow-gpu = 2.10 (for using GPUs)
  - keras = 2.10
  - pyteomics

- For setting our environment (on Linux CentOS 7.9) :
  - conda env create -f environment.yaml
  - conda activate NovoB

- If you use tensorflow < 2.10, we recommand to use --save_weights (Learning.py) and --load_weights (Prediction.py).

***

##### Initail Model
- NovoBInit
  - The model of NovoB which does not learn weights.
##### Learned Model
- LearnedModel/ricebean
  - The model of NovoB which learns weights.
  - We used git-lfs to upload files (>100M). So, use "git lfs pull" after "git clone"
##### leaned Weights
- Weights/yeast/variables
  - We used git-lfs to upload files (>100M). So, use "git lfs pull" after "git clone"
##### Sample Spectra 
- MGF/yeast.10k.mgf
- MGF/ricebean.10k.mgf

***

### Learn Model
For learning model, use Learning.py

```
python Learning.py -h

usage: Learning.py [-h] -m MODEL_PATH [--save_model SAVE_MODEL] [--save_weights SAVE_WEIGHTS] -l LEARNING_FILE.mgf -v VALIDATION_FILE.mgf [-b BATCH_SIZE] [-e EPOCHS] [-g]
                   [-n]
options:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        Saved Model Path (folder)
  --save_model SAVE_MODEL
                        Path to save Model (folder), default = model
  --save_weights SAVE_WEIGHTS
                        Path/file_name to save weights (path/file_name)
  -l LEARNING_FILE.mgf, --learning_file LEARNING_FILE.mgf
                        Learning File (.mgf)
  -v VALIDATION_FILE.mgf, --validation_file VALIDATION_FILE.mgf
                        Validation File (.mgf)
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for learning model (on single GPU), default=128
  -e EPOCHS, --epochs EPOCHS
                        the number of epochs, default=30
  -g, --no_multigpu     Do not use multigpu
  -n, --no_nccl         Do not use nccl (when using multigpu)
```

- Example)
  - NovoB Model: NovoBInit (do not learn weights)
  - Learning Spectra : MGF/yeast.10k.mgf
  - Validation Specra : MGF/yeast.10k.mgf
  - Output: Model ("model" folder)
```
python Learning.py -m NovoBInit -l MGF/yeast.10k.mgf -v MGF/yeast.10k.mgf
```

- If you want to save only weights, use --save_weigths option as follows.
  - Output: Weights (weights file in "model" folder)
```
python Learning.py -m NovoBInit --save_weights model/weights -l MGF/yeast.10k.mgf -v MGF/yeast.10k.mgf
```


***

### Load Model and Weights to predict peptides
For predict peptide, use Prediction.py

```
python Prediction.py -h

usage: Prediction.py [-h] -m MODEL_PATH [--load_weights LOAD_WEIGHTS] -i SPECTRUM_FILE [-o OUTPUT_FILE] [-b BATCH_SIZE] [-g] [-n]

options:
  -h, --help            show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                        Saved Model Path (folder)
  --load_weights LOAD_WEIGHTS
                        Load weights after loading model
  -i SPECTRUM_FILE, --spectrum_file SPECTRUM_FILE
                        spectrum file (.mgf)
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        output file (text file), default=result.txt
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size to predict peptides (on single GPU), default=256
  -g, --no_multigpu     Do not use multigpu
  -n, --no_nccl         Do not use nccl (when using multigpu)
```

- Example)
  - NovoB Model: NovoBInit (do not learn weights)
  - Learned weights : variables in "Weights/yeast" folder
  - Sample specra : MGF/yeast.10k.mgf
```
python Prediction.py -m NovoBInit/ --load_weights Weights/yeast/variables -i MGF/yeast.10k.mgf
```

- If you want to use learned model, don't use --load_weights option as follows.
  - NovoB Model : LearnedModel/ricebean
  - Recommand to use tensorflow >= 2.10
```
python Prediction.py -m LearnedModel/ricebean/ -i MGF/ricebean.10k.mgf
```
