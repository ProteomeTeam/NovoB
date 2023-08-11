# NovoB
A Bidirectional De Novo Peptide Sequencing Using Transformer Model

We recommand conda environment for runing NovoB.
NovoB requires
  - python
  - tensorflow (tensorflow-gpu for using GPUs) & keras
  - pyteomics

Our environment (on Linux CentOS 7.9)
  - python = 3.10
  - tensorflow-gpu = 2.10 (for using GPUs)
  - keras = 2.10
  - pyteomics

For setting environment (on Linux CentOS 7.9) :
  - conda env create -f environment.yaml
  - conda activate NovoB


Learn Model
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

Example)
- NovoB Model: NovoBInit (do not learn weights)
- Learning Spectra : MGF/yeast.10k.mgf
- Validation Specra : MGF/yeast.10k.mgf
```
python Learning.py -m NovoBInit -l MGF/yeast.10k.mgf -v MGF/yeast.10k.mgf
```

Load Model and Weights to predict peptides
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

Example)
- NovoB Model: NovoBInit (do not learn weights)
- Learned weights : Weights/yeast/variables
- Sample specra : MGF/yeast.10k.mgf
```
python Prediction.py -m NovoBInit/ --load_weights Weights/yeast/variables -i MGF/yeast.10k.mgf
```
