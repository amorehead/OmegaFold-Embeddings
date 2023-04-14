![header](figure.png)

# OmegaFold: High-resolution de novo Structure Prediction from Primary Sequence

#### This is the release code for paper [High-resolution de novo structure prediction from primary sequence](https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1).

We will continue to optimize this repository for more ease of use, for
instance, reducing the GRAM required to inference long proteins and
releasing possibly stronger models.

## Update Notes

### Model 2 release notes Dec 9. 2022

Now you can use model 2 by setting `--model 2` in the command line!

### Huge GRAM reduction

We have optimized (to some extent) the GRAM usage of OmegaFold model in our
latest release. Now the model can inference protein sequence as long as
_4096_ on NVIDIA A100 Graphics card with 80 GB of memory with
`--subbatch_size` set to 448 without hitting full memory.
This version's model is more sensitive to `--subbatch_size`.

### Setting Subbatch

Subbatch makes a trade-off between time and space.
One can greatly reduce the space requirements by setting `--subbatch_size`
very low.
The default is the number of residues in the sequence and the lowest
possible number is 1.
For now we do not have a rule of thumb for setting the `--subbatch_size`,
but we suggest half the value if you run into GPU memory limitations.

### MacOS Users

For macOS users, we support MPS (Apple Silicon) acceleration if the user
installs the latest nightly version of PyTorch.
Also, current code also requires macOS users need to `git clone` the
repository and use `python main.
py` (see below) to run the model.

## Setup

To prepare the environment to run OmegaFold,

- from source

```commandline
pip install git+https://github.com/amorehead/OmegaFold-Embeddings.git
```

- clone the repository

```commandline
git clone https://github.com/amorehead/OmegaFold-Embeddings.git
cd OmegaFold
python setup.py install
```

should get you where you want.

The `INPUT_FILE.fasta` should be a normal fasta file with possibly many
sequences with a comment line starting with `>` or `:` above the amino
acid sequence itself.

This command will download the weight
from https://helixon.s3.amazonaws.com/release1.pt
to `~/.cache/omegafold_ckpt/model.pt`
and load the model

## Running (for structure prediction)

You could simply

```commandline
omegafold INPUT_FILE.fasta OUTPUT_DIRECTORY
```

And voila!

## Running (for embedding generation)

You could simply

```python
import torch
import omegafold as of
from omegafold import pipeline

# Specify input sequence as a string
input_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

# Process model arguments
args, state_dict, forward_config = pipeline.get_args(generate_pdb_file_outputs=False)
assert state_dict is not None, "Must perform model inference using pretrained weights."

# Specify computation device
args.device = "cuda"

# Load OmegaFold model
model = of.OmegaFold(of.make_config(args.model))
if "model" in state_dict:
	state_dict = state_dict.pop("model")
model.load_state_dict(state_dict)
model.eval()  # disable dropout for deterministic results
model.to(args.device)

# Generate model inputs from input sequence
input_data = pipeline.sequence2input(
	input_sequence,
	num_pseudo_msa=args.num_pseudo_msa,
	device=args.device,
	mask_rate=args.pseudo_msa_mask_rate,
	num_cycle=args.num_cycle,
	deterministic=True
)

# Run model inference
with torch.no_grad():
	model_outputs = model(
		input_data,
		predict_with_confidence=True,
		fwd_cfg=forward_config
	)

# Extract desired e.g., sequence embeddings
sequence_representations = model_outputs["final_plm_node_representations"]
sequence_representations = sequence_representations.view(len(input_sequence), -1)  # unravel `heads` dimension
```

### Alternatively (For MacOS users - for structure prediction)

Even if this failed, since we use minimal 3rd party libraries, you can
always just install the latest
[PyTorch](https://pytorch.org) and [biopython](https://biopython.org)
(and that's it!) yourself.
For mps accelerator, macOS users may need to install the lastest nightly
version of PyTorch.
In this case, for structure prediction you could run

```commandline
python main.py INPUT_FILE.fasta OUTPUT_DIRECTORY
```

### Notes on resources

However, since we have implemented sharded execution, it is possible to

1. trade computation time for GRAM: by changing `--subbatch_size`. The
   smaller
   this value is, the longer the execution can take, and the less memory is
   required, or,
2. trade computation time for average prediction quality, by changing
   `--num_cycle`

For more information, run

```commandline
omegafold --help
```

where we provide several options for both speed and weights utilities.

## Output

We produce one pdb for each of the sequences in `INPUT_FILE.fasta` saved in
the `OUTPUT_DIRECTORY`. We also put our confidence value the place of
b_factors in pdb files.

## Cite

If this is helpful to you, please consider citing the paper with

```tex
@article{OmegaFold,
	author = {Wu, Ruidong and Ding, Fan and Wang, Rui and Shen, Rui and Zhang, Xiwen and Luo, Shitong and Su, Chenpeng and Wu, Zuofan and Xie, Qi and Berger, Bonnie and Ma, Jianzhu and Peng, Jian},
	title = {High-resolution de novo structure prediction from primary sequence},
	elocation-id = {2022.07.21.500999},
	year = {2022},
	doi = {10.1101/2022.07.21.500999},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/07/22/2022.07.21.500999},
	eprint = {https://www.biorxiv.org/content/early/2022/07/22/2022.07.21.500999.full.pdf},
	journal = {bioRxiv}
}

```

## Note

Also some of the comments might be out-of-date as of now, and will be
updated very soon