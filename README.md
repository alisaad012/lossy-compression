# Installation

Git clone this repository, and `cd` into directory for remaining commands
```
git clone https://github.com/alisaad012/lossy-compression.git && cd lossy-compression
```

Then, follow instructions for either native or Docker installation.

## Native Installation

All steps can optionally be done in a virtual environment using tools such as `virtualenv` or `conda`.

Install tensorflow 1.12 (with GPU support, if you have a GPU and want everything to run faster)
```
pip3 install tensorflow==1.12.0
```
or
```
pip3 install tensorflow-gpu==1.12.0
```

Install other python packages:
```
pip3 install -r requirements.txt
```

Download the model data
```
python3 download_model.py 124M
```
# Run

To run the program
```
python3 src/main.py samples/<text file>
```

# Flow Chart
GPT-2 Encoding
![GPT-2 Encoding](samples/GPT-2_Encoding.png)
Arithmetic Coding
![Arithmetic Coding](samples/Arithmetic_Coding.png)
