# Data

The competition data is **not** stored in this repository (Kaggle competition rules + file size).

Download it from the Kaggle competition page, **Heavy Equipment Selling Price Prediction Challenge**, and place the files here:

```
data/raw/
├── train.csv
└── test.csv
```

With the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) (after accepting the competition rules on the website):

```bash
kaggle competitions download -c heavy-equipment-selling-price-prediction-challenge -p data/raw
unzip data/raw/*.zip -d data/raw
```

Running on Kaggle itself? Point the scripts at the input folder instead:

```bash
python -m src.train --data-dir /kaggle/input/competitions/heavy-equipment-selling-price-prediction-challenge
```
