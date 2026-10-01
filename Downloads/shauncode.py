import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

# run after code.py, this is the remaining preprocesing stuff and then also correlation analysis.

np.random.seed(42)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

BASE = os.environ.get("EODP_BASE", r"C:\Users\shaun\EODP_ASS2")   #change as needed
IN_PATH = os.path.join(BASE, "outputs", "listings_processed.csv")   
OUT_DIR = os.path.join(BASE, "outputs") + os.sep
os.makedirs(OUT_DIR, exist_ok=True)

LOG = []
def log(msg=""):
    print(msg)
    LOG.append(str(msg))

df = pd.read_csv(IN_PATH)
log(f"loaded processed data: {df.shape}")