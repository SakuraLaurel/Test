import multiprocessing
import argparse
import os
import pandas as pd
import shutil
import json
import gc
import torch
from pathlib import Path
from transp import Category
from transp import Config, Transp


def run():
    cat = Category()


if __name__ == "__main__":
    t = Transp(Config)
    t.submit()