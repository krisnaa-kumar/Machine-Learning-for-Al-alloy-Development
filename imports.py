from __future__ import annotations
import argparse, json, re, textwrap, uuid, io
from pathlib import Path
from typing import List, Dict, Any
import base64
import fitz                                # PyMuPDF
import torch
import yaml
import requests
import numpy as np
import pandas as pd
import ast
from PIL import Image
from tqdm.auto import tqdm
from pydantic import BaseModel
import clip   
