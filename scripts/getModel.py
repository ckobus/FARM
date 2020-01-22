import logging

import os, sys
import torch

from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import BertStyleLMProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel

language_model = LanguageModel.load(sys.argv[1])
#language_model = LanguageModel.load("bert-base-cased")
dir='../models/'+sys.argv[1]
os.mkdir(dir)
language_model.save(dir)
