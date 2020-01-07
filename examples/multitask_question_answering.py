import json
import re

from farm.infer import Inferencer
from farm.train import Trainer
from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import SquadbisProcessor
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.prediction_head import QuestionAnsweringHead
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.language_model import LanguageModel
from farm.modeling.tokenization import Tokenizer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings

set_all_seeds(seed=42)
device, n_gpu = initialize_device_settings(use_cuda=True)

batch_size = 64
n_epochs = 2

base_LM_model = "bert-base-cased"
#base_LM_model = "bert-large-uncased-whole-word-masking"
train_filename = "train-v2.0.json"
dev_filename = "dev-v2.0.json"
grad_acc_steps=1
evaluate_every = 500
max_seq_len = 384
#max_seq_len = 256
learning_rate = 3e-5
warmup_proportion = 0.1
save_dir = "./MultiTask_QA_Classification_" + str(base_LM_model) + "_max_seq_len_" + str(max_seq_len) + "_grad_acc_steps_" + str(grad_acc_steps)
print(save_dir)

inference = True
train = True

#variables for inference
predictions_file = save_dir + "/predictions.json"
full_predictions_file = save_dir + "/full_predictions.json"
inference_file = dev_filename

if train:
  if re.match(r'.*base.*', base_LM_model):
    hidden_size=768
  else:
    hidden_size=1024

  if re.match(r'.*cased.*', base_LM_model):
    do_lower_case=False
  else:
    do_lower_case=True

  # 1.Create a tokenizer
  tokenizer = Tokenizer.load(
                pretrained_model_name_or_path=base_LM_model, 
                do_lower_case=do_lower_case
              )

  # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
  classification_label_list = ["NONE", "SPAN"]
  ## to include later "YES" and "NO"
  qa_label_list = ["start_token", "end_token"]
  metric="squad"
  metrics=["squad", "f1_macro"]
  processor = SquadbisProcessor(
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                label_list=[classification_label_list, qa_label_list],
                metric=metrics,
                train_filename=train_filename,
                dev_filename=dev_filename,
                test_filename=None,
                data_dir="../data/squad20",
                #metrics=metrics
              )

  dicts = processor.file_to_dicts(file="../data/squad20/"+dev_filename)
  
  #samples = processor._dict_to_samples(dictionary=processor.apply_tokenization(dicts[0]))
  #print(samples[0])
  #exit

  # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a few descriptive statistics of our datasets
  data_silo = DataSilo(processor=processor, batch_size=batch_size, distributed=False)

  # 4. Create an AdaptiveModel
  # a) which consists of a pretrained language model as a basis
  language_model = LanguageModel.load(base_LM_model)
  # b) and a prediction head on top that is suited for our task => Question Answering
  qa_prediction_head = QuestionAnsweringHead(layer_dims=[hidden_size, len(qa_label_list)])
  qa_classification_head = TextClassificationHead(layer_dims=[hidden_size, len(classification_label_list)])

  model=AdaptiveModel(
    language_model=language_model,
    prediction_heads=[qa_prediction_head, qa_classification_head],
    embeds_dropout_prob=0.1,
    lm_output_types=["per_token","per_sequence"],
    device=device
  )

  # 5. Create an optimizer
  model, optimizer, lr_schedule = initialize_optimizer(
      model=model,
      learning_rate=learning_rate,
      schedule_opts={"name": "LinearWarmup", "warmup_proportion": 0.2},
      n_batches=len(data_silo.loaders["train"]),
      n_epochs=n_epochs,
      device=device,
      grad_acc_steps=grad_acc_steps,
  )

  # 6. Feed everything to the Trainer, which keeps care of growing our model and evaluates it from time to time
  trainer = Trainer(
      optimizer=optimizer,
      data_silo=data_silo,
      epochs=n_epochs,
      n_gpu=n_gpu,
      #warmup_linear=warmup_linear,
      lr_schedule=lr_schedule,
      evaluate_every=evaluate_every,
      device=device,
      grad_acc_steps=grad_acc_steps,
  )

  # 7. Let it grow! Watch the tracked metrics live on the public mlflow server: https://public-mlflow.deepset.ai
  model = trainer.train(model)

  # 8. Hooray! You have a model. Store it:
  model.save(save_dir)
  processor.save(save_dir)

if inference:
  model = Inferencer.load(save_dir, batch_size=32, gpu=True)
  full_result = model.inference_from_file(
    file=inference_file,
    max_processes=max_processes_for_inference,
  )

  for x in full_result:
    print(x)
    print()

  result = {r["id"]: r["preds"][0][0] for r in full_result}
  full_result =  {r["id"]: r["preds"] for r in full_result}

  json.dump(result,
            open(predictions_file, "w"),
            indent=4,
            ensure_ascii=False)
  json.dump(full_result,
            open(full_predictions_file, "w"),
            indent=4,
            ensure_ascii=False)
