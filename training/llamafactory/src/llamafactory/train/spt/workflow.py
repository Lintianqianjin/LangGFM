# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, List, Optional

from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor, ComputeExactMatch, ComputeRegressionMetrics, ComputeAucMetrics, BinaryClassificationProbabilityCalculator
from .trainer import CustomSeq2SeqTrainer
from .utils import get_evaluation_settings

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments
import csv
import os


logger = get_logger(__name__)


def run_spt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction
    
    ####################DEBUG Start#########################
    # # debug for learning sample logic
    sample = dataset_module['eval_dataset']['experiments__raw_ofiicial_split__bace__test'][0]
    print(f"{sample=}")
    print(f"{len(sample['input_ids'])=}")
    print(f"{len(sample['attention_mask'])=}")
    # # 提取样本的input_ids和attention_mask
    # input_ids = sample['input_ids']
    # attention_mask = sample['attention_mask']
    
    # # 创建一个CSV文件来保存token信息
    
    # # 确保token_ids是一个张量并且只取第一个样本（如果是批量的）
    # if hasattr(input_ids, 'shape') and len(input_ids.shape) > 1:
    #     token_ids = input_ids[0].tolist()
    #     masks = attention_mask[0].tolist() if hasattr(attention_mask, 'shape') else attention_mask
    # else:
    #     token_ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else input_ids
    #     masks = attention_mask.tolist() if hasattr(attention_mask, 'tolist') else attention_mask
    
    # # 创建CSV文件
    # csv_file_path = 'token_analysis.csv'
    # os.makedirs(training_args.output_dir, exist_ok=True)
    
    # with open(csv_file_path, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['Index', 'Token ID', 'Token Text', 'Attention Mask'])
        
    #     for idx, (token_id, mask) in enumerate(zip(token_ids, masks)):
    #         token_text = tokenizer.decode([token_id])
    #         writer.writerow([idx, token_id, token_text, mask])
    
    # print(f"Token analysis saved to: {csv_file_path}")
    # # 将token IDs转换回文本，验证tokenization的准确性
    # # decoded_text = tokenizer.decode(token_ids[0])
    # # print("Decoded text:", decoded_text)
    
    # exit()
    ##################DEBUG END#####################
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    metric_module["evaluation_settings_dict"] = {
        dataset_name: get_evaluation_settings(dataset_name=dataset_name,tokenizer=tokenizer) for dataset_name in dataset_module["eval_dataset"].keys()
    }
    # if training_args.predict_with_generate:
    #     # metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    #     metric_module["compute_metrics"] = ComputeExactMatch(tokenizer=tokenizer)
    #     # metric_module["compute_metrics"] = ComputeRegressionMetrics(tokenizer=tokenizer)
    # elif finetuning_args.compute_accuracy:
    #     metric_module["compute_metrics"] = ComputeAccuracy()
    #     metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
    
    # metric_module["evaluation_settings_dict"] = {
    #     "experiments__debug__shortest_path_3200__test": {
    #         "compute_metrics": ComputeExactMatch(tokenizer=tokenizer),
    #         "gen_kwargs": {
    #             "do_sample": False,
    #             "max_new_tokens": 64,
    #         },
    #     },
    #     "experiments__debug__movielens1m__test": {
    #         "compute_metrics": ComputeRegressionMetrics(tokenizer=tokenizer),
    #         "gen_kwargs": {
    #             "do_sample": False,
    #             "max_new_tokens": 32
    #         },
    #     },
    #     "experiments__debug__bace__test": {
    #         "compute_metrics": ComputeAucMetrics(tokenizer=tokenizer),
    #         "preprocess_generated_output_logits_for_metrics": BinaryClassificationProbabilityCalculator(
    #             tokenizer=tokenizer, positive_token_text=" Yes", negative_token_text=" No"), # space is needed
    #         "gen_kwargs": {
    #             "do_sample": False,
    #             "max_new_tokens": 32,
    #             "output_logits": True,
    #             "return_dict_in_generate": True,
    #         },
    #     },
    # }

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()
    
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation
    
    # print(f"{dataset_module['eval_dataset']=}")
    # print(f"{dataset_module['eval_dataset']['experiments__debug__bace__test']['labels']=}")
    # exit()
    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)