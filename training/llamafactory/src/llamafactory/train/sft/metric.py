# Copyright 2024 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Union, List, Any

import re
import numpy as np
import torch
import torch.nn.functional as F

from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available


from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


if is_rouge_available():
    from rouge_chinese import Rouge


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""
    Computes the token with the largest likelihood to reduce memory footprint.
    """
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)



class BinaryClassificationProbabilityCalculator:
    def __init__(self, tokenizer, positive_token_text="Yes", negative_token_text="No"):
        """
        Initializes with a tokenizer, positive token text, and negative token text.
        Computes the token ids for both positive and negative tokens.
        Note: This assumes that each token text corresponds to a single token.
        """
        positive_token_ids = tokenizer.encode(positive_token_text, add_special_tokens=False)
        negative_token_ids = tokenizer.encode(negative_token_text, add_special_tokens=False)
        
        if len(positive_token_ids) != 1 or len(negative_token_ids) != 1:
            raise ValueError("positive_token_text and negative_token_text must each correspond to a single token")
        
        self.positive_token_id = positive_token_ids[0]
        self.negative_token_id = negative_token_ids[0]
        # print(f"{self.positive_token_id=}, {self.negative_token_id=}")
        # exit()
    
    def __call__(self, output_logits):
        """
        Takes output_logits (shape [batch_size, sequence_length, vocab_size]) and returns a tensor 
        of shape [batch_size, sequence_length], where each element is the probability of the positive token.
        """
        # Extract logits for the positive and negative tokens
        positive_logits = output_logits[:, :, self.positive_token_id]
        negative_logits = output_logits[:, :, self.negative_token_id]
        
        # Stack the logits to form a tensor of shape [batch_size, sequence_length, 2]
        # combined_logits = torch.stack([positive_logits, negative_logits], dim=-1)
        logit_diff = positive_logits - negative_logits
        
        # Apply softmax on the last dimension to obtain the probability distribution
        # probabilities = torch.softmax(combined_logits, dim=-1)
        positive_prob = 1 / (1 + torch.exp(-logit_diff))
        
        # print(positive_prob.shape)
        # Return the probability corresponding to the positive token (assumed at index 0)
        return positive_prob


@dataclass
class ComputeAccuracy:
    r"""
    Computes accuracy and supports `batch_eval_metrics`.
    """

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        # decode text
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""
    Computes text similarity scores and supports `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # accuray f1 

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()


@dataclass
class ComputeExactMatch:
    r"""
    Computes ExactMatch and supports `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"exact_match": []}
        # print(f"{result=}")
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        for pred, label in zip(decoded_preds, decoded_labels):
            # print(f"pred : {pred}")
            # print(f"label: {label}")
            # print("\n")
            if pred == label:
                self.score_dict["exact_match"].append(1)
            else:
                self.score_dict["exact_match"].append(0)
            # print(f"pred == label, {pred == label}")
            
        if compute_result:
            # print(f"Sample Count: {len(preds)=}")
            return self._dump()


from langgfm.utils.number_process import NumberFormatter
@dataclass
class ComputeRegressionMetrics:
    """
    Computes regression metrics and supports `batch_eval_metrics`.
    Extracts numerical values from model outputs and labels for regression tasks.
    """
    
    tokenizer: "PreTrainedTokenizer"
    
    def __post_init__(self):
        self._dump()
    
    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}
            # print(f"{result=}")
        self.score_dict = {
            "mae": [],
            "rmse": [],
            "spearman_corr": [],
            "pearson_corr": []
        }
        return result
    
    
    def extract_value_from_text(self, text: str) -> float:
        """
        Extract numerical value from text, specifically looking for content between <answer> and </answer> tags.
        If no valid number is found, return 0.0.
        """
        # Find content between <answer> and </answer> tags
        answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        answer_match = answer_pattern.search(text)
        
        if answer_match:
            content = answer_match.group(1).strip()
            # Try to extract a number (float or int) from the content
            # number_pattern = re.compile(r'[-+]?\d*\.?\d+')
            # number_match = number_pattern.search(content)
            try:
                number_match = NumberFormatter.parse_formatted_text(content)
                return number_match
            except:
                return 0.0
            # if number_match:
            #     # num_str = number_match
            #     # Convert to float (works for both integers and floats)
            #     # return float(num_str)
            #     return number_match
        
        return 0.0
    
    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        
        # print(f"{eval_preds=}")
        # exit()
        
        # Extract predictions and labels
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        
        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        
        # Extract numerical values from decoded text
        pred_values = []
        label_values = []
        
        for pred, label in zip(decoded_preds, decoded_labels):
            
            # print("\n")
            # Extract values
            pred_value = self.extract_value_from_text(pred)
            label_value = self.extract_value_from_text(label)
            
            pred_values.append(pred_value)
            label_values.append(label_value)
            
            # Individual metrics
            self.score_dict["mae"].append(abs(pred_value - label_value))
            self.score_dict["rmse"].append((pred_value - label_value)**2)
            
            print("\n")
            print(f"pred : {pred}")
            print(f"label: {label}")
            print(f"Extracted: pred={pred_value}, label={label_value}, diff={abs(pred_value - label_value)}")
            print("\n")
            
        # Convert to numpy arrays for batch calculations
        pred_values = np.array(pred_values)
        label_values = np.array(label_values)
        
        # Compute dataset-level metrics
        # RMSE needs to be square-rooted after averaging
        self.score_dict["rmse"] = [np.sqrt(np.mean(self.score_dict["rmse"]))]
        
        # Compute correlations for the entire batch
        if len(pred_values) > 1:  # Correlation requires at least 2 points
            try:
                spearman = spearmanr(pred_values, label_values)[0]
                if np.isnan(spearman):
                    spearman = 0.0
                self.score_dict["spearman_corr"] = [spearman]
            except:
                self.score_dict["spearman_corr"] = [0.0]
                
            try:
                pearson = pearsonr(pred_values, label_values)[0]
                if np.isnan(pearson):
                    pearson = 0.0
                self.score_dict["pearson_corr"] = [pearson]
            except:
                self.score_dict["pearson_corr"] = [0.0]
        else:
            # Cannot compute correlation with a single sample
            self.score_dict["spearman_corr"] = [0.0]
            self.score_dict["pearson_corr"] = [0.0]
            
        if compute_result:
            # print(f"Sample Count: {len(preds)=}")
            return self._dump()


@dataclass
class ComputeAucMetrics:
    """
    Computes AUC for Yes/No answers within <answer> tags and supports batch evaluation.
    
    This metrics computer works by finding Yes/No answers in <answer> tags in the generated text,
    extracting the logits for Yes/No tokens, and computing AUC based on the true labels.
    """
    tokenizer: Any
    score_dict: Dict[str, List[float]] = field(default_factory=lambda: {"auc": []})
    positive_text: str = "Yes"
    negative_text: str = "No"
    
    
    def _dump(self) -> Optional[Dict[str, float]]:
        """Compute final metrics from accumulated scores and reset score_dict."""
        result = None
        if hasattr(self, "score_dict") and self.score_dict["auc"]:
            # Calculate mean of the scores if we have any
            result = {"auc": float(np.mean(self.score_dict["auc"]))}
        
        # Reset scores for next evaluation
        self.score_dict = {"auc": []}
        return result
    
    def __post_init__(self):
        """Initialize token IDs and reset scores after initialization."""
        # Get the token IDs for "<answer>"
        self.answer_tag_tokens = self.tokenizer.encode("<answer>", add_special_tokens=False)
        self._dump()
    
    def find_token_position(self, pred_tokens, target_token_idx_in_answer=0):
        # Convert pred_tokens to a list if it's a tensor or array
        if not isinstance(pred_tokens, list):
            pred_tokens = pred_tokens.tolist()
        
        # Find where the answer tag appears in pred_tokens
        for idx in range(len(pred_tokens) - len(self.answer_tag_tokens) + 1):
            if pred_tokens[idx:idx+len(self.answer_tag_tokens)] == self.answer_tag_tokens:
                # The response starts right after the "<answer>" tag
                response_start_idx = idx + len(self.answer_tag_tokens)
                
                # The i-th token position is response_start_idx + i
                return response_start_idx + target_token_idx_in_answer
    
        return None  # Tag not found

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        """
        Process each prediction and compute AUC.
        
        Args:
            eval_preds: Predictions and labels
            compute_result: Whether to calculate and return final results
            
        Returns:
            Dictionary with metrics if compute_result is True, otherwise None
        """
        predictions, label_ids = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        pred_tokens, postive_token_prob = predictions # output_logits shape -> [batch_size, response_sequence_length]
        
        # print(f"{output_logits=}")
        # print(f"{output_logits.shape=}")
        # exit()
        
        # left_padding of labels is INGOE_INDEX due to pad_to_multiple_of for GPU efficient computing
        # Extract ground truth labels (1 for Yes, 0 for No)
        label_ids = np.where(label_ids != IGNORE_INDEX, label_ids, self.tokenizer.pad_token_id)
        label_texts = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        true_labels = []
        for label_text in label_texts:            
            # Extract Yes/No from <answer> tag
            answer_match = re.search(r'<answer>(.*?)</answer>', label_text)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                if answer_text == self.positive_text:
                    true_labels.append(1)
                elif answer_text == self.negative_text:
                    true_labels.append(0)
                else:
                    raise ValueError(f"Unknown answer text: {answer_text}")
            else:
                raise ValueError("No <answer> tag found in label")
            # print(f"{label_text=}, Assigned Label={true_labels[-1]}")
        # left padding of pred_tokens is pad_token_id (padded in input_ids)
        # right padding of pred_tokens is IGNORE_INDEX (padded in EvalLoopContainer.add)
        # Extract predicted probabilities
        pred_tokens = np.where(pred_tokens != IGNORE_INDEX, pred_tokens, self.tokenizer.pad_token_id)
        pred_texts = self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        
        # def extract_non_special_tokens(tokenizer, tokens):
        #     """
        #     参数:
        #     tokenizer: 已初始化的tokenizer，需提供：
        #         - tokenizer.all_special_tokens: 一个列表，包含所有special token（例如"[CLS]"、"[SEP]"等）
        #         - tokenizer.convert_tokens_to_string: 将一个token列表转换成对应的文本piece
        #     tokens: token的列表，可能包含special tokens

        #     返回:
        #     一个字典，键为非special token（注意：如果有重复token，后者会覆盖前者），
        #     值为一个字典，包含两个键：
        #         - "text_piece": 对应的文本piece
        #         - "non_special_index": 该token在所有非special token中的顺序（从0开始）
        #     """
        #     result = []
        #     non_special_counter = 0

        #     for token in tokens:
        #         # 如果该token是special token，则跳过
        #         text_piece = tokenizer.decode([token])
        #         if text_piece in tokenizer.all_special_tokens:
        #             continue
        #         result.append((
        #             non_special_counter, text_piece, int(token)
        #         ))
        #         non_special_counter += 1

        #     return result


        predicted_probs = []
        for sample_idx, pred_text in enumerate(pred_texts):
            
            # print(f"\n{label_texts[sample_idx]=}\n")
            # print(f"\n{label_ids[sample_idx]=}\n")


            # print(f"\n{pred_text=}\n")
            sample_pred_tokens = pred_tokens[sample_idx]
            # ignore left padding
            sample_pred_tokens = sample_pred_tokens[sample_pred_tokens != self.tokenizer.pad_token_id]
            # print(f"\n{sample_pred_tokens=}\n")
            # print(f"\n{pred_tokens[sample_idx]=}\n")
            # token_text_mapping = extract_non_special_tokens(self.tokenizer, pred_tokens[sample_idx])
            # print(f"\n{token_text_mapping=}\n")
            # exit()
            # Extract text from <answer> tag
            answer_match = re.search(r'<answer>(.*?)</answer>', pred_text)
            
            if not answer_match:
                # No <answer> tag found
                predicted_probs.append(0.5)
                continue
            
            answer_text = answer_match.group(1).strip()
            
            if answer_text not in [self.positive_text, self.negative_text]:
                # Answer is neither Yes nor No
                predicted_probs.append(0.5)
                continue
            
            
            classification_target_token_idx = self.find_token_position(sample_pred_tokens, target_token_idx_in_answer=0)
            # print(f"{classification_target_token_idx=}")
            predicted_probs.append(postive_token_prob[sample_idx][classification_target_token_idx])
        
        # compute AUC
        auc_score = roc_auc_score(true_labels, predicted_probs)
        self.score_dict["auc"] = [auc_score]
        
        if compute_result:
            return self._dump()
        return None
