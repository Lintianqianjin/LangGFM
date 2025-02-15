# python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/node_counting/test/instruction_dataset.json --model_name lms/qwen2.5-72b-instruct/ --url http://8.147.110.29:8018/v1 --api_key zkxue_test
# python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/edge_counting/test/instruction_dataset.json --model_name lms/qwen2.5-72b-instruct/ --url http://8.147.110.29:8018/v1 --api_key zkxue_test
# python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/node_attribute_retrieval/test/instruction_dataset.json --model_name lms/qwen2.5-72b-instruct/ --url http://8.147.110.29:8018/v1 --api_key zkxue_test
# python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/edge_attribute_retrieval/test/instruction_dataset.json --model_name lms/qwen2.5-72b-instruct/ --url http://8.147.110.29:8018/v1 --api_key zkxue_test
# python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/degree_counting/test/instruction_dataset.json --model_name lms/qwen2.5-72b-instruct/ --url http://8.147.110.29:8018/v1 --api_key zkxue_test
# python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/edge_existence/test/instruction_dataset.json --model_name lms/qwen2.5-72b-instruct/ --url http://8.147.110.29:8018/v1 --api_key zkxue_test
# python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/connectivity/test/instruction_dataset.json --model_name lms/qwen2.5-72b-instruct/ --url http://8.147.110.29:8018/v1 --api_key zkxue_test
# python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/cycle_checking/test/instruction_dataset.json --model_name lms/qwen2.5-72b-instruct/ --url http://8.147.110.29:8018/v1 --api_key zkxue_test
# python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/graph_structure_detection/test/instruction_dataset.json --model_name lms/qwen2.5-72b-instruct/ --url http://8.147.110.29:8018/v1 --api_key zkxue_test
# python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/shortest_path/test/instruction_dataset.json --model_name lms/qwen2.5-72b-instruct/ --url http://8.147.110.29:8018/v1 --api_key zkxue_test --tmp_instruction "\n Answer the question in a list format like [4,2,1,3]."
# python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/hamilton_path/test/instruction_dataset.json --model_name lms/qwen2.5-72b-instruct/ --url http://8.147.110.29:8018/v1 --api_key zkxue_test 
# python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/graph_automorphic/test/instruction_dataset.json --model_name lms/qwen2.5-72b-instruct/ --url http://8.147.110.29:8018/v1 --api_key zkxue_test

# # llama-3.3-70b

# python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/node_counting/test/instruction_dataset.json --model_name meta-llama/Llama-3.3-70B-Instruct
# python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/edge_counting/test/instruction_dataset.json --model_name meta-llama/Llama-3.3-70B-Instruct
# python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/node_attribute_retrieval/test/instruction_dataset.json --model_name meta-llama/Llama-3.3-70B-Instruct

python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/edge_attribute_retrieval/test/instruction_dataset.json --model_name meta-llama/Llama-3.3-70B-Instruct
python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/degree_counting/test/instruction_dataset.json --model_name meta-llama/Llama-3.3-70B-Instruct
python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/edge_existence/test/instruction_dataset.json --model_name meta-llama/Llama-3.3-70B-Instruct
python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/connectivity/test/instruction_dataset.json --model_name meta-llama/Llama-3.3-70B-Instruct
python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/cycle_checking/test/instruction_dataset.json --model_name meta-llama/Llama-3.3-70B-Instruct
python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/graph_structure_detection/test/instruction_dataset.json --model_name meta-llama/Llama-3.3-70B-Instruct
python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/shortest_path/test/instruction_dataset.json --model_name meta-llama/Llama-3.3-70B-Instruct --tmp_instruction "\n Answer the question in a list format like [4,2,1,3]."
python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/hamilton_path/test/instruction_dataset.json --model_name meta-llama/Llama-3.3-70B-Instruct
python scripts/eval_open_llm_api.py --file_path experiments/langgfm_i/graph_automorphic/test/instruction_dataset.json --model_name meta-llama/Llama-3.3-70B-Instruct

