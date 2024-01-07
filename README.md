# tensorrt_llm使用流程：

## 1. 使用Docker Image：

tongxuanliu/dev:tensorrtllm ，已安装tensorrt_llm v0.7.1版本及相关包

## 2. 修复tensorrt_llmv0.7.1 bug

进入Docker后，手动将本代码库中的model_runner.py，覆盖/usr/local/lib/python3.10/dist-packages/tensorrt_llm/runtime/model_runner.py （修复v0.7.1版本中的bug）

## 3. 下载TensorRT-LLM代码

git clone https://github.com/NVIDIA/TensorRT-LLM.git

## 4. huggingface模型转换trt ckpt格式

python TensorRT-LLM/examples/opt/convert_checkpoint.py --model_dir=/data/tensorrtllm_test/opt-13b/ --dtype float16 --output_dir=/data/tensorrtllm_test/tllm_13b_checkpoint --workers 2

重要参数：
--workers 并行个数（tensor并行个数）
--model_dir huggingface模型的位置
--dtype 类型
--output_dir 输出ckpt位置

## 5. trt ckpt build

trtllm-build --checkpoint_dir /data/tensorrtllm_test/tllm_13b_checkpoint --use_gemm_plugin float16 --use_gpt_attention_plugin float16 --max_batch_size 64 --max_input_len 924  --max_output_len 128  --workers 2 --output_dir /data/tensorrtllm_test/opt-13b-trtllm-build 

--max_batch_size batch_size大小
--max_input_len 输入length
--max_output_len 输出length
--output_dir 输出文件路径
--checkpoint_dir ckpt的路径
--workers 并行个数（tensor并行个数）

## 6. 单卡跑某个数据集命令

 python run.py --max_output_len=100 --every_batch_cost_print True --tokenizer_dir /data/tensorrtllm_test/opt-13b/ --engine_dir /data/tensorrtllm_test/opt-13b-trtllm-build/ --input_file /data/opt-13b-test/Chatbot_group_10.json --batch_size 8


## 7. 两卡跑某个数据集命令

mpirun -n 2  python run.py --max_output_len=100 --every_batch_cost_print True --tokenizer_dir /data/tensorrtllm_test/opt-13b/ --engine_dir /data/tensorrtllm_test/opt-13b-trtllm-build/ --input_file /data/opt-13b-test/Chatbot_group_10.json --batch_size 8
