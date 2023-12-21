# 用Lora和deepspeed微调LLaMA2-Chat

在一块P100或T4（16G）上微调Llama-2-7b-chat模型。

数据源采用了alpaca格式，由train和validation两个数据源组成。

## 1、显卡要求

16G显存及以上（P100或T4及以上），一块或多块。

## 2、Clone源码

```bash
git clone https://github.com/linzhonghong/llama2-lora-fine-tuning
cd llama2-lora-fine-tuning
```

## 3、安装依赖环境

```bash
# 创建虚拟环境
conda create -n llama2 python=3.9 -y
conda activate llama2
# 安装依赖包
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
# 验证bitsandbytes，如果cuda driver和runtime不一致或者有多个版本，需要指定版本先：export BNB_CUDA_VERSION=110，接下来运行微调也需要。
python -m bitsandbytes

```

## 4、下载原始模型

```bash
python model_download.py --repo_id daryl149/llama-2-7b-chat-hf
```

## 5、扩充中文词表

```bash
# 使用了https://github.com/ymcui/Chinese-LLaMA-Alpaca.git的方法扩充中文词表
# 扩充完的词表在merged_tokenizes_sp（全精度）和merged_tokenizer_hf（半精度）
# 在微调时，将使用--tokenizer_name ./merged_tokenizer_hf参数
python merge_tokenizers.py \
  --llama_tokenizer_dir ./models/daryl149/llama-2-7b-chat-hf \
  --chinese_sp_model_file ./chinese_sp.model
```

## 6、微调参数说明

有以下几个参数可以调整：

| 参数                        | 说明                       | 取值                                                         |
| --------------------------- | -------------------------- | ------------------------------------------------------------ |
| load_in_bits                | 模型精度                   | 4和8，如果显存不溢出，尽量选高精度8                          |
| block_size                  | token最大长度              | 首选2048，内存溢出，可选1024、512等                          |
| per_device_train_batch_size | 训练时每块卡每次装入批量数 | 只要内存不溢出，尽量往大选                                   |
| per_device_eval_batch_size  | 评估时每块卡每次装入批量数 | 只要内存不溢出，尽量往大选                                   |
| include                     | 使用的显卡序列             | 如两块：localhost:1,2（特别注意的是，序列与nvidia-smi看到的不一定一样） |
| num_train_epochs            | 训练轮数                   | 至少3轮                                                      |

## 7、微调

```bash
chmod +x finetune-lora.sh
# 微调
screen -L -Logfile train.log ./finetune-lora.sh
```

## 8、测试

不用等整个训练过程完成，因为每200步会产生一个检查点，采用以下命令在命令行测试推理效果：

```bash
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --base_model './models/daryl149/llama-2-7b-chat-hf' \
    --lora_weights 'output/checkpoint-2000' \
    --load_8bit #不加这个参数是用的4bit
```

## 9、多机多卡微调

使用多机多卡需要配置ssh无密码登录和修改脚本参数，注意多机上的环境和目录需要一致

### 9.1、配置ssh无密码登录

#### master节点创建公钥和私钥
在主节点(master)上输入以下命令创建一组公钥和私钥:
```
ssh-keygen
```
直接enter就可以。

生成是密钥在`~/.ssh`目录下
```
ll ~/.ssh/
total 16
-rw------- 1 root root 1028 Dec 20 15:26 authorized_keys
-rw------- 1 root root 2590 Dec 20 14:46 id_rsa
-rw-r--r-- 1 root root  557 Dec 20 14:46 id_rsa.pub
-rw-r--r-- 1 root root  362 Dec 20 16:17 known_hosts
```
`id_rsa`是私钥，`id_rsa.pub`是公钥，需要将公钥拷贝到`authorized_keys`，并将公钥上传到其他各节点，拷贝到`authorized_keys`

#### 测试登录
```
ssh root@ip
```

#### host配置
这里可配置也可不配置，不配置的话，就直接使用IP即可。
```
cat /etc/hosts
192.168.1.2 master
192.168.1.3 worker
```

### 9.2、安装和配置pdsh
pdsh是deepspeed里面的一种分布式训练工具。在多机情况下使用，它的优点是只需要在一台机上运行脚本就可以，pdsh会自动把命令和环境变量推送到其他节点上，然后汇总所有节点的日志到主节点。
```
yum install epel-release
yum install dnf
yum install pdsh
dnf install pdsh-rcmd-ssh
```

### 9.3、多机多卡代码修改
各机器上的代码路径和代码内容需一致。

虚拟环境位置也要一致，注意如果已经安装了conda，可能各个节点安装路径不一样，需要修改成一样的。

#### 创建hostfile
```
vim hostfile
master slots=1
worker slots=1
```
第一列是hostname，也可以直接填IP，slots是卡的数量。

#### 修改finetune-lora.sh脚本
需要在脚本里面为`deepspeed`添加两个参数，指定`hostfile`和每台节点实际使用的卡号。
```
vim finetune-lora.sh
deepspeed --hostfile hostfile --include master:0@worker:0
```
保存脚本后就可以进行多机多卡微调了。
