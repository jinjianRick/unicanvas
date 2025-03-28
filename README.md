### 配置环境
cd stable-diffusion-main
conda env create -f environment.yaml
conda activate ldm
pip install clip-retrieval tqdm

#### 预训练模型下载
wget -P ./pretrain_model https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt 






#### fine-tuning
bash train_two.sh "<new1> <new2> <new3> teddybear" data/subject/teddybear real_reg/samples_teddybear teddybear finetune_joint.yaml ./pretrain_model/sd-v1-4.ckpt "Photo of <new4> sofa <new5> wall <new6> floor." data/source/chair2 teddybear+chair2

# 参数解释
"<new1> <new2> <new3> teddybear": foreground subject, <new1> <new2> <new3> is learnable token
data/subject/teddybear: path to the refernce images of the foreground subject
real_reg/samples_teddybear: reg data of foreground subject
teddybear: corase class noun of foreground subject
"Photo of <new4> sofa <new5> wall <new6> floor.": textual description of source image
data/source/chair2: path to source image
teddybear+chair2: task name

# 注意事项
1. 会得到两种fine-tuned model : 一种前缀是embeddings_gs-，大小为200多M，另一种前缀是epoch=0000，4个多G，这两个模型的作用是等效的，建议只留前面那个小的
2. fine_tuning中涉及的一些其他参数在train_two.sh文件中更改
3. 当更改foreground subject 时, train_two.sh文件中的figure_num, token_pos等参数需要更改







#### inference
python3 sample.py --prompt "<new1> <new2> <new3> teddybear on the sofa." --prompt_ldm "Photo of <new4> sofa <new5> wall <new6> floor." --delta_ckpt log/2024-04-19T07-12-17_teddybear+chair2-sdv4/checkpoints/embeddings_gs-1499.ckpt --ckpt ./pretrain_model/sd-v1-4.ckpt --device "cuda:1" --specific_epoch '1499' --layout_sup True --use_dci True --length_custom 8 --length_ldm 9 --mask_custom 0,1,2,3 --mask_ldm 0,1,2,3,4,5,6,7,8 --cx 250 --cy 350 --random_scale_x 200 --random_scale_y 200

# 参数解释
"<new1> <new2> <new3> teddybear on the sofa.": textual condition of subject branch
"Photo of <new4> sofa <new5> wall <new6> floor.": textual condition of image branch
log/2024-04-19T07-12-17_teddybear+chair2-sdv4/checkpoints/embeddings_gs-1499.ckpt: path to the fine-tuned model
specific_epoch: specific the index of the save file
length_custom: the token length of the textual condition of subject branch (only the tokens need to provide attention map, i.e., "<new1> <new2> <new3> teddybear" in this demo)
length_ldm: the token length of the textual condition of image branch (only the tokens need to provide attention map, i.e., "Photo of <new4> sofa <new5> wall <new6> floor." in this demo)
mask_custom: the index of the token that need to provide attention map for subject branch (i.e., the index of "<new1>", "<new2>", "<new3>", and "teddybear" in this demo)
mask_ldm: the index of the token that need to provide attention map for image branch.
cx, cy, random_scale_x, random_scale_y: these four parameters specify the location of the target region of foreground subject, cx: top, cy: left, random_scale_x: height, random_scale_y: width

# 注意事项
当更换foreground subject的时候，如果length_custom 发生了改变，要对sample.py 434行的tokens变量做相应改变






