# UniCanvas: Affordance-Aware Unified Real Image Editing via Customized Text-to-image Generation (IJCV)

<a href="https://jinjianrick.github.io/unicanvas/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>

![teaser](pics/teaser.png)

Given a source real image and a target subject specified by several reference images, UniCanvas can seamlessly render the target subject into a designated region of the source image, while simultaneously being able to perform semantic edits on the resultant image in a precise and effortless manner.

> <a href="https://jinjianrick.github.io/unicanvas/">**UniCanvas: Affordance-Aware Unified Real Image Editing via Customized Text-to-image Generation (IJCV)**</a>
>
> <a href="https://chenjin.netlify.app/">Jian Jin<sup>1</sup></a>,
    <a href="https://rt416.github.io/">Yang Shen<sup>1</sup></a>, 
    <a href="https://uk.linkedin.com/in/amrutha-saseendran">Xinyang Zhao<sup>1</sup> </a>, 
    <a href="https://tomdiethe.com/">Zhenyong Fu<sup>1</sup></a>, 
    <a href="https://uk.linkedin.com/in/philteare">Jian Yang<sup>1</sup></a><br>
>
> In International Journal of Computer Vision 2025

## Results
![teaser](pics/res1.png)

![teaser](pics/res3.png)

![teaser](pics/res2.png)


## Getting Started

Our code builds on, and shares requirements with [Latent Diffusion Models (LDM)](https://github.com/CompVis/latent-diffusion). To set up their environment, please run:

```
cd stable-diffusion-main
conda env create -f environment.yaml
conda activate ldm
pip install clip-retrieval tqdm
```

```
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
cd ./src/taming-transformers
pip install -e .
```

Pre-trained model download

```
wget -P ./pretrain_model https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt 
```

## fine-tuning
```
bash train_two.sh
```

#### Parameters in train_two.sh

- **`caption="<new1> <new2> <new3> teddybear"`**: Caption of foreground subject. `<new1> <new2> <new3>` are learnable tokens.
- **`task_token=3`**: The number of learnable tokens in the caption of foreground subject.
- **`datapath='data/subject/teddybear'`**: Path to the reference images of the foreground subject.
- **`reg_data='real_reg/samples_teddybear'`**: Regularization data for the foreground subject.
- **`class_name='teddybear'`**: Coarse class noun for the foreground subject.
- **`caption_background="Photo of <new4> sofa <new5> wall <new6> floor."`**: Image caption of the source image.
- **`backgroundpath='data/source/chair2'`**: Path to the source image.
- **`task_name='teddybear+chair2'`**: Task name.

Increase the fine-tuning step ```max_steps``` in ```configs/finetune_joint.yaml``` if the background reconstruction is poor, and decrease it if the foreground object fails to render.

## inference

```
python3 sample.py \
   --prompt_foreground "<new1> <new2> <new3> teddybear on the sofa." \
   --prompt_subject "<new1> <new2> <new3> teddybear" \
   --prompt_background "Photo of <new4> sofa <new5> wall <new6> floor." \
   --delta_ckpt logs/2025-03-27T06-20-29_teddybear+chair2-sdv4/checkpoints/embeddings_gs-1499.ckpt \
   --ckpt ./pretrain_model/sd-v1-4.ckpt \
   --device "cuda:1" \
   --specific_epoch '1499' \
   --layout_sup True \
   --cx 250 \
   --cy 350 \
   --scale_x 200 \
   --scale_y 200
```


#### Parameter Explanation

- **`"<new1> <new2> <new3> teddybear on the sofa."`**: Textual condition for the subject branch.
- **`"Photo of <new4> sofa <new5> wall <new6> floor."`**: Textual condition for the image branch.
- **`log/2024-04-19T07-12-17_teddybear+chair2-sdv4/checkpoints/embeddings_gs-1499.ckpt`**: Path to the fine-tuned model.
- **`specific_epoch`**: Specifies the index of the saved checkpoint file.
- **`cx, cy, scale_x, scale_y`**: These four parameters specify the location of the target region of the foreground subject.  
  - `cx`: x center  
  - `cy`: y center 
  - `scale_x`: Height  
  - `scale_y`: Width  

---

##  Acknowledgements
This code is based on the [Stable Diffusion](https://github.com/CompVis/latent-diffusion) and [Custom Diffusion](https://github.com/adobe-research/custom-diffusion). Thank them for their outstanding work.

## Citation

If you make use of our work, please cite our paper:

```
@article{jin2025unicanvas,
        title={UniCanvas: Affordance-Aware Unified Real Image Editing via Customized Text-to-Image Generation},
        author={Jin, Jian and Shen, Yang and Zhao, Xinyang and Fu, Zhenyong and Yang, Jian},
        journal={International Journal of Computer Vision},
        pages={1--25},
        year={2025},
        publisher={Springer}
      }
```
