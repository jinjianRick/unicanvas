# This code is built from the Stable Diffusion repository: https://github.com/CompVis/stable-diffusion.
# Copyright (c) 2022 Robin Rombach and Patrick Esser and contributors.
# CreativeML Open RAIL-M
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2022 Adobe Research. All rights reserved.
# Adobe’s modifications are licensed under the Adobe Research License. To view a copy of the license, visit
# LICENSE.md.
#
# ==========================================================================================
#
# CreativeML Open RAIL-M License
#
# Section I: PREAMBLE

# Multimodal generative models are being widely adopted and used, and have the potential to transform the way artists, among other individuals, conceive and benefit from AI or ML technologies as a tool for content creation.

# Notwithstanding the current and potential benefits that these artifacts can bring to society at large, there are also concerns about potential misuses of them, either due to their technical limitations or ethical considerations.

# In short, this license strives for both the open and responsible downstream use of the accompanying model. When it comes to the open character, we took inspiration from open source permissive licenses regarding the grant of IP rights. Referring to the downstream responsible use, we added use-based restrictions not permitting the use of the Model in very specific scenarios, in order for the licensor to be able to enforce the license in case potential misuses of the Model may occur. At the same time, we strive to promote open and responsible research on generative models for art and content generation.

# Even though downstream derivative versions of the model could be released under different licensing terms, the latter will always have to include - at minimum - the same use-based restrictions as the ones in the original license (this license). We believe in the intersection between open and responsible AI development; thus, this License aims to strike a balance between both in order to enable responsible open-science in the field of AI.

# This License governs the use of the model (and its derivatives) and is informed by the model card associated with the model.

# NOW THEREFORE, You and Licensor agree as follows:

# 1. Definitions

# - "License" means the terms and conditions for use, reproduction, and Distribution as defined in this document.
# - "Data" means a collection of information and/or content extracted from the dataset used with the Model, including to train, pretrain, or otherwise evaluate the Model. The Data is not licensed under this License.
# - "Output" means the results of operating a Model as embodied in informational content resulting therefrom.
# - "Model" means any accompanying machine-learning based assemblies (including checkpoints), consisting of learnt weights, parameters (including optimizer states), corresponding to the model architecture as embodied in the Complementary Material, that have been trained or tuned, in whole or in part on the Data, using the Complementary Material.
# - "Derivatives of the Model" means all modifications to the Model, works based on the Model, or any other model which is created or initialized by transfer of patterns of the weights, parameters, activations or output of the Model, to the other model, in order to cause the other model to perform similarly to the Model, including - but not limited to - distillation methods entailing the use of intermediate data representations or methods based on the generation of synthetic data by the Model for training the other model.
# - "Complementary Material" means the accompanying source code and scripts used to define, run, load, benchmark or evaluate the Model, and used to prepare data for training or evaluation, if any. This includes any accompanying documentation, tutorials, examples, etc, if any.
# - "Distribution" means any transmission, reproduction, publication or other sharing of the Model or Derivatives of the Model to a third party, including providing the Model as a hosted service made available by electronic or other remote means - e.g. API-based or web access.
# - "Licensor" means the copyright owner or entity authorized by the copyright owner that is granting the License, including the persons or entities that may have rights in the Model and/or distributing the Model.
# - "You" (or "Your") means an individual or Legal Entity exercising permissions granted by this License and/or making use of the Model for whichever purpose and in any field of use, including usage of the Model in an end-use application - e.g. chatbot, translator, image generator.
# - "Third Parties" means individuals or legal entities that are not under common control with Licensor or You.
# - "Contribution" means any work of authorship, including the original version of the Model and any modifications or additions to that Model or Derivatives of the Model thereof, that is intentionally submitted to Licensor for inclusion in the Model by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, "submitted" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Model, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as "Not a Contribution."
# - "Contributor" means Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Model.

# Section II: INTELLECTUAL PROPERTY RIGHTS

# Both copyright and patent grants apply to the Model, Derivatives of the Model and Complementary Material. The Model and Derivatives of the Model are subject to additional terms as described in Section III.

# 2. Grant of Copyright License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare, publicly display, publicly perform, sublicense, and distribute the Complementary Material, the Model, and Derivatives of the Model.
# 3. Grant of Patent License. Subject to the terms and conditions of this License and where and as applicable, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this paragraph) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Model and the Complementary Material, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Model to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Model and/or Complementary Material or a Contribution incorporated within the Model and/or Complementary Material constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for the Model and/or Work shall terminate as of the date such litigation is asserted or filed.

# Section III: CONDITIONS OF USAGE, DISTRIBUTION AND REDISTRIBUTION

# 4. Distribution and Redistribution. You may host for Third Party remote access purposes (e.g. software-as-a-service), reproduce and distribute copies of the Model or Derivatives of the Model thereof in any medium, with or without modifications, provided that You meet the following conditions:
# Use-based restrictions as referenced in paragraph 5 MUST be included as an enforceable provision by You in any type of legal agreement (e.g. a license) governing the use and/or distribution of the Model or Derivatives of the Model, and You shall give notice to subsequent users You Distribute to, that the Model or Derivatives of the Model are subject to paragraph 5. This provision does not apply to the use of Complementary Material.
# You must give any Third Party recipients of the Model or Derivatives of the Model a copy of this License;
# You must cause any modified files to carry prominent notices stating that You changed the files;
# You must retain all copyright, patent, trademark, and attribution notices excluding those notices that do not pertain to any part of the Model, Derivatives of the Model.
# You may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions - respecting paragraph 4.a. - for use, reproduction, or Distribution of Your modifications, or for any such Derivatives of the Model as a whole, provided Your use, reproduction, and Distribution of the Model otherwise complies with the conditions stated in this License.
# 5. Use-based restrictions. The restrictions set forth in Attachment A are considered Use-based restrictions. Therefore You cannot use the Model and the Derivatives of the Model for the specified restricted uses. You may use the Model subject to this License, including only for lawful purposes and in accordance with the License. Use may include creating any content with, finetuning, updating, running, training, evaluating and/or reparametrizing the Model. You shall require all of Your users who use the Model or a Derivative of the Model to comply with the terms of this paragraph (paragraph 5).
# 6. The Output You Generate. Except as set forth herein, Licensor claims no rights in the Output You generate using the Model. You are accountable for the Output you generate and its subsequent uses. No use of the output can contravene any provision as stated in the License.

# Section IV: OTHER PROVISIONS

# 7. Updates and Runtime Restrictions. To the maximum extent permitted by law, Licensor reserves the right to restrict (remotely or otherwise) usage of the Model in violation of this License, update the Model through electronic means, or modify the Output of the Model based on updates. You shall undertake reasonable efforts to use the latest version of the Model.
# 8. Trademarks and related. Nothing in this License permits You to make use of Licensors’ trademarks, trade names, logos or to otherwise suggest endorsement or misrepresent the relationship between the parties; and any rights not expressly granted herein are reserved by the Licensors.
# 9. Disclaimer of Warranty. Unless required by applicable law or agreed to in writing, Licensor provides the Model and the Complementary Material (and each Contributor provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Model, Derivatives of the Model, and the Complementary Material and assume any risks associated with Your exercise of permissions under this License.
# 10. Limitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Model and the Complementary Material (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.
# 11. Accepting Warranty or Additional Liability. While redistributing the Model, Derivatives of the Model and the Complementary Material thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.
# 12. If any provision of this License is held to be invalid, illegal or unenforceable, the remaining provisions shall be unaffected thereby and remain valid as if such provision had not been set forth herein.

# END OF TERMS AND CONDITIONS




# Attachment A

# Use Restrictions

# You agree not to use the Model or Derivatives of the Model:
# - In any way that violates any applicable national, federal, state, local or international law or regulation;
# - For the purpose of exploiting, harming or attempting to exploit or harm minors in any way;
# - To generate or disseminate verifiably false information and/or content with the purpose of harming others;
# - To generate or disseminate personal identifiable information that can be used to harm an individual;
# - To defame, disparage or otherwise harass others;
# - For fully automated decision making that adversely impacts an individual’s legal rights or otherwise creates or modifies a binding, enforceable obligation;
# - For any use intended to or which has the effect of discriminating against or harming individuals or groups based on online or offline social behavior or known or predicted personal or personality characteristics;
# - To exploit any of the vulnerabilities of a specific group of persons based on their age, social, physical or mental characteristics, in order to materially distort the behavior of a person pertaining to that group in a manner that causes or is likely to cause that person or another person physical or psychological harm;
# - For any use intended to or which has the effect of discriminating against individuals or groups based on legally protected characteristics or categories;
# - To provide medical advice and medical results interpretation;
# - To generate or disseminate information for the purpose to be used for administration of justice, law enforcement, immigration or asylum processes, such as predicting an individual will commit fraud/crime commitment (e.g. by text profiling, drawing causal relationships between assertions made in documents, indiscriminate and arbitrarily-targeted use).

import argparse, os, sys, glob, json
sys.path.append('stable-diffusion')
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torch.nn.functional as F
import clip

sys.path.append("./stable-diffusion-main") 

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import wandb

def load_model_from_config(config, ckpt, verbose=False, device=None):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)

    token_weights = sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
    del sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
    m, u = model.load_state_dict(sd, strict=False)
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[:token_weights.shape[0]] = token_weights
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default=None,
        help="the prompt to render"
    )
    parser.add_argument(
        "--prompt_ldm",
        type=str,
        nargs="?",
        default=None,
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=6,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=6.,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/custom-diffusion/finetune.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to checkpoint of the pre-trained model",
    )
    parser.add_argument(
        "--delta_ckpt",
        type=str,
        default=None,
        help="path to delta checkpoint of fine-tuned custom diffusion block",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--wandb_log",
        action='store_true',
        help="save grid images to wandb.",
    )
    parser.add_argument(
        "--compress",
        action='store_true',
        help="delta path provided is a compressed checkpoint.",
    )
    parser.add_argument(
        "--modifier_token",
        type=str,
        default=None,
        help="A token to use as a modifier for the concept.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device",
    )
    parser.add_argument(
        "--specific_epoch",
        type=str,
        default=None,
        help="the fine-tuning epoch of ckpt",
    )
    parser.add_argument(
        "--add_prompt",
        type=str,
        default=None,
        help="add task-specific context and class noun in the query prompt",
    )
    parser.add_argument(
        "--length_ldm",
        type=int,
        default=5,
        help="length of ldm prompt",
    )
    parser.add_argument(
        "--length_custom",
        type=int,
        default=0,
        help="length of custom prompt",
    )
    parser.add_argument(
        "--mask_custom",
        type=str,
        default=None,
        help="custom prompt mask",
    )

    parser.add_argument(
        "--mask_ldm",
        type=str,
        default=[1],
        help="ldm prompt mask",
    )
    parser.add_argument(
        "--use_dci",
        default=False,
        help="whether use dci",
    )
    parser.add_argument(
        "--dci_type",
        type=str,
        default='same',
        help="the fine-tuned model used in dci whether the same one",
    )
    parser.add_argument(
        "--layout_sup",
        default=False,
        help="whether use layout supervision at inference-time",
    )
    parser.add_argument(
        "--b_edit",
        default=False,
        help="whether edit the background",
    )
    parser.add_argument(
        "--bl_path",
        type=str,
        default=None,
        help="path to the layout of the background",
    )
    parser.add_argument(
        "--bk_opt",
        type=str,
        default='get',
        help="path to the layout of the background",
    )
    parser.add_argument(
        "--cor_dict",
        type=str,
        default=None,
        help="mapping from origin to edited",
    )
    parser.add_argument(
        "--cx",
        type=int,
        default=0,
        help="cx",
    )
    parser.add_argument(
        "--cy",
        type=int,
        default=0,
        help="cy",
    )
    parser.add_argument(
        "--random_scale_x",
        type=int,
        default=0,
        help="random_scale_x",
    )
    parser.add_argument(
        "--random_scale_y",
        type=int,
        default=0,
        help="random_scale_y",
    )

    opt = parser.parse_args()
    device = opt.device

    # create mask

    if opt.layout_sup:
        #cx = 95
        #cy = 255
        cx = opt.cx
        cy = opt.cy
        img_size = 512
        random_scale_x = opt.random_scale_x
        random_scale_y = opt.random_scale_y
        sp_sz = 64
        concept_length = 6
        concept_start = 1
        sizereg = 1

        #tokens = {'1':[1,2,3,4,5,6], '2':[7,8,9,10,11]}
        #tokens = {"1": [1,2,3,4,5,6,7,8,9], '2':[10,11,12,13]}
        #tokens = {'1':[1,2], '2':[3,4,5]}
        #tokens = {'1':[1,2,3,4,5,6,7], '2':[8,9,10,11,12,13,14,15,16]}
        #tokens = {'1':[1,2,3,4,5,6,7,8], '2':[9,10,11,12,13,14,15,16,17]}
        #tokens = {'1':[1,2,3,4,5,6,7,8,9], '2':[10,11,12,13,14,15,16,17,18]}
        #tokens = {'1':[1,2,3,4,5,6], '2':[7,8,9,10]} 
        #tokens = {'1':[1,2,3,4,5,6], '2':[7,8,9,10]}   ## tortoise 
        #tokens = {'1':[1,2,3,4,5], '2':[6,7,8,9]}      ## pet dog
        tokens = {'1':[1,2,3,4], '2':[5,6,7,8]}      ## lighthouse
        #tokens = {'1':[1,2,3], '2':[4,5,6,7]}      ## pet dog
        #tokens = {'1':[1,2], '2':[3,4,5,6]} 
        #tokens = {'1':[1,2,3,4,5,6,7,8], '2':[9,10,11,12]}
        #tokens = {'1':[1,2,3,4,5,6], '2':[7]}

        #concept
        layouts = {}
        size_dict = {}
        layouts['1'] = np.zeros((img_size // 8, img_size // 8))
        layouts['1'][(cx - random_scale_x // 2) // 8 + 1: (cx + random_scale_x // 2) // 8 - 1, (cy - random_scale_y // 2) // 8 + 1: (cy + random_scale_y // 2) // 8 - 1] = 1.
        size_dict['1'] = 1 - (random_scale_x*random_scale_y)/np.power(img_size,2)
        #size_dict['1'] = 10

        #background
        layouts['2'] = np.ones((img_size // 8, img_size // 8))
        layouts['2'][(cx - random_scale_x // 2) // 8 + 1: (cx + random_scale_x // 2) // 8 - 1, (cy - random_scale_y // 2) // 8 + 1: (cy + random_scale_y // 2) // 8 - 1] = 0.
        size_dict['2'] = (random_scale_x*random_scale_y)/np.power(img_size,2)
        #size_dict['2'] = 0.1

        pww_maps = torch.zeros(1,77,sp_sz,sp_sz)
        token_size_reg = torch.zeros(1,77)
        for key in tokens.keys():
            pos_list = tokens[key]
            for pos in pos_list:
                pww_maps[:, pos,:,:] = torch.tensor(layouts[key])
                token_size_reg[:, pos] = torch.tensor(size_dict[key])

        #np.savetxt('b2.txt', pww_maps.view(77,-1).detach().cpu().numpy(), fmt='%d')

        creg_maps = {}
        for r in range(4):
            res = int(sp_sz/np.power(2,r))
            layout_c = F.interpolate(pww_maps,(res,res),mode='nearest').view(1,77,-1).permute(0,2,1).repeat(opt.n_samples,1,1) ####!!!!! .repeat(bzs,1,1)
            creg_maps[np.power(res, 2)] = layout_c.to(device)

        sreg_maps = {}
        reg_sizes = {}
        c_reg_sizes = {}
        layout_all = []
        for key in layouts.keys():
            layout_all.append(torch.tensor(layouts[key]).unsqueeze(0))
        layout_all = torch.cat(layout_all)

        for r in range(4):
            res = int(sp_sz/np.power(2,r))
            layouts_s = F.interpolate(layout_all.unsqueeze(1),(res, res),mode='nearest')
            layouts_s = (layouts_s.view(layouts_s.size(0),1,-1)*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0).repeat(opt.n_samples,1,1)
            reg_sizes[np.power(res, 2)] = 1-sizereg*layouts_s.sum(-1, keepdim=True)/(np.power(res, 2))
            #np.savetxt(str(r)+'.txt', layouts_s.view(layouts_s.shape[0], -1).detach().cpu().numpy())
            sreg_maps[np.power(res, 2)] = layouts_s
            c_reg_sizes[np.power(res, 2)] = token_size_reg.repeat(opt.n_samples, 1, 1)


        layout_info = {}
        layout_info['creg_maps'] = creg_maps
        layout_info['sreg_maps'] = sreg_maps
        layout_info['c_size_reg'] = c_reg_sizes
        layout_info['size_reg'] = reg_sizes
        layout_info['creg'] = 5    # tank toy 1
        
    else:
        layout_info = None


    # end create mask

    if opt.wandb_log:
        if opt.delta_ckpt is not None:
            name = opt.delta_ckpt.split('/')[-3]
        elif 'checkpoints' in opt.ckpt:
            name = opt.ckpt.split('/')[-3]
        else:
            name = opt.ckpt.split('/')[-1]
        wandb.init(project="custom-diffusion", entity="cmu-gil", name=name )

    if opt.delta_ckpt is not None:
        if len(glob.glob(os.path.join(opt.delta_ckpt.split('checkpoints')[0], "configs/*.yaml"))) > 0:
            opt.config = sorted(glob.glob(os.path.join(opt.delta_ckpt.split('checkpoints')[0], "configs/*.yaml")))[-1]
    else:
        if len(glob.glob(os.path.join(opt.ckpt.split('checkpoints')[0], "configs/*.yaml"))) > 0:
            opt.config = sorted(glob.glob(os.path.join(opt.ckpt.split('checkpoints')[0], "configs/*.yaml")))[-1]

    #opt.config = "configs/custom-diffusion/finetune_addtoken.yaml"
    #opt.config = "configs/custom-diffusion/finetune_joint.yaml"
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    config.model.params.run_type='inference'
    print('opop', opt.config)
    if opt.modifier_token is not None:
        config.model.params.cond_stage_config.target = 'src.custom_modules.FrozenCLIPEmbedderWrapper'
        config.model.params.cond_stage_config.params = {}
        config.model.params.cond_stage_config.params.modifier_token = opt.modifier_token
    model = load_model_from_config(config, f"{opt.ckpt}", device = device)
    if opt.use_dci:
        model_ldm = model

    if opt.delta_ckpt is not None:
        delta_st = torch.load(opt.delta_ckpt)
        embed = None
        if 'embed' in delta_st['state_dict']:
            embed = delta_st['state_dict']['embed'].reshape(-1,768)
            del delta_st['state_dict']['embed']
            print(embed.shape)
        delta_st = delta_st['state_dict']
        if opt.compress:
            for name in delta_st.keys():
                if 'to_k' in name or 'to_v' in name:
                    delta_st[name] = model.state_dict()[name] + delta_st[name]['u']@delta_st[name]['v']
            model.load_state_dict(delta_st, strict=False)
        elif "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight" in delta_st.keys():
            model.load_state_dict(delta_st, strict=False)
        elif "transformer.text_model.embeddings.token_embedding.weight" in delta_st.keys():
            token_weights = delta_st["transformer.text_model.embeddings.token_embedding.weight"]
            del delta_st["transformer.text_model.embeddings.token_embedding.weight"]
            delta_load = {}
            for k in delta_st.keys():
                delta_load['model.diffusion_model.'+k] = delta_st[k]   #####
                #del delta_st[k]
            model.load_state_dict(delta_load, strict=False)
            #model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data = token_weights 
            model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[:token_weights.shape[0]] = token_weights 

        if embed is not None:
            print("loading new embedding")
            print(model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data.shape)
            model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[-embed.shape[0]:] = embed
            #model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[-3] = embed[0]

    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if opt.use_dci:
        model_list = [model_ldm, model]
    else:
        model_list = model

    if opt.plms:
        sampler = PLMSSampler(model_list, opt.dci_type)
    else:
        sampler = DDIMSampler(model_list, opt.dci_type)

    if opt.delta_ckpt is not None:
        outpath = os.path.dirname(os.path.dirname(opt.delta_ckpt))
    else:
        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir

    if opt.specific_epoch is not None:
        outpath = os.path.join(outpath, opt.specific_epoch)
        os.makedirs(outpath, exist_ok=True)

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
        if opt.add_prompt is not None:
            data = [batch_size * [prompt.format(opt.add_prompt)]]
        else:
            data = [batch_size * [prompt]]
        if opt.use_dci:
            prompt_ldm = opt.prompt_ldm
            data_ldm = [batch_size * [prompt_ldm]]
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            if opt.add_prompt is not None:
                data = [batch_size * [prompt.format(opt.add_prompt)] for prompt in data]
            else:
                data = [batch_size * [prompt] for prompt in data]

    # outpath = './tmp_output'
    if opt.specific_epoch is not None:
        outpath = os.path.join(outpath, opt.specific_epoch)

    sample_path = os.path.join(outpath, "samples")

    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    if opt.use_dci:
        mask_ldm = opt.mask_ldm.split(',')
        print('mask_ldm', mask_ldm)
        mask_ldm = [int(i) for i in mask_ldm]
        mask_custom = opt.mask_custom.split(',')
        print('mask_custom', mask_custom)
        mask_custom = [int(i) for i in mask_custom]
        if opt.cor_dict is not None:
            opt.cor_dict = json.loads(opt.cor_dict)
        extra_info = {'length_ldm': opt.length_ldm, 'length_custom': opt.length_custom, 'mask_ldm': mask_ldm, 'mask_custom': mask_custom, 'b_edit': opt.b_edit, 'bl_path': opt.bl_path, 'bk_opt':opt.bk_opt, 'cor_dict':opt.cor_dict}
    else:
        extra_info = None

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    images_paths = []
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                prompt_count = 0
                for prompts in tqdm(data, desc="data"):
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        print(prompts[0])
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        print('lll', prompts)
                        c = model.get_learned_conditioning(prompts, index='test')
                        if opt.use_dci:
                            c_ldm = {}
                            c_ldm['ldm'] = model_ldm.get_learned_conditioning(data_ldm[prompt_count])
                        else:
                            c_ldm=None
                       
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         conditioning_ldm=c_ldm,
                                                         extra_info = extra_info,
                                                         layout_info=layout_info,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)
                        # print(samples_ddim.size())
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu()

                        if not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                images_paths.append(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_samples_ddim)

                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))
                        sampling_method = 'plms' if opt.plms else 'ddim'
                        if opt.prompt_ldm is not None:
                            img.save(os.path.join(outpath, f'{prompts[0].replace(" ", "-")}_{opt.prompt_ldm.replace(" ", "-")}_{opt.scale}_{sampling_method}_{opt.ddim_steps}_{opt.ddim_eta}.png'))
                        else:
                            img.save(os.path.join(outpath, f'{prompts[0].replace(" ", "-")}_{opt.scale}_{sampling_method}_{opt.ddim_steps}_{opt.ddim_eta}.png'))

                        if opt.wandb_log:
                            wandb.log({  f'{prompts[0].replace(" ", "-")}_{opt.scale}_{sampling_method}_{opt.ddim_steps}_{opt.ddim_eta}.png'  : [wandb.Image(img)]})
                        grid_count += 1
                
                prompt_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

def clip_score_cal(images_paths, ref_image_paths, prompts, device):
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = []
    ref_image = []
    for image_path in images_paths:
        image.append(preprocess(Image.open(image_path)).unsqueeze(0).to(device))
    for image_path in ref_image_paths:
        ref_image.append(preprocess(Image.open(image_path)).unsqueeze(0).to(device))
    image = torch.cat(image, dim=0)
    ref_images = torch.cat(ref_image, dim=0)
    text = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        ref_image_features = model.encode_image(ref_images)
        text_features = model.encode_text(text)
        image_features = F.normalize(image_features, dim=-1)
        ref_image_features = F.normalize(ref_image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        text_scores_all = image_features @ text_features.T
        text_scores = (image_features @ text_features.T).mean(-1).mean(-1)
        images_scores_all = image_features @ ref_image_features.T
        image_scores = (image_features @ ref_image_features.T).mean(-1).mean(-1)
        logits_per_image, logits_per_text = model(image, text)
    print("text_scores_all:", text_scores_all)
    print("text_scores:", text_scores)
    print("images_scores_all:", images_scores_all)
    print("images_scores:", image_scores)


if __name__ == "__main__":
    main()
