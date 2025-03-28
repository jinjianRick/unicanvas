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

import torch, copy, os
from einops import rearrange, repeat
from torch import nn, einsum

from ldm.models.diffusion.ddpm import LatentDiffusion as LatentDiffusion
from ldm.util import default
from ldm.modules.attention import BasicTransformerBlock as BasicTransformerBlock
from ldm.modules.attention import CrossAttention as CrossAttention
from ldm.util import log_txt_as_img, exists, ismap, isimage, mean_flat, count_params, instantiate_from_config
from torchvision.utils import make_grid
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from pytorch_lightning.utilities.distributed import rank_zero_only

class UniCanvas(LatentDiffusion):
    def __init__(self,
                 freeze_model='crossattn-kv',
                 cond_stage_trainable=False,
                 add_token=False,
                 init_ctx=None,
                 ctx_end_init=None,
                 figure_num=0,
                 class_name='cat',
                 n_ctx=0,
                 n_end_ctx=0,
                 add_class=True,
                 per_token=True,
                 end_token=True,
                 task_token=2,
                 run_type='fine_tune',      # fine_tune or inference
                 layout_sup=False,
                 *args, **kwargs):

        self.add_class = add_class
        self.per_token = per_token
        self.end_token = end_token

        self.task_token = task_token

        self.freeze_model = freeze_model
        self.add_token = add_token
        self.cond_stage_trainable = cond_stage_trainable
        self.init_ctx = init_ctx
        self.figure_num = figure_num
        self.n_ctx = n_ctx
        self.n_end_ctx = n_end_ctx
        self.class_name = class_name
        self.ctx_end_init = ctx_end_init
        self.run_type = run_type
        self.layout_sup = layout_sup
        super().__init__(cond_stage_trainable=cond_stage_trainable, *args, **kwargs)

        if self.freeze_model == 'crossattn-kv':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' not in x[0]:
                    x[1].requires_grad = False
                elif not ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0]):
                    x[1].requires_grad = False
                else:
                    x[1].requires_grad = True
        elif self.freeze_model == 'crossattn':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' not in x[0]:
                    x[1].requires_grad = False
                elif not 'attn2' in x[0]:
                    x[1].requires_grad = False
                else:
                    x[1].requires_grad = True

        def change_checkpoint(model):
            for layer in model.children():
                if type(layer) == BasicTransformerBlock:
                    layer.checkpoint = False
                else:
                    change_checkpoint(layer)

        change_checkpoint(self.model.diffusion_model)

        def new_forward(self, x, context=None, mask=None, prompt_dict=None, info_dict=None, latent=None, layer_info=None, layout_info=None):
            h = self.heads
            crossattn = False
            if context is not None:
                crossattn = True
            q = self.to_q(x)

            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            if crossattn:
                modifier = torch.ones_like(k)
                modifier[:, :1, :] = modifier[:, :1, :]*0.
                k = modifier*k + (1-modifier)*k.detach()
                v = modifier*v + (1-modifier)*v.detach()

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            if layout_info is not None:

                if crossattn:
                    treg = torch.pow(layout_info['timesteps']/1000, 5).unsqueeze(-1).repeat(1, self.heads).view(-1) .unsqueeze(-1).unsqueeze(-1).to(sim.device)
                    mask = layout_info['creg_maps'][sim.size(1)].repeat(1, self.heads,1,1).view(-1, sim.shape[-2], sim.shape[-1])
                    if layout_info['timesteps'][0].item()/1000 > 0.7:
                        creg = layout_info['creg']
                        if prompt_dict['run_type']=='inference':
                            if info_dict is not None: 
                                min_value = sim[int(sim.size(0)/4):int(sim.size(0)/2)].min(-1)[0].unsqueeze(-1)
                                max_value = sim[int(sim.size(0)/4):int(sim.size(0)/2)].max(-1)[0].unsqueeze(-1) 
                            else:
                                min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1)
                                max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1) 
                        else:
                            min_value = sim.min(-1)[0].unsqueeze(-1)
                            max_value = sim.max(-1)[0].unsqueeze(-1) 

                        size_reg = layout_info['size_reg'][sim.size(1)].clone().detach().unsqueeze(1).repeat(1, self.heads,1,sim.shape[-1]).view(-1,sim.shape[-2],sim.shape[-1]).to(sim.device)
                        
                        if prompt_dict['run_type']=='inference':
                            if info_dict is not None:
                                sim[int(sim.size(0)/4):int(sim.size(0)/2)] += (mask>0)*size_reg*creg*treg*(max_value-sim[int(sim.size(0)/4):int(sim.size(0)/2)])
                                sim[int(sim.size(0)/4):int(sim.size(0)/2)] -= ~(mask>0)*size_reg*creg*treg*(sim[int(sim.size(0)/4):int(sim.size(0)/2)]-min_value)

                            else:
                                sim[int(sim.size(0)/2):] = sim[int(sim.size(0)/2):]*mask + ~(mask>0)*(-1e2)
                        else:
                            creg = creg.unsqueeze(-1).repeat(1, self.heads).view(-1).unsqueeze(-1).unsqueeze(-1)

                            sim += (mask>0)*size_reg*creg*treg*(max_value-sim)
                            sim -= ~(mask>0)*size_reg*creg*treg*(sim-min_value)
        
                else:
                    pass

            if info_dict is not None and 'bk_opt' in info_dict:
                sim = rearrange(sim, '(b h) n d -> b n (h d)', h=h)
                attn_tmp = sim[-int((sim.shape[0])/4):]
                attn_tmp = rearrange(attn_tmp, 'b n (h d) -> (b h) n d', h=h)

                if info_dict['bk_opt'] == 'use':
                    if layout_info['timesteps'][0].item()/1000 > 0.7:
                        cor_dict = info_dict['cor_dict']
                        for key in cor_dict.keys():
                            attn_tmp[:, :, int(key)] = info_dict['bk_attn'][layer_info['layer_type']+layer_info['layer_count']][:, :, cor_dict[key]].repeat(int(sim.shape[0]/4),1)

                        attn_tmp = rearrange(attn_tmp, '(b h) n d -> b n (h d)', h=h)
                        sim[-int((sim.shape[0])/4):] = attn_tmp

                elif info_dict['bk_opt'] == 'get':
                    info_dict['bk_attn'][layer_info['layer_type']+layer_info['layer_count']] = attn_tmp[0:h, :, 0:1+info_dict['length_ldm']].detach().clone()
                
                sim = rearrange(sim, 'b n (h d) -> (b h) n d', h=h)
                

            attn = sim.softmax(dim=-1)  # [96, 1024, 77]

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

            if crossattn and prompt_dict['run_type']=='inference' and info_dict is not None:

                if not info_dict['sep_custom']:
                    #print('One !')
                    attn = rearrange(attn, '(b h) n d -> b n (h d)', h=h)
                    attn_uncond1 = attn[:int((out.shape[0])/4)]
                    attn_cond1 = attn[int((out.shape[0])/4):-int((out.shape[0])/2)]
                    attn_uncond2 = attn[int((out.shape[0])/2):-int((out.shape[0])/4)]
                    attn_cond2 = attn[-int((out.shape[0])/4):]

                    attn_cond1 = rearrange(attn_cond1, 'b n (h d) -> (b h) n d', h=h)
                    attn_cond2 = rearrange(attn_cond2, 'b n (h d) -> (b h) n d', h=h)

                    atten_list1 = []
                    atten_list2 = []
                    attn_cond1 = (10*attn_cond1[:, :, 1:1+info_dict['length_custom']]).softmax(dim=-1)
                    attn_cond2 = (10*attn_cond2[:, :, 1:1+info_dict['length_ldm']]).softmax(dim=-1)

                    for index in info_dict['mask_custom']:
                        atten_list1.append(attn_cond1[:, :, index].unsqueeze(-1))
                    attn1 = torch.cat(atten_list1, dim=2).sum(dim=2)

                    for index  in info_dict['mask_ldm']:
                        atten_list2.append(attn_cond2[:, :, index].unsqueeze(-1))
                    attn2 = torch.cat(atten_list2, dim=2).sum(dim=2)

                    attn2 = attn_cond2.sum(dim=2)-attn2

                    attn1 = 500*(10*attn1).softmax(-1)
                    attn2 = 500*(10*attn2).softmax(-1)

                    mask1 = (1-mask[0,:, 1].unsqueeze(0))*(-1e6) + mask[0,:, 1].unsqueeze(0)    ####
                    mask2 = (1-mask[0,:, info_dict['length_custom']].unsqueeze(0))*(-1e6) + mask[0,:, info_dict['length_custom']].unsqueeze(0)
                    layout_mask = torch.cat([mask1, mask2])

                    attn = torch.cat([(attn1.unsqueeze(dim=0)), -5*(attn2.unsqueeze(dim=0))], dim=0).softmax(dim=0)

                    out_uncond1 = out[:int((out.shape[0])/4)]
                    out1 = out[int((out.shape[0])/4):-int((out.shape[0])/2)]
                    out_uncond2 = out[int((out.shape[0])/2):-int((out.shape[0])/4)]
                    out2 = out[-int((out.shape[0])/4):]

                    out1 = rearrange(out1, 'b n (h d) -> (b h) n d', h=h)
                    out2 = rearrange(out2, 'b n (h d) -> (b h) n d', h=h)
                    
                    b_mask1 = mask[0,:, 1].unsqueeze(0)    ####
                    b_mask2 = mask[0,:, info_dict['length_custom']].unsqueeze(0)


                    if str(layer_info['layer_type']+layer_info['layer_count']) == "out11" and False:
                        out1 = attn[0].unsqueeze(-1)*out1 + attn[1].unsqueeze(-1)*out2
                        out1 = out1*b_mask1.unsqueeze(-1) + out2*b_mask2.unsqueeze(-1)

                    out1 = attn[0].unsqueeze(-1)*out1 + attn[1].unsqueeze(-1)*out2
                    out1 = out1*b_mask1.unsqueeze(-1) + out2*b_mask2.unsqueeze(-1)
                    out1 = rearrange(out1, '(b h) n d -> b n (h d)', h=h)
                    out2 = rearrange(out2, '(b h) n d -> b n (h d)', h=h)

                    out = torch.cat([out_uncond1, out1, out_uncond2, out2])

            return self.to_out(out)

        def change_forward(model):
            for layer in model.children():
                if type(layer) == CrossAttention:
                    bound_method = new_forward.__get__(layer, layer.__class__)
                    setattr(layer, 'forward', bound_method)
                else:
                    change_forward(layer)

        change_forward(self.model.diffusion_model)

        self.re_weight = False
        reweight = 2
        self.weight_tuple = [(7,reweight), (8,reweight), (9,reweight), (10,reweight), (11,reweight), (12,reweight), (13,reweight), (14,reweight), (15,reweight) ] #leaf
        self.weight_tuple = [(7,reweight), (8,reweight), (9,reweight), (10,reweight), (11,reweight), (12,reweight), (13,reweight)] #leaf
        self.weight_tuple = [(7,reweight), (8,reweight), (9,reweight), (10,reweight), (11,reweight)]

        if self.cond_stage_config.target == "ldm.modules.encoders.modules.BERTEmbedder":
            ctx_init = "Photo of"
            figure_num = image_number
            ctx_dim = 1280
            n_ctx = 2
            dtype = torch.float32


            if ctx_init:
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = len(ctx_init.split(" "))
                prompt = self.cond_stage_model.tknz_fn(ctx_init).to(self.device)
                with torch.no_grad():
                    embedding = self.cond_stage_model.transformer.token_emb(prompt).type(dtype)
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
                ctx_vectors = ctx_vectors.repeat(figure_num, 1, 1)
                prompt_prefix = ctx_init
            else:
                # random initialization
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(figure_num, n_ctx, ctx_dim, dtype=dtype)

                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)

            print(f'Initial context: "{prompt_prefix}"')
            print(f"Number of context words (tokens): {n_ctx}")

            self.ctx = nn.Parameter(ctx_vectors, requires_grad=True)

            if self.add_class:
                prompts = prompt_prefix + " " + "*" + " " + self.class_name
            else:
                prompts = prompt_prefix + " " + "*"
            print('aaa', prompts)
            tokenized_prompts = self.cond_stage_model.tknz_fn(prompts).to(self.device)

            with torch.no_grad():
                embedding = self.cond_stage_model.transformer.token_emb(tokenized_prompts).type(dtype)

            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 2 + n_ctx :, :]) 

        elif self.cond_stage_config.target == "src.custom_modules.FrozenCLIPEmbedderWrapper":
            version="openai/clip-vit-large-patch14"

            self.tokenizer = CLIPTokenizer.from_pretrained(version)
            self.transformer = CLIPTextModel.from_pretrained(version)

            #ctx_init = "Photo of"
            ctx_init = self.init_ctx
            figure_num =  self.figure_num
            ctx_dim = 768
            n_ctx = self.n_ctx
            dtype = torch.float32
            max_length = 77

            if self.end_token:
                ctx_end_init = self.ctx_end_init
                n_end_ctx = self.n_end_ctx

                self.name_lens = len(self.tokenizer.encode(self.class_name)) - 2

            if ctx_init:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                n_ctx = len(ctx_init.split(" "))
                prompt = self.tokenizer(ctx_init, truncation=True, max_length=max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")["input_ids"].to(self.device)
                
                with torch.no_grad():
                    embedding = self.transformer.text_model.embeddings(input_ids=prompt).type(dtype)
                    #embedding = self.transformer.text_model.embeddings.token_embedding(prompt).type(dtype)
                    
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
                ctx_vectors = ctx_vectors.repeat(figure_num, 1, 1)
                prompt_prefix = ctx_init

                if self.end_token:
                    ctx_end_init = ctx_end_init.replace("_", " ")
                    n_end_ctx = len(ctx_end_init.split(" "))
                    prompt_end = self.tokenizer(ctx_end_init, truncation=True, max_length=max_length, return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")["input_ids"].to(self.device)
                    
                    with torch.no_grad():
                        embedding_end = self.transformer.text_model.embeddings(input_ids=prompt_end).type(dtype)
                        #embedding = self.transformer.text_model.embeddings.token_embedding(prompt).type(dtype)
                        
                    ctx_end_vectors = embedding_end[0, 1 : 1 + n_end_ctx, :]
                    ctx_end_vectors = ctx_end_vectors.repeat(figure_num, 1, 1)
                    prompt_suffix = ctx_end_init

            else:
                # random initialization
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(figure_num, n_ctx, ctx_dim, dtype=dtype)

                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)

                if self.end_token:
                    ctx_end_vectors = torch.empty(figure_num, n_end_ctx, ctx_dim, dtype=dtype)

                    nn.init.normal_(ctx_end_vectors, std=0.02)
                    prompt_suffix = " ".join(["X"] * n_end_ctx)

            print(f'Initial context: "{prompt_prefix}"')
            print(f"Number of context words (tokens): {n_ctx}")

            self.ctx = nn.Parameter(ctx_vectors, requires_grad=True)

            if self.end_token:
                self.end_ctx = nn.Parameter(ctx_end_vectors, requires_grad=True)

            if self.add_class:
                prompts = prompt_prefix + " " + "* "*self.task_token + self.class_name

                if self.end_token:
                    prompts = prompt_prefix + " " + "* "*self.task_token + self.class_name + " " + prompt_suffix
            else:
                prompts = prompt_prefix + " *"*self.task_token

                if self.end_token:
                    prompts = prompt_prefix + " " + "* "*self.task_token + prompt_suffix
            
            print('prompt', prompts)
            tokenized_prompts = self.tokenizer(prompts, truncation=True, max_length=max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")["input_ids"].to(self.device)

            with torch.no_grad():
                embedding = self.transformer.text_model.embeddings(input_ids=tokenized_prompts).type(dtype)
                #embedding = self.transformer.text_model.embeddings.token_embedding(tokenized_prompts).type(dtype)

            if not self.end_token:
                self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
                self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + self.task_token :, :]) 
            else:
                self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
                self.register_buffer('class_name_embedding', embedding[:, 1 + self.task_token + n_ctx : 1 + self.task_token + n_ctx + self.name_lens, :])
                self.register_buffer("token_suffix", embedding[:, 1 + self.task_token + n_ctx +self.name_lens + n_end_ctx:, :])

        self.prompt_dict = {}
        if self.per_token:
            self.prompt_dict['ctx'] = (self.ctx)
            self.prompt_dict['token_prefix'] = (self.token_prefix).squeeze(0)
            self.prompt_dict['token_suffix'] = (self.token_suffix).squeeze(0)
            if self.end_token:
                self.prompt_dict['end_ctx'] = (self.end_ctx)
                self.prompt_dict['class_name_embedding'] = self.class_name_embedding.squeeze(0)
        
        if self.add_class:
            self.prompt_dict['class_name'] = self.class_name

        if self.re_weight:
            self.prompt_dict['weight_tuple'] = self.weight_tuple

        self.prompt_dict['run_type'] = self.run_type
        self.prompt_dict['per_token'] = self.per_token
        self.prompt_dict['add_class'] = self.add_class
        self.prompt_dict['end_token'] = self.end_token
        self.prompt_dict['task_token'] = self.task_token
        self.prompt_dict['re_weight'] = self.re_weight

    def get_learned_conditioning(self, c, index = None):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                if index==None:
                    c = self.cond_stage_model.encode(c, device = self.device)
                else:
                    self.prompt_dict['state'] = 'train'
                    if index == 'test': 
                        self.prompt_dict['per_token'] = False
                        self.prompt_dict['state'] = 'test'

                    if self.per_token and index != 'test':
                        self.prompt_dict['ctx'] = self.prompt_dict['ctx'].to(self.device)
                        self.prompt_dict['token_prefix'] = self.prompt_dict['token_prefix'].to(self.device)
                        self.prompt_dict['token_suffix'] = self.prompt_dict['token_suffix'].to(self.device)
                        if self.end_token:
                            self.prompt_dict['end_ctx'] = self.prompt_dict['end_ctx'].to(self.device)
                            self.prompt_dict['class_name_embedding'] = self.prompt_dict['class_name_embedding'].to(self.device)

                    c = self.cond_stage_model.encode(c, prompt_dict=self.prompt_dict, device = self.device, index=index)
                    
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)

        return c

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        if self.freeze_model == 'crossattn-kv':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' in x[0]:
                    if 'attn2.to_k' in x[0] or 'attn2.to_v' in x[0]:
                        params += [x[1]]
                        print(x[0])
        elif self.freeze_model == 'crossattn':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' in x[0]:
                    if 'attn2' in x[0]:
                        params += [x[1]]
                        print(x[0])
        else:
            params = list(self.model.parameters())

        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            if self.add_token:
                params = params + list(self.cond_stage_model.transformer.text_model.embeddings.token_embedding.parameters())
            else:
                params = params + list(self.cond_stage_model.parameters())

        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        
        if self.per_token:
            if not self.end_token:
                opt = torch.optim.AdamW([{"params": params, "lr": lr}, {"params": self.ctx}], lr=lr)
            else:
                print('opoopopkkk')
                opt = torch.optim.AdamW([{"params": params, "lr": lr}, {"params": self.ctx, "lr": 45*lr}, {"params": self.end_ctx, "lr": 45*lr}])
        else:
            opt = torch.optim.AdamW(params, lr=lr)

        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    def p_losses(self, x_start, cond, t, mask=None, noise=None, layout_info=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond, layout_info=layout_info)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False)
        if mask is not None:
            loss_simple = (loss_simple*mask).sum([1, 2, 3])/mask.sum([1, 2, 3])
        else:
            loss_simple = loss_simple.mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = (self.logvar.to(self.device))[t]
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False)
        if mask is not None:
            loss_vlb = (loss_vlb*mask).sum([1, 2, 3])/mask.sum([1, 2, 3])
        else:
            loss_vlb = loss_vlb.mean([1, 2, 3])
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def get_input_withmask(self, batch, **args):
        out = super().get_input(batch, self.first_stage_key, **args)
        mask = batch["mask"]
        if len(mask.shape) == 3:
            mask = mask[..., None]
        mask = rearrange(mask, 'b h w c -> b c h w')
        mask = mask.to(memory_format=torch.contiguous_format).float()
        out += [mask]
        return out

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            train_batch = batch[0]
            train2_batch = batch[1]
            loss_train, loss_dict = self.shared_step(train_batch)
            loss_train2, _ = self.shared_step(train2_batch)
            loss = loss_train + loss_train2
        else:
            train_batch = batch
            loss, loss_dict = self.shared_step(train_batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    """ def shared_step(self, batch, **kwargs):
        x, c, mask = self.get_input_withmask(batch, **kwargs)
        loss = self(x, c, mask=mask)
        return loss """

    def shared_step(self, batch, **kwargs):
        x, c, mask = self.get_input_withmask(batch, **kwargs)
        #x, c = self.get_input(batch, self.first_stage_key)
        if self.per_token:
            index = batch['index'].to(self.device)
            self.prompt_dict['c_type'] = batch['c_type']
            if self.layout_sup:
                layout_info = {}
                layout_info['creg_maps'] = batch['creg_maps']
                for key in layout_info['creg_maps'].keys():
                    layout_info['creg_maps'][key] = layout_info['creg_maps'][key].to(self.device)
                layout_info['size_reg'] = batch['size_reg']
                layout_info['c_size_reg'] = batch['c_size_reg']
                layout_info['creg'] = batch['creg']
                #print('iop', layout_info)
                loss = self(x, c, mask=mask, index=index, layout_info=layout_info)
            else:
                loss = self(x, c, mask=mask, index=index)
        else:
            loss = self(x, c, mask=mask)
        return loss
    
    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if 'layout_info' in  kwargs.keys():
            kwargs['layout_info']['timesteps'] = t
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                if 'index' in kwargs.keys():
                    c = self.get_learned_conditioning(c, kwargs['index'])      ###
                    del kwargs['index']
                else:
                    c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

        return self.p_losses(x, c, t, *args, **kwargs)

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        if isinstance(batch, list):
            batch = batch[0]
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                unconditional_guidance_scale=6.
                unconditional_conditioning = self.get_learned_conditioning(len(c) * [""])
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta,
                                                        unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples_scaled"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
    
    def apply_model(self, x_noisy, t, cond, info_dict=None, latent=None, layout_info=None, return_ids=False):
        #print('ff', layout_info)
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids  
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization
        else:
            cond['prompt_dict'] = self.prompt_dict
            cond['info_dict'] = info_dict
            cond['latent'] = latent
            cond['layout_info'] = layout_info
            
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        params = self.cond_stage_model.state_dict()

        params = {}

        for key_val in self.model.diffusion_model.named_parameters():
            if 'transformer_blocks' in key_val[0]:
                    if 'attn2.to_k' in key_val[0] or 'attn2.to_v' in key_val[0]:
                        params[key_val[0]] = key_val[1]
        
        params['transformer.text_model.embeddings.token_embedding.weight']=self.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data

        checkpoint = {
            "state_dict": params,
        }
        torch.save(checkpoint, os.path.join(self.trainer.checkpoint_callback.dirpath, f"embeddings_gs-{self.global_step}.ckpt"))
