import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
import math
'''
mmsam2.py
功能：定义MMSAM2模型架构。
主要作用：
1. 构建基于SAM2的基础模型。
2. 定义卷积模块（DoubleConv）和上采样模块（Up），用于构建改进的解码器或特征融合网络。
3. 整合编码器和解码器部分，形成完整的图像分割/预测网络。
'''
from sam2.modeling.backbones.MFB import MFB_modified

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""
    # todo 可以将in_channels 参数删除，因为是可以推断的关系
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 进行创新
class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

# 1、提取bak_function.py中的函数
# 2、使用 memory_attention sam_prompt_encoder sam_mask_decoder _encode_new_memory 等函数
# 3、将net 作为参数传入memory_forward 初始化中
# 4、bak_function.train_sam中代码逻辑封装进入memory_forward 中

class DynamicMemoryBank():
    def __init__(self, max_size=8, min_size=4, similarity_threshold=0.85, decay_factor=0.98):
        self.memories = []
        self.max_size = max_size
        self.min_size = min_size
        self.similarity_threshold = similarity_threshold
        self.decay_factor = decay_factor
        self.usage_counts = []
        self.timestamps = []
        self.current_time = 0

        
    def update(self, new_feature, pos_enc, iou_score, image_embed, current_mm=None):
        """
        Update memory bank with new feature, using similarity and IoU-based strategies.
        
        Args:
            new_feature: The new feature to possibly add to memory
            pos_enc: Position encoding for the feature
            iou_score: IoU score of the feature's mask prediction
            image_embed: Global image embedding for similarity comparison
            current_mm: Optional multi-scale context feature for blending
        """
        self.current_time += 1
        
        # If empty or not enough memories yet, just add
        if len(self.memories) < self.min_size:
            self._add_memory(new_feature, pos_enc, iou_score, image_embed)
            return
            
        # Compute similarities with existing memories
        memory_features = [m[0].reshape(-1) for m in self.memories]
        memory_features = torch.stack(memory_features)
        new_feature_flat = new_feature.reshape(-1)
        
        # Normalize for cosine simi larity
        memory_norm = F.normalize(memory_features, p=2, dim=1)
        new_feature_flat = new_feature_flat.to(dtype=memory_norm.dtype)
        new_norm = F.normalize(new_feature_flat, p=2, dim=0).unsqueeze(1)
        
        # Calculate similarity scores between the new feature and all memories
        similarity_scores = torch.mm(memory_norm, new_norm).squeeze() # 新特征与所有记忆的相似度
        
        # Calculate similarity matrix between all existing memories
        similarity_matrix = torch.mm(memory_norm, memory_norm.t()) # 所有记忆之间的相似度
        
        # Create version with diagonal set to -inf for finding max similarities
        similarity_matrix_no_diag = similarity_matrix.clone()
        diag_indices = torch.arange(similarity_matrix_no_diag.size(0))
        similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')
        
        # STRATEGY 1: If very similar to existing memory, merge instead of add
        max_sim, max_idx = torch.max(similarity_scores, dim=0)
        if max_sim > self.similarity_threshold:
            # Update existing memory with weighted average
            alpha = self.adaptive_blend_factor(iou_score, self.memories[max_idx][2], 
                                        self.usage_counts[max_idx])
            
            # If multi-scale context is provided, use blended feature
            if current_mm is not None:
                blend_factor = 0.5  # Could be made a parameter or adaptive
                push_feature = blend_factor * new_feature + (1 - blend_factor) * current_mm
            else:
                push_feature = new_feature
                
            self.memories[max_idx][0] = alpha * push_feature + (1 - alpha) * self.memories[max_idx][0]
            self.memories[max_idx][2] = max(iou_score, self.memories[max_idx][2])  # Take best IoU
            self.usage_counts[max_idx] += 1
            self.timestamps[max_idx] = self.current_time
            return
                
        # STRATEGY 2: Find the memory with minimum similarity to the new feature
        min_similarity_index = torch.argmin(similarity_scores)
        
        # Find the memory that has highest similarity to this minimum-similar memory
        max_similarity_index = torch.argmax(similarity_matrix_no_diag[min_similarity_index])
        
        # If new feature is less similar to min_similarity_index than max_similarity_index
        # This means we're replacing a memory that already has a similar representation
        if similarity_scores[min_similarity_index] < similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
            # Only if new IoU is better (with tolerance)
            if iou_score > self.memories[max_similarity_index][2] - 0.1:
                # Create a blended feature if multi-scale context available
                if current_mm is not None:
                    blend_factor = 0.5  # Could be self.scale_factor
                    push_feature = blend_factor * new_feature + (1 - blend_factor) * current_mm
                else:
                    push_feature = new_feature
                    
                # Remove the redundant memory
                self.memories.pop(max_similarity_index)
                self.usage_counts.pop(max_similarity_index)
                self.timestamps.pop(max_similarity_index)
                
                # Add the new memory
                self._add_memory(push_feature, pos_enc, iou_score, image_embed)
                return
        
        # STRATEGY 3: If needs to add but no clear replacement based on similarity
        if len(self.memories) >= self.max_size:
            # Calculate removal score based on uniqueness, usage and age
            removal_scores = self._calculate_removal_scores(similarity_scores)
            remove_idx = torch.argmax(removal_scores).item()
            
            # Remove the memory with highest removal score
            self.memories.pop(remove_idx)
            self.usage_counts.pop(remove_idx)
            self.timestamps.pop(remove_idx)
        
        # Blend with multi-scale context if available
        if current_mm is not None:
            blend_factor = 0.5  # Could be self.scale_factor
            push_feature = blend_factor * new_feature + (1 - blend_factor) * current_mm
        else:
            push_feature = new_feature
            
        # Add the new memory
        self._add_memory(push_feature, pos_enc, iou_score, image_embed)  
    
    #  暂时不用
    def update_similarity(self, new_feature, pos_enc, iou_score, image_embed):
        self.current_time += 1
        
        # If empty, just add
        if len(self.memories) < self.min_size:
            self._add_memory(new_feature, pos_enc, iou_score, image_embed)
            return
            
        # Compute similarities with existing memories
        memory_features = [m[0].reshape(-1) for m in self.memories]
        memory_features = torch.stack(memory_features)
        new_feature_flat = new_feature.reshape(-1)
        
        # Normalize for cosine similarity
        memory_norm = F.normalize(memory_features, p=2, dim=1)
            # Ensure consistent data type before normalization
        new_feature_flat = new_feature_flat.to(dtype=memory_norm.dtype)
        new_norm = F.normalize(new_feature_flat, p=2, dim=0).unsqueeze(1)
        
        # Calculate similarity scores
        similarities = torch.mm(memory_norm, new_norm).squeeze()
        
        # If very similar to existing memory, merge instead of add
        max_sim, max_idx = torch.max(similarities, dim=0)
        if max_sim > self.similarity_threshold:
            # Update existing memory with weighted average
            alpha = self.adaptive_blend_factor(iou_score, self.memories[max_idx][2], 
                                           self.usage_counts[max_idx])
            self.memories[max_idx][0] = alpha * new_feature + (1 - alpha) * self.memories[max_idx][0]
            self.memories[max_idx][2] = max(iou_score, self.memories[max_idx][2])  # Take best IoU
            self.usage_counts[max_idx] += 1
            self.timestamps[max_idx] = self.current_time
            return
            
        # Need to add new memory - check if we need to remove one
        if len(self.memories) >= self.max_size:
            # Calculate removal score based on:
            # 1. Low similarity to other memories (unique)
            # 2. Low usage count (not frequently needed)
            # 3. Old (not recently accessed)
            removal_scores = self._calculate_removal_scores(similarities)
            remove_idx = torch.argmax(removal_scores).item()
            
            # Remove the memory with highest removal score
            self.memories.pop(remove_idx)
            self.usage_counts.pop(remove_idx)
            self.timestamps.pop(remove_idx)
            
        # Add the new memory
        self._add_memory(new_feature, pos_enc, iou_score, image_embed)
        
    def _add_memory(self, feature, pos_enc, iou, embed):
        self.memories.append([feature.detach(), pos_enc.detach(), iou, embed.detach()])
        self.usage_counts.append(1)
        self.timestamps.append(self.current_time)
        
    def _calculate_removal_scores(self, current_similarities):

        memory_matrix = torch.stack([m[0].reshape(-1) for m in self.memories])
        memory_norm = F.normalize(memory_matrix, p=2, dim=1)
        
        # Calculate similarity matrix between all memories
        sim_matrix = torch.mm(memory_norm, memory_norm.t())
        
        # Device to use for all tensors
        device = sim_matrix.device
        
        # Calculate uniqueness (low similarity to other memories is good to keep)
        sim_matrix.fill_diagonal_(0)  # Remove self-similarity
        uniqueness = 1 - torch.mean(sim_matrix, dim=1)
        
        # Age factor (newer is better to keep)
        max_time = float(self.current_time)
        age_factor = torch.tensor([(max_time - t) / max_time for t in self.timestamps], device=device)
        
        # Usage factor (more used is better to keep)
        usage = torch.tensor([1.0 / (c + 1) for c in self.usage_counts], device=device)
        
        # Combine factors - higher score means more likely to remove
        removal_scores = 0.6 * uniqueness + 0.2 * age_factor + 0.2 * usage
        return removal_scores

        
    def adaptive_blend_factor(self, new_iou, old_iou, usage_count):
        # # More useful (higher IoU) and less used memories get updated more
        # iou_factor = torch.sigmoid(torch.tensor(new_iou - old_iou + 0.1)) * 0.5 + 0.25
        # usage_factor = 1.0 / (1 + math.log(1 + usage_count))
        # return float(iou_factor * usage_factor + 0.3)  # Ensure some minimum update
            # More useful (higher IoU) and less used memories get updated more
        # diff = 0.1
        if isinstance(new_iou, torch.Tensor) or isinstance(old_iou, torch.Tensor):
            # Handle tensor inputs properly
            if isinstance(new_iou, torch.Tensor):
                new_iou_val = new_iou.item()
            else:
                new_iou_val = new_iou
                
            if isinstance(old_iou, torch.Tensor):
                old_iou_val = old_iou.item()
            else:
                old_iou_val = old_iou
                
            diff_val = new_iou_val - old_iou_val + 0.1
            # Create tensor on the same device as input tensors if they're tensors
            device = new_iou.device if isinstance(new_iou, torch.Tensor) else (
                old_iou.device if isinstance(old_iou, torch.Tensor) else None)
            diff_tensor = torch.tensor(diff_val, device=device)
        else:
            # Handle scalar inputs
            diff_tensor = torch.tensor(new_iou - old_iou + 0.1)
        
        iou_factor = torch.sigmoid(diff_tensor) * 0.5 + 0.25
        usage_factor = 1.0 / (1 + math.log(1 + usage_count))
        return float(iou_factor * usage_factor + 0.3)  # Ensure some minimum update
    
    # 提取最相似的记忆   
    def retrieve(self, query_embedding, top_k=None):
        if not self.memories:
            return None, None

        # Normalize query
        query_norm = F.normalize(query_embedding, p=2, dim=0)
        
        # Get all memory embeddings 
        memory_embeds = torch.stack([m[3] for m in self.memories])
        memory_norm = F.normalize(memory_embeds, p=2, dim=1)
        
        # Calculate similarity
        similarities = torch.mm(memory_norm, query_norm.unsqueeze(1)).squeeze()
        

           # Handle case when there's only one memory (similarities becomes 0-d tensor)
        if similarities.dim() == 0:
            similarities = similarities.unsqueeze(0)
        # Get most similar memories
        if top_k is None:
            top_k = len(self.memories)
        
        _, indices = torch.topk(similarities, min(top_k, len(self.memories)))


            # Handle case when indices is 0-d tensor (single memory)
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        
        # Update usage counts for retrieved memories
        for idx in indices:
            self.usage_counts[idx] += 1
            self.timestamps[idx] = self.current_time
            
        return [self.memories[i] for i in indices], similarities[indices]
    
class MMSAM2(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(MMSAM2, self).__init__()    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            self.model = build_sam2(model_cfg, checkpoint_path)
        else:
            self.model = build_sam2(model_cfg)

        del self.model.mask_downsample
        del self.model.obj_ptr_tpos_proj
        del self.model.obj_ptr_proj
      
        # self.image_encoder = self.model.image_encoder

        # 动态记忆库
        self.memory_bank = DynamicMemoryBank(max_size=12, min_size=6)
        self.down_2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.down_4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.scale_factor = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # 添加多尺度特征融合的卷积层
        # 注意：这里使用固定的通道数，基于 SAM2 的默认配置
        hidden_dim = 256  # 从 model.hidden_dim 获取，通常是 256
        self.conv_s2 = nn.Conv2d(64, hidden_dim, kernel_size=1, stride=1)  # 中间尺度
        self.conv_s3 = nn.Conv2d(32, hidden_dim, kernel_size=1, stride=1)  # 大尺度
        
        # MFB 特征增强模块
        # 基于 SAM2 memory encoder 的输出通道数，通常是 64
        memory_dim = 64
        self.feat_small = MFB_modified(memory_dim, memory_dim)
        # self.feat_mid = MFB_modified(memory_dim, memory_dim)  
        # self.feat_large = MFB_modified(memory_dim, memory_dim)


        # - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # freze the encoder
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
        for param in self.model.image_encoder.neck.mrb_convs.parameters():
            param.requires_grad = True

        blocks = []
        for block in self.model.image_encoder.trunk.blocks:
            blocks.append(
                Adapter(block)
            )
        self.model.image_encoder.trunk.blocks = nn.Sequential(
            *blocks
        )
        # for param in self.model.image_encoder.trunk.blocks.parameters():
        #     param.requires_grad = True
        self.up1 = (Up(512, 256))#  与 yaml 中 d_model 相对应 128 = 64*2 
        self.up2 = (Up(512, 256))
        self.up3 = (Up(512, 256))
        self.up4 = (Up(512, 256))

        #todo 优化 128=>1 很突兀，建议降的速度慢一些
        self.side1 = nn.Conv2d(256, 1, kernel_size=1)
        self.side2 = nn.Conv2d(256, 1, kernel_size=1)
        self.head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x , click):
        # backbone_out = self.image_encoder(x)
        backbone_out = self.model.forward_image(x) # net.forward_image(imgs)
        _, vision_feats, vision_pos_embeds, _ = self.model._prepare_backbone_features(backbone_out)
        # self.model._prepare_backbone_features 特征全局视野权重计算
        #  在 _run_single_frame_inference->_get_image_feature 中 
        # vision_feats 
        # torch.Size([7744, 3, 32]) 88
        # torch.Size([1936, 3, 64]) 44
        # torch.Size([484, 3, 256]) 22
        _,_,y3 = backbone_out["vision_features"],backbone_out["vision_pos_enc"],backbone_out["backbone_fpn_ori"] #neck 暂时没啥用只是将连个多尺度融合然后平均，那么输出呢，会不会增加准确率
        x1, x2, x3, x4 = y3[0], y3[1], y3[2], y3[3]
        # torch.Size([12, 256, 88, 88]) x1
        # torch.Size([12, 256, 44, 44]) x2
        # torch.Size([12, 256, 22, 22]) x3
        # torch.Size([12, 256, 11, 11]) x4

        B = vision_feats[-1].size(1)  # batch size 
        if len(self.memory_bank.memories) == 0: # 不用memory bank
            vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, self.model.hidden_dim)).to(device="cuda")
            vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, self.model.hidden_dim)).to(device="cuda")
            # vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, 128)).to(device="cuda")
            # vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, 128)).to(device="cuda")
        else:
                to_cat_memory_dynamic = []
                to_cat_memory_pos_dynamic = []
                to_cat_image_embed_dynamic = []
                vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 22, 22) 
                vision_feats_temp = vision_feats_temp.reshape(B, -1)
                #  -----------------------
                for b in range(B):# 每个批次的当前特征都需要进行更新
                        # Retrieve relevant memories for this batch item
                        # retrieved_memories 取回的是topK相似的记忆库
                        # 其中len(retrieved_memories[i]) ==  4 => 是 push 进去 4个元素
                        retrieved_memories, similarities = self.memory_bank.retrieve(F.normalize(vision_feats_temp[b], p=2, dim=0),top_k=4)
                        if retrieved_memories:
                            # Apply attention weights based on similarity
                            weights = F.softmax(similarities, dim=0)
                            for i, (memory, pos_enc, _, image_emd) in enumerate(retrieved_memories):
                                # Add weighted memory features
                                to_cat_memory_dynamic.append(memory.cuda(non_blocking=True).flatten(2).permute(2, 0, 1) * weights[i])
                                to_cat_memory_pos_dynamic.append(pos_enc.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                                to_cat_image_embed_dynamic.append(image_emd.cuda(non_blocking=True))

                memory_stack_ori = torch.stack(to_cat_memory_dynamic, dim=0) # 将所有的记忆库进行堆叠
                memory_pos_stack_ori = torch.stack(to_cat_memory_pos_dynamic, dim=0)
                image_embed_stack_ori = torch.stack(to_cat_image_embed_dynamic, dim=0)
                # vision_feats_temp 当前特征  memory_stack_ori 是记忆库中特征堆叠
                image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1) #标准化 图像255*22*22
                vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1) #当前输入图像的标准化  
                similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t() #利用cos 相似性
                
                similarity_scores = F.softmax(similarity_scores, dim=1) 
                #采样函数，从挑出的最相似的特种中再次进行采样
                sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)  

                memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))


                vision_feats[-1] = self.model.memory_attention(
                    curr=[vision_feats[-1]],
                    curr_pos=[vision_pos_embeds[-1]],
                    memory=memory,
                    memory_pos=memory_pos,
                    num_obj_ptr_tokens=0
                    )   
                
        feat_sizes = []
        for feat in vision_feats:
            H = W = int(torch.sqrt(torch.tensor(feat.size(0))))  # HW is first dimension
            feat_sizes.append((H, W))
        

        feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size) 
                     for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]

        image_embed = feats[-1]
        high_res_feats = feats[:-1]

        # 此处怎么解决
        '''prompt encoder'''         
        with torch.no_grad():

            click_torch = torch.as_tensor(click, dtype=torch.float, device=torch.device("cuda")).unsqueeze(1)
            click_lable_torch = torch.as_tensor([1]*x.size(0), dtype=torch.int, device=torch.device("cuda")).view(-1, 1)
            click_points=(click_torch, click_lable_torch)
            se, de = self.model.sam_prompt_encoder(  #point  prompt
                points=click_points, #(coords_torch, labels_torch)
                boxes=None,
                masks=None,
            ) 
            '''train mask decoder'''      
            # prodict 
            low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = self.model.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=self.model.sam_prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False, # args.multimask_output if you want multiple masks
                    repeat_image=False,  # the image is already batched
                    high_res_features = high_res_feats
                )  
                    # resize prediction
            pred = F.interpolate(low_res_multimasks,size=(x.size(2),x.size(3)))
            high_res_multimasks = F.interpolate(low_res_multimasks, size=(x.size(2),x.size(3)),
                                                mode="bilinear", align_corners=False)

            '''memory encoder'''       
            # new caluculated memory features # memory_encoder  控制通道
            # based on mutiple scale ,before input ,the cannel is 256 
            # add by leiwb start
            # 32 64 256
            pix_channel = [ele.size(2) for ele in vision_feats]
            # self.conv_s2 = nn.Conv2d(pix_channel[-2], self.model.hidden_dim, kernel_size=1, stride=1,device=torch.device("cuda"))
            # self.conv_s3 = nn.Conv2d(pix_channel[-3], self.model.hidden_dim, kernel_size=1, stride=1,device=torch.device("cuda"))

            # # top-level feature, (HW)BC => BCHW
            for i in range(len(vision_feats)):
                vision_feats[i] = vision_feats[i].permute(1, 2, 0).view(B, pix_channel[i], *feat_sizes[i])
            
            # 
            # vision_feats[-2] = self.conv_s2(vision_feats[-2])
            # vision_feats[-3] = self.conv_s3(vision_feats[-3])
            vision_feats[-2] = self.conv_s2(vision_feats[-2])
            vision_feats[-3] = self.conv_s3(vision_feats[-3])
            # BCHW => (HW)BC
            vision_feats = [feat.flatten(2).permute(2, 0, 1) for feat in vision_feats]
            # add by leiwb end

            # self.model._encode_new_memory 经过编码将当前特征编码 =>记忆库中的权重
            # 将记忆权重进行放入后续记忆库中，输出记忆权重可以进行使用 多尺度采样融合，然后进行进入记忆库中
            maskmem_features, maskmem_pos_enc,memory_out_array = self.model._encode_new_memory(
                current_vision_feats=vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_multimasks,
                is_mask_from_pts=True)  
            # dimension hint for your future use
            # maskmem_features: torch.Size([batch, 64, 64, 64])
            # maskmem_pos_enc: [torch.Size([batch, 64, 64, 64])]
            # memory_out_array[0]["vision_features"].shape torch.Size([12, 64, 22, 22])
            # memory_out_array[1]["vision_features"].shape torch.Size([12, 64, 44, 44])
            # memory_out_array[2]["vision_features"].shape torch.Size([12, 64, 88, 88])

            mm_44      = memory_out_array[1]["vision_features"]
            mmpos_44   = memory_out_array[1]["vision_pos_enc"]

            mm_88      = memory_out_array[2]["vision_features"]
            mmpos_88   = memory_out_array[2]["vision_pos_enc"]
    
            # 使用预定义的 MFB 模块
            maskmem_features_mfb = self.feat_small(maskmem_features)
            # mm_44_mfb = self.feat_mid(mm_44)
            # mm_88_mfb = self.feat_large(mm_88)


            # mm_2 = self.down_2(mm_44_mfb)
            # mm_4 = self.down_4(mm_88_mfb)

            # current_mm  =  (maskmem_features_mfb+ mm_2 + mm_4)/3
            current_mm  =  maskmem_features_mfb

            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(device=torch.device("cuda"), non_blocking=True)
            maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
            maskmem_pos_enc = maskmem_pos_enc.to(device=torch.device("cuda"), non_blocking=True)


            # add single maskmem_features, maskmem_pos_enc, iou
            if len(self.memory_bank.memories)== 0:
                for batch in range(maskmem_features.size(0)):
                    
                    self.memory_bank.update(
                                            (maskmem_features[batch].unsqueeze(0)).detach(),
                                            (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                            iou_predictions[batch, 0],
                                            image_embed[batch].reshape(-1).detach()
                                        )
            else:
                for batch in range(maskmem_features.size(0)):

                    push_maskmem_features = self.scale_factor*maskmem_features[batch].unsqueeze(0) + (1-self.scale_factor)*current_mm[batch]

                    self.memory_bank.update(push_maskmem_features, # 更新输出由self.model._encode_new_memory
                                            maskmem_pos_enc[batch].unsqueeze(0),
                                            iou_predictions[batch, 0],
                                            image_embed[batch].reshape(-1)
                                            )
        # 内存机制的地方
        # 以下 x1 x2 x3 x4  需要相同的通道
        # 上采样的过程
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        pr =  (high_res_multimasks + out)/2
        # return out, out1, out2
        return pr, out1, out2

if __name__ == "__main__":
    with torch.no_grad():
        # 输出部分进行融合
        model = MMSAM2().cuda()
        x = torch.randn(2, 3, 352, 352).cuda()
        out, out1, out2 = model(x)
        print(out.shape, out1.shape, out2.shape)
        # SAM2UNet 中初始化:
        # memory1 = memory_forward(256, 256, self)  # 传入self以使用SAM2的memory attention
