"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train

from lavis.transformers_v4p34_local import AutoTokenizer, OPTForCausalLM, OPTConfig
import lavis.transformers_v4p34_local as transformers


@registry.register_model("blip2_opt")
class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 OPT requires transformers>=4.27"
        
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        logging.info("freeze language model")
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        
        total_params = 0
        for param in self.named_parameters():
            total_params += param[1].numel()
        print("Total parameters: ", total_params)

    def forward(self, config, samples):
        '''
        samples:
            ['image_list'([64, 3, 364, 364]), 'caption_list'(64, x), 'masked_images_list'([123, 3, 364, 364]), 'gen_caps_list'(123, x), \
                'masked_item_lens'(len: 64, sum: 123), 'entities_list'(len: 123), 'prefixes_list'(len: 123), \
                'epoch', 'num_iters_per_epoch', 'iters']
        '''
        images = samples["image_list"]
        
        (masked_images, gen_caps, masked_item_lens, entities, prefixes) = samples["masked_images_list"], \
            samples["gen_caps_list"], samples["masked_item_lens"], samples["entities_list"], samples["prefixes_list"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(images))
            if config.run_cfg['do_TE'] or config.run_cfg['do_NDE']:
                masked_images_embeds = self.ln_vision(self.visual_encoder(masked_images))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(images.device)
        if config.run_cfg['do_TE'] or config.run_cfg['do_NDE']:
            masked_image_atts = torch.ones(masked_images_embeds.size()[:-1], dtype=torch.long).to(images.device)


        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        if config.run_cfg['do_TE'] or config.run_cfg['do_NDE']:
            masked_query_tokens = self.query_tokens.expand(masked_images_embeds.shape[0], -1, -1)
            masked_query_output = self.Qformer.bert(
                query_embeds=masked_query_tokens,
                encoder_hidden_states=masked_images_embeds,
                encoder_attention_mask=masked_image_atts,
                return_dict=True,
            )


        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(images.device)

        if config.run_cfg['do_TE'] or config.run_cfg['do_NDE']:
            masked_inputs_opt = self.opt_proj(masked_query_output.last_hidden_state)
            masked_atts_opt = torch.ones(masked_inputs_opt.size()[:-1], dtype=torch.long).to(images.device)


        self.opt_tokenizer.padding_side = "right"

        texts = [t + "\n" for t in samples["caption_list"]]

        opt_tokens = self.opt_tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(images.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt


        if config.run_cfg['do_TE'] or config.run_cfg['do_NDE']:
            ### note that self.opt_tokenizer if different from self.tokenizer
            prefixes_lens = [len(self.opt_tokenizer.encode(prefix, return_tensors="pt")[0,1:]) for prefix in prefixes]
            prefixes_lens = torch.tensor(prefixes_lens).to(images.device)
            entities_ids = [self.opt_tokenizer.encode(entity, return_tensors="pt")[0,1:] for entity in entities]
            
            gen_cap_texts = [t + "\n" for t in gen_caps]
            gen_cap_texts = self.opt_tokenizer(
                gen_cap_texts,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt"
            ).to(images.device)
            gen_cap_decoder_targets = gen_cap_texts.input_ids.masked_fill(
                gen_cap_texts.input_ids == self.opt_tokenizer.pad_token_id, -100
            )
            if self.prompt:
                gen_cap_decoder_targets[:, : self.prompt_length] = -100


        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(images.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        if config.run_cfg['do_TE'] or config.run_cfg['do_NDE']:
            gen_cap_empty_targets = (
                torch.ones(masked_atts_opt.size(), dtype=torch.long).to(images.device).fill_(-100)
            )
            gen_cap_decoder_targets = torch.cat([gen_cap_empty_targets, gen_cap_decoder_targets], dim=1)


        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        if config.run_cfg['do_TE'] or config.run_cfg['do_NDE']:
            gen_cap_inputs_embeds = self.opt_model.model.decoder.embed_tokens(gen_cap_texts.input_ids)
            gen_cap_inputs_embeds_te = torch.cat([masked_inputs_opt, gen_cap_inputs_embeds], dim=1)
            gen_cap_attention_mask = torch.cat([masked_atts_opt, gen_cap_texts.attention_mask], dim=1)
            gen_cap_inputs_embeds_nde, gen_cap_attention_mask_nde = None, None
            if config.run_cfg['do_NDE']:
                repeated = torch.repeat_interleave(torch.arange(len(masked_item_lens)).to(images.device), \
                                                torch.tensor(masked_item_lens).to(images.device))
                repeated_inputs_opt = torch.index_select(inputs_opt, 0, repeated)
                repeated_atts_opt = torch.index_select(atts_opt, 0, repeated)
                gen_cap_inputs_embeds_nde = torch.cat([repeated_inputs_opt, gen_cap_inputs_embeds], dim=1)
                gen_cap_attention_mask_nde = torch.cat([repeated_atts_opt, gen_cap_texts.attention_mask], dim=1)


        if config.run_cfg['do_TE'] or config.run_cfg['do_NDE']:
            with self.maybe_autocast():
                outputs, losses = self.opt_model(
                    inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask,
                    labels = targets,
                    alpha = config.run_cfg['alpha'],
                    do_TE = config.run_cfg['do_TE'],
                    do_NDE = config.run_cfg['do_NDE'],
                    gen_cap_inputs_embeds = gen_cap_inputs_embeds_te,
                    gen_cap_attention_mask = gen_cap_attention_mask,
                    gen_cap_decoder_targets = gen_cap_decoder_targets,
                    gen_cap_inputs_embeds_nde = gen_cap_inputs_embeds_nde,
                    gen_cap_attention_mask_nde = gen_cap_attention_mask_nde,
                    masked_item_lens = masked_item_lens,
                    entities_ids = entities_ids,
                    prefixes_lens = prefixes_lens,
                    prompt_length = self.prompt_length[0].to(images.device),
                    return_dict = True,
                    is_training = True,
                )
        else:
            with self.maybe_autocast():
                outputs = self.opt_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
        agg_loss = outputs.loss

        if config.run_cfg['do_NDE']:
            return {"agg_loss": agg_loss, "lm_loss": losses[0].item(), "nde_loss": losses[2].item()}
        elif config.run_cfg['do_TE']:
            return {"agg_loss": agg_loss, "lm_loss": losses[0].item(), "te_loss": losses[1].item()}
        else:
            return {"agg_loss": agg_loss, "lm_loss": agg_loss.item()}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=20,
        min_length=5,
        top_k=20,
        top_p=1,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        images = samples["image_list"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(images))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                images.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                images.device
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * images.size(0)

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(images.device)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # new version for transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                # top_k=top_k,
                # top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            
            output_text = [text.strip() for text in output_text]
            return output_text
        
        
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
        
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # require transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
        
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model
