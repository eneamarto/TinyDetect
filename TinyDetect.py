import json
import os
import random
import re
import traceback
from collections import defaultdict, Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    ViTImageProcessor,
    ViTModel,
    get_linear_schedule_with_warmup,
)

data_dir = r"/kaggle/input/coco-2017-dataset/coco2017"
images_dir = os.path.join(data_dir, "train2017")
valid_img_dir = os.path.join(data_dir, "val2017")
annotations_path = os.path.join(data_dir, "annotations", "instances_train2017.json")
val_annotations_path = os.path.join(data_dir, "annotations", "instances_val2017.json")

@dataclass
class TrainingConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    vit_name: str = "google/vit-base-patch16-224"
    data_file: str = "detection_training_data.json"
    val_data_file: str = "detection_validation_data.json"
    image_dir: str = images_dir
    val_image_dir: str = valid_img_dir
    max_length: int = 128
    output_dir: str = "/kaggle/working/checkpoints"
    num_epochs: int = 6
    batch_size: int = 16
    learning_rate: float = 1e-3
    gradient_accumulation_steps = 2
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    save_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 250

def training_prompts():
    return [
        "What objects do you see and where are they located?",
        "Detect all people and cars in this image with their coordinates.",
        "Find the bounding boxes for all visible objects.",
        "How many people and cars are in this image? Provide their locations.",
        "Count all objects and give me their exact positions.",
        "Where exactly are the people and cars positioned in this image?",
        "Identify the precise locations of all visible subjects.",
        "Can you spot any people or vehicles? Where are they?",
        "Please identify and locate all visible objects in the scene.",
    ]

def standardized_response(objects, img_width, img_height):
    if not objects:
        return "I don't see any people or cars in this image."

    descriptions = []
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        x1_norm = max(0, min(int((x1 / img_width) * 1000), 1000))
        y1_norm = max(0, min(int((y1 / img_height) * 1000), 1000))
        x2_norm = max(x1_norm + 1, min(int((x2 / img_width) * 1000), 1000))
        y2_norm = max(y1_norm + 1, min(int((y2 / img_height) * 1000), 1000))
        descriptions.append(f"{obj['name']} at <{x1_norm},{y1_norm},{x2_norm},{y2_norm}>")

    if len(objects) == 1:
        return f"I found 1 object: {descriptions[0]}."
    else:
        return f"I found {len(objects)} objects: {', '.join(descriptions)}."

def create_balanced_coco_dataset(coco_file, output_file, images_dir, target_categories=["person", "car"], samples_per_class=1000):
    with open(coco_file, "r") as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    cat_name_to_id = {cat["name"]: cat["id"] for cat in coco["categories"]}
    target_cat_ids = [cat_name_to_id[name] for name in target_categories if name in cat_name_to_id]

    print(f"Target categories: {target_categories}")

    image_annotations = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] in target_cat_ids and ann["bbox"][2] > 0 and ann["bbox"][3] > 0:
            image_annotations[ann["image_id"]].append(ann)

    class_samples = {cat_name: [] for cat_name in target_categories}
    
    for img in coco["images"]:
        if img["id"] in image_annotations:
            objects = []
            img_width, img_height = img["width"], img["height"]
            
            category_counts = Counter()
            for ann in image_annotations[img["id"]]:
                category_counts[categories[ann["category_id"]]] += 1
                x, y, w, h = ann["bbox"]
                objects.append({
                    "name": categories[ann["category_id"]],
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                })

            if objects:
                dominant_class = category_counts.most_common(1)[0][0]
                
                sample_data = {
                    "image": img["file_name"],
                    "image_width": img_width,
                    "image_height": img_height,
                    "objects": objects,
                    "category_counts": dict(category_counts)
                }
                
                if len(class_samples[dominant_class]) < samples_per_class:
                    class_samples[dominant_class].append(sample_data)

    min_samples = min(len(samples) for samples in class_samples.values())

    balanced_samples = []
    for cat_name, samples in class_samples.items():
        random.shuffle(samples)
        balanced_samples.extend(samples[:min_samples])

    random.shuffle(balanced_samples)

    training_data = []
    prompts = training_prompts()

    for sample in balanced_samples:
        response = standardized_response(sample["objects"], sample["image_width"], sample["image_height"])
        training_example = {
            "image": sample["image"],
            "image_width": sample["image_width"],
            "image_height": sample["image_height"],
            "conversations": [
                {"from": "human", "value": random.choice(prompts)},
                {"from": "gpt", "value": response},
            ],
        }
        training_data.append(training_example)

    with open(output_file, "w") as f:
        json.dump(training_data, f, indent=2)

    final_class_counts = Counter()
    for sample in balanced_samples:
        for cat_name in sample["category_counts"]:
            final_class_counts[cat_name] += sample["category_counts"][cat_name]

    print(f"{len(training_data)} training examples")
    print(f"Data saved to {output_file}")
    
    return training_data

class DetectionDataset(Dataset):
    def __init__(self, data_file, image_dir, tokenizer, image_processor, max_length=128):
        with open(data_file, "r") as f:
            self.data = json.load(f)

        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        print(f"Dataset loaded: {len(self.data)} samples")

        for i in range(min(3, len(self.data))):
            img_path = os.path.join(self.image_dir, self.data[i]["image"])
            if not os.path.exists(img_path):
                print(f"image not found")
            else:
                print(f"image found: {img_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            image_path = os.path.join(self.image_dir, item["image"])
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path).convert("RGB")
            image_inputs = self.image_processor(image, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"].squeeze(0)

            conversation = item["conversations"]
            prompt = conversation[0]["value"]
            response = conversation[1]["value"]

            full_text = (
                f"USER: {prompt}\nASSISTANT: {response}{self.tokenizer.eos_token}"
            )

            encodings = self.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = encodings["input_ids"].squeeze(0)
            attention_mask = encodings["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            prefix = f"USER: {prompt}\nASSISTANT:"
            prefix_ids = self.tokenizer(
                prefix,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )["input_ids"].squeeze(0)

            assistant_token_start = min(len(prefix_ids), self.max_length)
            labels[:assistant_token_start] = -100

            labels[attention_mask == 0] = -100

            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

        except Exception:
            return {
                "pixel_values": torch.zeros(3, 224, 224),
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
                "labels": torch.full((self.max_length,), -100, dtype=torch.long),
            }

class TinyLlamaViTDetector(nn.Module):
    def __init__(self, llama_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", vit_model_name="google/vit-base-patch16-224"):
        super().__init__()
        self.llama_model_name = llama_model_name
        self.vit_model_name = vit_model_name
        self._load_models()
        self._init_projector()
        self._setup_tokenizer()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _load_models(self):
        self.llama = LlamaForCausalLM.from_pretrained(
            self.llama_model_name, 
            torch_dtype=torch.float32, 
            trust_remote_code=True
        )
        self.vit = ViTModel.from_pretrained(
            self.vit_model_name, 
            torch_dtype=torch.float32, 
            add_pooling_layer=False
        )
        for param in self.vit.parameters():
            param.requires_grad = False

    def _init_projector(self):
        vision_input_dim = 1536
        llama_hidden_size = self.llama.config.hidden_size

        self.vision_projector = nn.Sequential(
            nn.Linear(vision_input_dim, 1792),
            nn.LayerNorm(1792),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1792, llama_hidden_size),
            nn.LayerNorm(llama_hidden_size),
        )

        with torch.no_grad():
            for module in self.vision_projector:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def _setup_tokenizer(self):
        self.image_processor = ViTImageProcessor.from_pretrained(self.vit_model_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.llama_model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        self.llama.resize_token_embeddings(len(self.tokenizer))

    def setup_lora(self):
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.2,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.llama.gradient_checkpointing_enable()
        self.llama.config.use_cache = False
        self.llama = get_peft_model(self.llama, lora_config)
        self._print_trainable_params()

    def _print_trainable_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable:,} | Total: {total:,} | Percentage: {100*trainable/total:.2f}%")

    def encode_image(self, images):
        batch_size = images.shape[0]

        with torch.no_grad():
            vit_outputs = self.vit(images.to(self.vit.dtype))
            all_hidden_states = vit_outputs.last_hidden_state
            cls_features = all_hidden_states[:, 0, :]
            patch_features = all_hidden_states[:, 1:, :]
            avg_patch_features = patch_features.mean(dim=1)
        
        combined_features = torch.cat([cls_features, avg_patch_features], dim=1)
        image_embeds = self.vision_projector(combined_features)
        return image_embeds.unsqueeze(1)

    def forward(self, images=None, input_ids=None, attention_mask=None, labels=None):
        assert input_ids is not None, "input_ids cannot be None"
        batch_size = input_ids.shape[0]

        if images is not None:
            image_embeds = self.encode_image(images)
        else:
            image_embeds = torch.zeros(
                batch_size, 1, self.llama.config.hidden_size, device=self.device
            )

        text_embeds = self.llama.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        if attention_mask is not None:
            attention_mask = torch.cat(
                [torch.ones(batch_size, 1, device=self.device), attention_mask], 
                dim=1
            )

        if labels is not None:
            labels = torch.cat(
                [torch.full((batch_size, 1), -100, device=self.device), labels], 
                dim=1
            )

        outputs = self.llama(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            labels=labels
        )
        return outputs

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.llama.save_pretrained(output_dir, safe_serialization=True)
        self.tokenizer.save_pretrained(output_dir)
        self.image_processor.save_pretrained(output_dir)

        torch.save({
            "vision_projector": self.vision_projector.state_dict(),
            "config": {
                "llama_model_name": self.llama_model_name,
                "vit_model_name": self.vit_model_name,
            },
        }, os.path.join(output_dir, "custom_components.bin"))

    @classmethod
    def from_pretrained(cls, model_path):
        load_device = "cuda" if torch.cuda.is_available() else "cpu"
        custom_components = torch.load(
            os.path.join(model_path, "custom_components.bin"),
            map_location=load_device,
        )

        model = cls(
            llama_model_name=custom_components["config"]["llama_model_name"],
            vit_model_name=custom_components["config"]["vit_model_name"],
        )

        model.llama = PeftModel.from_pretrained(
            model.llama, model_path, is_trainable=False
        )

        vision_projector_state = custom_components["vision_projector"]
        model.vision_projector.load_state_dict(vision_projector_state)
        model.vision_projector = model.vision_projector.to(model.device)

        return model.to(model.device)

def evaluate_model(model, val_dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in val_dataloader:
            batch = {
                "images": batch_data["pixel_values"].to(device, dtype=model.vit.dtype),
                "input_ids": batch_data["input_ids"].to(device),
                "attention_mask": batch_data["attention_mask"].to(device),
                "labels": batch_data["labels"].to(device),
            }
            
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def train_model(config: TrainingConfig):    
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("initializing model...")
    model = TinyLlamaViTDetector(config.model_name, config.vit_name)
    model.setup_lora()
    model.to(device)
    
    scaler = torch.amp.GradScaler(enabled=config.mixed_precision)
    
    print("loading datasets...")
    train_dataset = DetectionDataset(
        config.data_file,
        config.image_dir,
        model.tokenizer,
        model.image_processor,
        config.max_length,
    )
    
    val_dataset = DetectionDataset(
        config.val_data_file,
        config.val_image_dir,
        model.tokenizer,
        model.image_processor,
        config.max_length,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    
    lora_params = [p for p in model.llama.parameters() if p.requires_grad]
    projector_params = list(model.vision_projector.parameters())

    optimizer = AdamW(
        [
            {"params": lora_params, "lr": 1e-3, "weight_decay": 0.0},
            {"params": projector_params, "lr": 1e-3, "weight_decay": 0.01},
        ],
        eps=1e-8,
    )
    
    num_training_steps = (
        config.num_epochs * len(train_dataloader) // config.gradient_accumulation_steps
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps,
    )
    
    model.train()
    progress_bar = tqdm(range(num_training_steps), desc="Training")
    global_step = 0
    total_loss = 0
    best_val_loss = float('inf')
    
    print("training...")
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, batch_data in enumerate(train_dataloader):
            try:
                batch = {
                    "images": batch_data["pixel_values"].to(device, dtype=model.vit.dtype),
                    "input_ids": batch_data["input_ids"].to(device),
                    "attention_mask": batch_data["attention_mask"].to(device),
                    "labels": batch_data["labels"].to(device),
                }
                
                with torch.autocast(device_type=device.type, enabled=config.mixed_precision):
                    outputs = model(**batch)
                    loss = outputs.loss / config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=0.5,
                        error_if_nonfinite=False,
                    )
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    
                    global_step += 1
                    progress_bar.update(1)
                
                total_loss += outputs.loss.item()
                epoch_loss += outputs.loss.item()
                
                if global_step % config.logging_steps == 0:
                    avg_loss = total_loss / config.logging_steps
                    current_lr = optimizer.param_groups[0]["lr"]
                    progress_bar.set_postfix({
                        "loss": avg_loss,
                        "lr": f"{current_lr:.2e}",
                        "step": global_step,
                    })
                    total_loss = 0
                    
            except RuntimeError as e:
                traceback.print_exc()
                raise
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        
        print("validating...")
        val_loss = evaluate_model(model, val_dataloader, device)
        model.train()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_dir = os.path.join(config.output_dir, "best_model")
            model.save_pretrained(best_model_dir)
            print(f"best model saved val_loss: {val_loss:.4f}")
        
        print(f"Epoch {epoch+1} complete | Train Loss: {avg_epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Best Val Loss: {best_val_loss:.4f}")
    
    final_dir = os.path.join(config.output_dir, "final_model")
    model.save_pretrained(final_dir)
    print(f"\nTraining completed!")
    print(f"Final model saved to {final_dir}")
    print(f"Best model saved to {os.path.join(config.output_dir, 'best_model')} with val_loss: {best_val_loss:.4f}")
    progress_bar.close()

class DetectionInference:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TinyLlamaViTDetector.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.image_processor = self.model.image_processor
        self.tokenizer = self.model.tokenizer

    def post_process_coordinates(self, response, img_width, img_height):
        norm_pattern = r"<(\d+),(\d+),(\d+),(\d+)>"
        matches = re.findall(norm_pattern, response)

        if not matches:
            return response

        for match in matches:
            x1_norm, y1_norm, x2_norm, y2_norm = map(int, match)

            x1 = int((x1_norm / 1000) * img_width)
            y1 = int((y1_norm / 1000) * img_height)
            x2 = int((x2_norm / 1000) * img_width)
            y2 = int((y2_norm / 1000) * img_height)

            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(x1, min(x2, img_width))
            y2 = max(y1, min(y2, img_height))

            old_coord = f"<{x1_norm},{y1_norm},{x2_norm},{y2_norm}>"
            new_coord = f"({x1},{y1},{x2},{y2})"
            response = response.replace(old_coord, new_coord)

        return response

    def detect(self, image_path: str, prompt=None):
        try:
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size
            image_inputs = self.image_processor(image, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"].to(
                device=self.device, dtype=self.model.vit.dtype
            )

            prompt = prompt or "What objects do you see and where are they located?"
            
            text_inputs = self.tokenizer(
                f"USER: {prompt}\nASSISTANT:",
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            with torch.inference_mode():
                image_embeds = self.model.encode_image(pixel_values)
                text_embeds = self.model.llama.get_input_embeddings()(text_inputs["input_ids"])
                inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

                attention_mask = torch.cat([
                    torch.ones(inputs_embeds.shape[0], 1, device=self.device),
                    text_inputs["attention_mask"],
                ], dim=1)

                generation_config = GenerationConfig(
                    max_new_tokens=200,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                outputs = self.model.llama.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                )

                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = decoded.split("ASSISTANT:")[-1].strip()
                response = self.post_process_coordinates(response, img_width, img_height)
                return response

        except Exception as e:
            print(f"error in detection: {str(e)}")
            traceback.print_exc()
            return None

def main():

    target_categories = ["person", "car"]
    
    train_data = create_balanced_coco_dataset(
        annotations_path,
        "detection_training_data.json",
        images_dir,
        target_categories=target_categories,
        samples_per_class=1500
    )

    val_data = create_balanced_coco_dataset(
        val_annotations_path,
        "detection_validation_data.json",
        valid_img_dir,
        target_categories=target_categories,
        samples_per_class=100
    )

    print(f"\ntraining examples: {len(train_data)}")
    print(f"validation examples: {len(val_data)}")

    config = TrainingConfig(
        data_file="detection_training_data.json",
        val_data_file="detection_validation_data.json",
        image_dir=images_dir,
        val_image_dir=valid_img_dir,
        max_length=128,
        num_epochs=10
    )

    print("\n" + "=" * 60)
    print("starting training...")
    print(f"batch size: {config.batch_size * config.gradient_accumulation_steps}")

    try:
        train_model(config)
        print("\n" + "=" * 60)
        print(f"final model: {os.path.join(config.output_dir, 'final_model')}")
        print(f"best model: {os.path.join(config.output_dir, 'best_model')}")

    except Exception as e:
        traceback.print_exc()
        return False

    return True
    
if __name__ == "__main__":
    main()
