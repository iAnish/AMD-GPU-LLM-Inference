import os
import streamlit as st
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from accelerate import Accelerator
import warnings
warnings.filterwarnings("ignore")

# ROCm and AMD GPU Optimizations
# Assume PyTorch ROCm is installed: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1/
# For bitsandbytes ROCm: compile from source as per ROCm docs
os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1100'  # Adjust for your AMD GPU architecture (e.g., gfx1100 for MI300X, gfx1030 for RX 6000)
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'  # Example for RDNA3; set based on rocm-smi
torch.backends.cudnn.benchmark = True  # Enables auto-tuning for ROCm (HIP CUDNN)
torch.backends.cudnn.deterministic = False
torch.backends.rocm.synchronize = False  # Allow asynchronous execution for better throughput
accelerator = Accelerator(mixed_precision="bf16")  # Use bfloat16 for AMD GPUs

# Model and Dataset Configs
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
DATASET_ID = "HuggingFaceH4/llava-instruct-mix-vsft"
SAMPLE_SIZE = 100  # Small sample for demo; increase for full training

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 24px;
        font-weight: bold;
        font-size: 16px;
    }
    .stSlider > div > div > div > div {
        background: #4ECDC4;
    }
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #4ECDC4;
    }
    .output-box {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

def collate_fn(examples, processor):
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    images = [example["images"] for example in examples]
    # Assume single image per example; adjust if multi-image supported
    images = [img[0] if isinstance(img, list) else img for img in images]
    
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    # Ignore image token in loss
    if hasattr(processor, 'image_token'):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
    batch["labels"] = labels
    return batch

# Streamlit App
st.title("🔥 Llama 3.2 Vision Fine-Tuning on AMD ROCm")
st.subheader("Efficient LoRA Fine-Tuning with Stunning UI! 🚀")

# Sidebar for configurations
st.sidebar.header("⚙️ Training Parameters")
epochs = st.sidebar.slider("Epochs", 1, 5, 1)
lr = st.sidebar.number_input("Learning Rate", value=4e-5, format="%.2e")
batch_size = st.sidebar.slider("Batch Size", 1, 8, 2)  # Adjust based on GPU memory
grad_acc_steps = st.sidebar.slider("Gradient Accumulation Steps", 1, 16, 4)
lora_r = st.sidebar.slider("LoRA Rank (r)", 8, 64, 16)
lora_alpha = st.sidebar.slider("LoRA Alpha", 8, 32, 16)
sample_size = st.sidebar.slider("Dataset Sample Size", 50, 1000, SAMPLE_SIZE)

st.sidebar.header("📊 ROCm Optimizations")
st.sidebar.write("• bfloat16 mixed precision for AMD GPUs")
st.sidebar.write("• LoRA for parameter-efficient fine-tuning")
st.sidebar.write("• Gradient checkpointing & auto device mapping")
st.sidebar.write("• Paged AdamW optimizer for memory efficiency")
st.sidebar.write("• Set HSA_OVERRIDE_GFX_VERSION for your GPU arch")

# Main content
if st.button("🚀 Start Fine-Tuning!"):
    with st.spinner("Loading dataset and model... This may take a while on first run."):
        # Load dataset (subset for sample)
        dataset = load_dataset(DATASET_ID, split="train")
        dataset = dataset.select(range(sample_size))
        st.success(f"Loaded {len(dataset)} samples from {DATASET_ID}")
        
        # Load processor and model
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "right"
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=False,  # Use bf16 instead for ROCm stability
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = MllamaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto",  # Accelerate handles ROCm GPUs
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"  # Flash for speed if supported
        )
        model.config.use_cache = False
        
        # LoRA Config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # For Llama
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Training Args with AMD Optimizations
        training_args = TrainingArguments(
            output_dir="./llama-vision-finetuned",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_acc_steps,
            optim="paged_adamw_32bit",  # Memory efficient for large models
            save_steps=50,
            logging_steps=10,
            learning_rate=lr,
            weight_decay=0.001,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="none",  # Or "tensorboard" if wandb installed
            bf16=True,  # Native bfloat16 on AMD GPUs
            gradient_checkpointing=True,  # Memory savings
            dataloader_pin_memory=False,  # Better for ROCm
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True}
        )
        
        # Trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=None,  # Add if needed
            peft_config=peft_config,
            dataset_text_field=None,  # Custom collate
            max_seq_length=512,  # Adjust for vision inputs
            tokenizer=processor.tokenizer,
            data_collator=lambda examples: collate_fn(examples, processor),
            packing=False,  # For vision, no packing
        )
        
        st.markdown('<div class="output-box">', unsafe_allow_html=True)
        st.write("**Trainable Parameters:**")
        st.write(f"{model.print_train
