from torch.utils.data import DataLoader
import torch
import tqdm
from torch.amp import GradScaler, autocast
import os
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

from config import MODEL, DEVICE
from dataset import LibriSpeechDataset
from config import load_new_model


def train(
    batch_size: int = 16,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    use_amp: bool = True,
):
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Load the dataset
    dataset = LibriSpeechDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(dataset)} samples")

    # Load the model
    model = MODEL.to(DEVICE)
    model.train()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for i, batch in enumerate(
            tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        ):
            # Extract and move batch to device
            input_features = batch["input_features"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward pass with mixed precision
            with autocast(device_type=DEVICE, enabled=use_amp):
                outputs = model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            # Backward pass with gradient scaling
            optimizer.zero_grad()

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

            print(f"Loss: {loss.item()}")

        # Print epoch summary
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Average Loss: {epoch_loss/len(dataloader):.4f}"
        )

        # # Save checkpoint after each epoch
        # torch.save(
        #     model.state_dict(), f"models/whisper-tiny-librispeech-epoch-{epoch+1}.pth"
        # )

    # Save the final model
    torch.save(model.state_dict(), "models/whisper-tiny-librispeech.pth")

    return model


class KLTrainer(Trainer):
    """
    Custom trainer that implements KL divergence between teacher and student models.
    """

    def __init__(self, teacher_model=None, kd_weight=0.5, **kwargs):
        """
        Initialize the KL Trainer.

        Args:
            teacher_model: The teacher model to distill knowledge from
            kd_weight: Weight for KL divergence loss (between 0 and 1)
            **kwargs: Additional arguments passed to the Trainer constructor
        """
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.kd_weight = kd_weight

        # Ensure teacher model is in eval mode
        if self.teacher_model is not None:
            self.teacher_model.eval()

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        Compute the combined loss: CE loss + KL divergence loss.

        Args:
            model: The student model
            inputs: The inputs to the model
            return_outputs: Whether to return the outputs along with the loss
            num_items_in_batch: Number of items in the batch (ignored)

        Returns:
            The combined loss, or a tuple of (loss, outputs) if return_outputs is True
        """
        # Ensure inputs are on the correct device
        model_inputs = {
            "input_features": inputs["input_features"].to(model.device),
            "attention_mask": inputs["attention_mask"].to(model.device),
            "labels": inputs["labels"].to(model.device),
        }

        outputs = model(**model_inputs)
        ce_loss = outputs.loss

        # If no teacher model or kd_weight is 0, just use CE loss
        if self.teacher_model is None or self.kd_weight == 0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # Get teacher logits (no grad needed)
        with torch.no_grad():
            # Ensure teacher model is on the same device as the student model
            if self.teacher_model.device != model.device:
                self.teacher_model = self.teacher_model.to(model.device)

            teacher_outputs = self.teacher_model(**model_inputs)
            teacher_logits = teacher_outputs.logits

        # Get student logits
        student_logits = outputs.logits

        # KL divergence loss
        # Apply log_softmax to student logits and softmax to teacher logits
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        # Calculate KL divergence loss
        kd_loss = F.kl_div(
            student_log_probs.view(-1, student_log_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction="batchmean",
        )

        # Combine losses
        loss = (1 - self.kd_weight) * ce_loss + self.kd_weight * kd_loss

        return (loss, outputs) if return_outputs else loss


def train_lora_kd(
    batch_size: int = 16,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    use_amp: bool = True,
    lora_r: int = 4,  # Smaller LoRA rank
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    kd_weight: float = 0.5,  # Weight for KL divergence loss
):
    """
    Train the model using LoRA with KL divergence loss using HuggingFace's Trainer.

    Args:
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        use_amp: Whether to use automatic mixed precision
        lora_r: Rank of the LoRA update matrices
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        kd_weight: Weight for KL divergence loss
    """
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Load the dataset
    dataset = LibriSpeechDataset()
    print(f"Loaded {len(dataset)} samples")

    # Load the base model (teacher)
    teacher_model = MODEL.to(DEVICE)
    teacher_model.eval()  # Teacher model in eval mode

    # Create a copy for the student model with LoRA
    student_model = load_new_model().to(DEVICE)

    # Define LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],  # Focus on attention modules
        inference_mode=False,
    )

    # Apply LoRA to the student model
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="models/whisper-tiny-lora-kd",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_strategy="no",
        fp16=use_amp and DEVICE == "cuda",
        report_to="none",
    )

    # Create the KL Trainer
    trainer = KLTrainer(
        model=student_model,
        teacher_model=teacher_model,
        kd_weight=kd_weight,
        args=training_args,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train()

    # Save the LoRA model properly
    save_path = "models/whisper-tiny-lora-kd"
    student_model.save_pretrained(save_path)

    print(f"LoRA adapter saved to {save_path}")

    return student_model
