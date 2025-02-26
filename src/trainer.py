from torch.utils.data import DataLoader
import torch
import tqdm
from torch.amp import GradScaler, autocast
import os
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

from config import MODEL, DEVICE
from dataset import LibriSpeechDataset, collate_fn
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
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
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


class KLEWCTrainer(Trainer):
    """
    Custom trainer that implements KL divergence between teacher and student models,
    with optional Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting.
    """

    def __init__(
        self,
        teacher_model=None,
        kd_weight=0.5,
        use_ewc=False,
        ewc_lambda=100.0,
        fisher_estimation_sample_size=100,
        **kwargs,
    ):
        """
        Initialize the KL Trainer with optional EWC.

        Args:
            teacher_model: The teacher model to distill knowledge from
            kd_weight: Weight for KL divergence loss (between 0 and 1)
            use_ewc: Whether to use Elastic Weight Consolidation
            ewc_lambda: Weight for the EWC penalty term
            fisher_estimation_sample_size: Number of samples for Fisher estimation
            **kwargs: Additional arguments passed to the Trainer constructor
        """
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.kd_weight = kd_weight
        self.use_ewc = use_ewc
        self.ewc_lambda = ewc_lambda
        self.fisher_estimation_sample_size = fisher_estimation_sample_size
        self.fisher = None
        self.original_params = None

        # Ensure teacher model is in eval mode
        if self.teacher_model is not None:
            self.teacher_model.eval()

    def compute_fisher_information_matrix(self):
        """
        Compute the Fisher Information Matrix, which represents parameter importance.
        """
        if not self.use_ewc:
            return None

        print("Computing Fisher Information Matrix...")
        dataset_length = len(self.train_dataset)
        sample_size = min(self.fisher_estimation_sample_size, dataset_length)

        # Create a small dataloader for Fisher estimation
        fisher_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=1,  # Process one sample at a time
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset, num_samples=sample_size, replacement=False
            ),
            collate_fn=self.data_collator,
        )

        # Initialize Fisher matrix and store original parameters
        fisher = {}
        self.original_params = {}

        for name, param in self.model.named_parameters():
            # Only track non-LoRA parameters that require gradients
            is_lora_param = "lora" in name
            if param.requires_grad and not is_lora_param:
                fisher[name] = torch.zeros_like(param, device=param.device)
                self.original_params[name] = param.data.clone()

        # Set model to evaluation mode for Fisher calculation
        was_training = self.model.training
        self.model.eval()

        # Compute Fisher Information Matrix
        for batch in fisher_dataloader:
            # Move batch to device
            batch_inputs = {
                "input_features": batch["input_features"].to(self.model.device),
                "attention_mask": batch["attention_mask"].to(self.model.device),
                "labels": batch["labels"].to(self.model.device),
            }

            # Forward pass
            self.model.zero_grad()
            outputs = self.model(**batch_inputs)
            loss = outputs.loss

            # Backward pass to get gradients
            loss.backward()

            # Accumulate squared gradients in the Fisher matrix
            for name, param in self.model.named_parameters():
                if name in fisher and param.grad is not None:
                    fisher[name] += param.grad.pow(2).detach() / sample_size

        # Store the computed Fisher matrix
        self.fisher = fisher

        # Restore model's training state
        if was_training:
            self.model.train()

        print(f"Fisher Information Matrix computed for {len(fisher)} parameters.")
        return fisher

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        """
        Compute the combined loss: CE loss + KL divergence loss + optional EWC penalty.

        Args:
            model: The student model
            inputs: The inputs to the model
            return_outputs: Whether to return the outputs along with the loss
            num_items_in_batch: Number of items in the batch (ignored)

        Returns:
            The combined loss, or a tuple of (loss, outputs) if return_outputs is True
        """
        # Explicitly select only the keys that the model expects
        model_inputs = {
            "input_features": inputs["input_features"].to(model.device),
            "attention_mask": inputs["attention_mask"].to(model.device),
            "labels": inputs["labels"].to(model.device),
        }

        outputs = model(**model_inputs)
        ce_loss = outputs.loss

        # Initialize combined loss with CE loss
        combined_loss = ce_loss

        # Add KL divergence if teacher model is available
        if self.teacher_model is not None and self.kd_weight > 0:
            # Get teacher logits (no grad needed)
            with torch.no_grad():
                # Ensure teacher model is on the same device as the student model
                if self.teacher_model.device != model.device:
                    self.teacher_model = self.teacher_model.to(model.device)

                teacher_outputs = self.teacher_model(**model_inputs)
                teacher_logits = teacher_outputs.logits

            # KL divergence loss between teacher and student logits
            student_log_probs = F.log_softmax(outputs.logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)

            # Calculate KL divergence loss
            kd_loss = F.kl_div(
                student_log_probs.view(-1, student_log_probs.size(-1)),
                teacher_probs.view(-1, teacher_probs.size(-1)),
                reduction="batchmean",
            )

            # Combine CE and KL losses
            combined_loss = (1 - self.kd_weight) * ce_loss + self.kd_weight * kd_loss

        # Add EWC penalty if enabled and Fisher information is available
        ewc_loss = 0.0
        if (
            self.use_ewc
            and self.fisher is not None
            and self.original_params is not None
        ):
            for name, param in model.named_parameters():
                if name in self.fisher and name in self.original_params:
                    # Calculate squared distance from original parameters
                    delta = param - self.original_params[name]
                    ewc_loss += (self.fisher[name] * delta.pow(2)).sum()

            # Add the weighted EWC penalty to the combined loss
            combined_loss += (self.ewc_lambda / 2) * ewc_loss

        return (combined_loss, outputs) if return_outputs else combined_loss


def train_lora_ewc_kd(
    batch_size: int = 16,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    use_amp: bool = True,
    lora_r: int = 4,  # Smaller LoRA rank
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    kd_weight: float = 0.5,  # Weight for KL divergence loss
    use_ewc: bool = True,  # Whether to use EWC
    ewc_lambda: float = 100.0,  # Weight for EWC penalty
    fisher_estimation_sample_size: int = 100,  # Samples for Fisher estimation
):
    """
    Train the model using LoRA with knowledge distillation and optional EWC
    to prevent catastrophic forgetting.

    Args:
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        use_amp: Whether to use automatic mixed precision
        lora_r: Rank of the LoRA update matrices
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        kd_weight: Weight for KL divergence loss (0-1)
        use_ewc: Whether to use Elastic Weight Consolidation
        ewc_lambda: Weight for EWC penalty
        fisher_estimation_sample_size: Number of samples for Fisher estimation
    """
    # Create output directory
    output_dir = "models/whisper-tiny-lora-kd"
    if use_ewc:
        output_dir += "-ewc"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = LibriSpeechDataset()
    print(f"Loaded {len(dataset)} samples")

    # Setup teacher model (original pretrained model)
    teacher_model = MODEL.to(DEVICE)
    teacher_model.eval()  # Teacher model in eval mode

    # Setup student model with LoRA for efficient fine-tuning
    student_model = load_new_model().to(DEVICE)

    # Configure LoRA for attention modules
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
    )

    # Apply LoRA to the student model
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=10,
        save_strategy="no",
        fp16=use_amp and DEVICE == "cuda",
        report_to="none",
    )

    # Create the KL Trainer with optional EWC
    trainer = KLEWCTrainer(
        model=student_model,
        teacher_model=teacher_model,
        kd_weight=kd_weight,
        use_ewc=use_ewc,
        ewc_lambda=ewc_lambda,
        fisher_estimation_sample_size=fisher_estimation_sample_size,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    # Compute Fisher Information Matrix if using EWC
    if use_ewc:
        trainer.compute_fisher_information_matrix()

    # Train the model
    trainer.train()

    # Save the LoRA model properly
    student_model.save_pretrained(output_dir)
    return student_model


def train_lora_simple(
    batch_size: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    use_amp: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
):
    """
    Train the model using plain LoRA fine-tuning with the standard Hugging Face Trainer.

    Args:
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        use_amp: Whether to use automatic mixed precision
        lora_r: Rank of the LoRA update matrices
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
    """
    # Create output directory
    output_dir = f"models/whisper-lora-r{lora_r}"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = LibriSpeechDataset()
    print(f"Loaded {len(dataset)} samples")

    # Load model for fine-tuning
    model = load_new_model().to(DEVICE)

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",  # Don't train bias parameters
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Print trainable parameters info
    model.print_trainable_parameters()

    # Define a custom data collator that filters out unexpected keys
    def filtered_collate_fn(batch):
        # First apply the original collate function
        processed_batch = collate_fn(batch)

        # Keep only the keys that WhisperForConditionalGeneration expects
        allowed_keys = ["input_features", "attention_mask", "labels"]
        filtered_batch = {k: v for k, v in processed_batch.items() if k in allowed_keys}

        return filtered_batch

    # Define training arguments using Hugging Face's TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        fp16=use_amp and DEVICE == "cuda",
        logging_steps=50,
        save_strategy="no",  # Don't save checkpoints during training
        report_to="tensorboard",
        remove_unused_columns=False,  # Important for audio models
    )

    # Create standard Trainer with the filtered collate function
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=filtered_collate_fn,
    )

    # Train the model
    trainer.train()

    # Save the final model
    model.save_pretrained(os.path.join(output_dir, "final_model"))

    return model
