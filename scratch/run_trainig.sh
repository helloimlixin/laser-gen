#!/bin/bash
# Run LISTA-VQGAN training with DeepSpeed

# Default parameters
DATA_DIR="./data"
OUTPUT_DIR="./output/deepspeed"
BATCH_SIZE=64
EPOCHS=10
DICT_SIZE=512
LATENT_DIM=16
SPARSE_DIM=512
LISTA_STEPS=5
LEARNING_RATE=0.0001
DICT_LR=0.001
SPARSITY_WEIGHT=0.1
ADV_WEIGHT=0.5
SEED=42
GPUS=-1  # Use all available GPUs
PRECISION="16-mixed"
ZERO_STAGE=2
GRADIENT_ACCUMULATION=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --dict_size)
      DICT_SIZE="$2"
      shift 2
      ;;
    --latent_dim)
      LATENT_DIM="$2"
      shift 2
      ;;
    --sparse_dim)
      SPARSE_DIM="$2"
      shift 2
      ;;
    --lista_steps)
      LISTA_STEPS="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --dict_lr)
      DICT_LR="$2"
      shift 2
      ;;
    --sparsity_weight)
      SPARSITY_WEIGHT="$2"
      shift 2
      ;;
    --adv_weight)
      ADV_WEIGHT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --precision)
      PRECISION="$2"
      shift 2
      ;;
    --zero_stage)
      ZERO_STAGE="$2"
      shift 2
      ;;
    --gradient_accumulation)
      GRADIENT_ACCUMULATION="$2"
      shift 2
      ;;
    --offload)
      OFFLOAD="--offload"
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --data_dir DIR               Path to CIFAR-10 dataset (default: ./data)"
      echo "  --output_dir DIR             Output directory (default: ./output/deepspeed)"
      echo "  --batch_size SIZE            Batch size per GPU (default: 64)"
      echo "  --epochs NUM                 Number of epochs (default: 100)"
      echo "  --dict_size SIZE             Dictionary size (default: 512)"
      echo "  --latent_dim DIM             Latent dimension (default: 256)"
      echo "  --sparse_dim DIM             Sparse code dimension (default: 256)"
      echo "  --lista_steps STEPS          Number of LISTA steps (default: 5)"
      echo "  --learning_rate RATE         Learning rate (default: 0.0001)"
      echo "  --dict_lr RATE               Dictionary learning rate (default: 0.001)"
      echo "  --sparsity_weight WEIGHT     Sparsity loss weight (default: 0.1)"
      echo "  --adv_weight WEIGHT          Adversarial loss weight (default: 0.5)"
      echo "  --seed SEED                  Random seed (default: 42)"
      echo "  --gpus NUM                   Number of GPUs to use, -1 for all (default: -1)"
      echo "  --precision PREC             Training precision (default: 16-mixed)"
      echo "  --zero_stage STAGE           DeepSpeed ZeRO stage (0-3, default: 2)"
      echo "  --gradient_accumulation STEPS Gradient accumulation steps (default: 1)"
      echo "  --offload                    Enable CPU offloading for DeepSpeed"
      echo "  --help                       Show this help message and exit"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if DeepSpeed is installed
if ! python -c "import deepspeed" &> /dev/null; then
  echo "DeepSpeed is not installed. Installing DeepSpeed..."
  pip install deepspeed
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Print configuration
echo "=== Training Configuration ==="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Dictionary size: $DICT_SIZE" 
echo "Latent dimension: $LATENT_DIM"
echo "Sparse dimension: $SPARSE_DIM"
echo "LISTA steps: $LISTA_STEPS"
echo "Learning rate: $LEARNING_RATE"
echo "Dictionary learning rate: $DICT_LR"
echo "Sparsity weight: $SPARSITY_WEIGHT"
echo "Adversarial weight: $ADV_WEIGHT"
echo "Random seed: $SEED"
echo "GPUs: $GPUS"
echo "Precision: $PRECISION"
echo "DeepSpeed ZeRO stage: $ZERO_STAGE"
echo "Gradient accumulation steps: $GRADIENT_ACCUMULATION"
echo "CPU offloading: ${OFFLOAD:-"disabled"}"
echo "=========================="

# Run the training script
python train.py \
  --data_dir $DATA_DIR \
  --output_dir $OUTPUT_DIR \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --dict_size $DICT_SIZE \
  --latent_dim $LATENT_DIM \
  --sparse_dim $SPARSE_DIM \
  --lista_steps $LISTA_STEPS \
  --learning_rate $LEARNING_RATE \
  --dict_lr $DICT_LR \
  --sparsity_weight $SPARSITY_WEIGHT \
  --adv_weight $ADV_WEIGHT \
  --seed $SEED \
  --gpus $GPUS \
  --precision $PRECISION \
  --zero_stage $ZERO_STAGE \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
  ${OFFLOAD:-""}
