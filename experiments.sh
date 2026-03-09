#!/bin/bash
set -x

# ============================================================
# SECTION 1: gpt-4o-mini
# ============================================================

# --- black_box ---

# python run_fhe.py \
#     --model gpt-4o-mini \
#     --challenge-dir ../fhe_challenge/black_box/challenge_relu \
#     --max-steps 10 \
#     --log-dir logs/relu_gpt4omini

# python run_fhe.py \
#     --model gpt-4o-mini \
#     --challenge-dir ../fhe_challenge/black_box/challenge_sigmoid \
#     --max-steps 10 \
#     --log-dir logs/sigmoid_gpt4omini

# python run_fhe.py \
#     --model gpt-4o-mini \
#     --challenge-dir ../fhe_challenge/black_box/challenge_sign \
#     --max-steps 10 \
#     --log-dir logs/sign_gpt4omini

# --- white_box/ml_inference ---

# python run_fhe.py \
#     --model gpt-4o-mini \
#     --challenge-dir ../fhe_challenge/white_box/ml_inference/challenge_cifar10 \
#     --max-steps 10 \
#     --log-dir logs/cifar10_gpt4omini

# python run_fhe.py \
#     --model gpt-4o-mini \
#     --challenge-dir ../fhe_challenge/white_box/ml_inference/challenge_sentiment_analysis \
#     --max-steps 10 \
#     --log-dir logs/sentiment_analysis_gpt4omini

# python run_fhe.py \
#     --model gpt-4o-mini \
#     --challenge-dir ../fhe_challenge/white_box/ml_inference/challenge_house_prediction \
#     --max-steps 10 \
#     --log-dir logs/house_prediction_gpt4omini

# python run_fhe.py \
#     --model gpt-4o-mini \
#     --challenge-dir ../fhe_challenge/white_box/ml_inference/challenge_svm \
#     --max-steps 10 \
#     --log-dir logs/svm_gpt4omini

# # --- white_box/openfhe ---

python run_fhe.py \
    --model gpt-4o-mini \
    --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_gelu \
    --max-steps 10 \
    --log-dir logs/gelu_gpt4omini

python run_fhe.py \
    --model gpt-4o-mini \
    --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_softmax \
    --max-steps 10 \
    --log-dir logs/softmax_gpt4omini

python run_fhe.py \
    --model gpt-4o-mini \
    --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_shl \
    --max-steps 10 \
    --log-dir logs/shl_gpt4omini

python run_fhe.py \
    --model gpt-4o-mini \
    --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_array_sorting \
    --max-steps 10 \
    --log-dir logs/array_sorting_gpt4omini

python run_fhe.py \
    --model gpt-4o-mini \
    --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_lookup_table \
    --max-steps 10 \
    --log-dir logs/lookup_table_gpt4omini

python run_fhe.py \
    --model gpt-4o-mini \
    --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_matrix_multiplication \
    --max-steps 10 \
    --log-dir logs/matrix_multiplication_gpt4omini

python run_fhe.py \
    --model gpt-4o-mini \
    --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_max \
    --max-steps 10 \
    --log-dir logs/max_gpt4omini

python run_fhe.py \
    --model gpt-4o-mini \
    --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_knn \
    --max-steps 10 \
    --log-dir logs/knn_gpt4omini

python run_fhe.py \
    --model gpt-4o-mini \
    --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_parity \
    --max-steps 10 \
    --log-dir logs/parity_gpt4omini

python run_fhe.py \
    --model gpt-4o-mini \
    --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_invertible_matrix \
    --max-steps 10 \
    --log-dir logs/invertible_matrix_gpt4omini

python run_fhe.py \
    --model gpt-4o-mini \
    --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_svd \
    --max-steps 10 \
    --log-dir logs/svd_gpt4omini


# ============================================================
# SECTION 2: gpt-5-mini-2025-08-07
# ============================================================

# --- black_box ---

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/black_box/challenge_relu \
#     --max-steps 10 \
#     --log-dir logs/relu_gpt5mini

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/black_box/challenge_sigmoid \
#     --max-steps 10 \
#     --log-dir logs/sigmoid_gpt5mini

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/black_box/challenge_sign \
#     --max-steps 10 \
#     --log-dir logs/sign_gpt5mini

# # --- white_box/ml_inference ---

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/white_box/ml_inference/challenge_cifar10 \
#     --max-steps 10 \
#     --log-dir logs/cifar10_gpt5mini

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/white_box/ml_inference/challenge_sentiment_analysis \
#     --max-steps 10 \
#     --log-dir logs/sentiment_analysis_gpt5mini

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/white_box/ml_inference/challenge_house_prediction \
#     --max-steps 10 \
#     --log-dir logs/house_prediction_gpt5mini

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/white_box/ml_inference/challenge_svm \
#     --max-steps 10 \
#     --log-dir logs/svm_gpt5mini

# # --- white_box/openfhe ---

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_gelu \
#     --max-steps 10 \
#     --log-dir logs/gelu_gpt5mini

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_softmax \
#     --max-steps 10 \
#     --log-dir logs/softmax_gpt5mini

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_shl \
#     --max-steps 10 \
#     --log-dir logs/shl_gpt5mini

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_array_sorting \
#     --max-steps 10 \
#     --log-dir logs/array_sorting_gpt5mini

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_lookup_table \
#     --max-steps 10 \
#     --log-dir logs/lookup_table_gpt5mini

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_matrix_multiplication \
#     --max-steps 10 \
#     --log-dir logs/matrix_multiplication_gpt5mini

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_max \
#     --max-steps 10 \
#     --log-dir logs/max_gpt5mini

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_knn \
#     --max-steps 10 \
#     --log-dir logs/knn_gpt5mini

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_parity \
#     --max-steps 10 \
#     --log-dir logs/parity_gpt5mini

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_invertible_matrix \
#     --max-steps 10 \
#     --log-dir logs/invertible_matrix_gpt5mini

# python run_fhe.py \
#     --model gpt-5-mini-2025-08-07 \
#     --challenge-dir ../fhe_challenge/white_box/openfhe/challenge_svd \
#     --max-steps 10 \
#     --log-dir logs/svd_gpt5mini
