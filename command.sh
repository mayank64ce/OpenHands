# poetry run python openhands/core/main.py \
#     -t "write python program for adding 2 numbers" \
#     -d ./workspace/ \
#     -c CodeActAgent

# python run_fhe.py \
#     --model gpt-4o-mini \
#     --challenge-dir ../fhe_challenge/black_box/challenge_sign \
#     --max-steps 5 \
#     --log-dir logs/sign_gpt4omini

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

## GPT 5

python run_fhe.py \
    --model gpt-5-mini \
    --challenge-dir ../fhe_challenge/black_box/challenge_sign \
    --max-steps 10 \
    --log-dir logs/sign_gpt5mini

python run_fhe.py \
    --model gpt-5-mini \
    --challenge-dir ../fhe_challenge/black_box/challenge_relu \
    --max-steps 10 \
    --log-dir logs/relu_gpt5mini

python run_fhe.py \
    --model gpt-5-mini \
    --challenge-dir ../fhe_challenge/black_box/challenge_sigmoid \
    --max-steps 10 \
    --log-dir logs/sigmoid_gpt5mini