# poetry run python openhands/core/main.py \
#     -t "write python program for adding 2 numbers" \
#     -d ./workspace/ \
#     -c CodeActAgent

python run_fhe.py \
    --model gpt-4o \
    --challenge-dir ../fhe_challenge/black_box/challenge_relu \
    --max-steps 10 \
    --log-dir logs/test_relu