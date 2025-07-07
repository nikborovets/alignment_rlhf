.PHONY: all sft rm reinforce eval clean

all: sft rm reinforce eval

sft:
	@echo "--- Downloading SFT model ---"
	python scripts/save_sft_model.py

rm:
	@echo "--- Training Reward Model ---"
	PYTHONPATH=$(CURDIR) python scripts/train_rm_manual.py

reinforce:
	@echo "--- Training with REINFORCE ---"
	PYTHONPATH=$(CURDIR) python scripts/train_reinforce.py

eval:
	@echo "--- Evaluating models ---"
	PYTHONPATH=$(CURDIR) python scripts/eval.py

clean:
	@echo "--- Cleaning up models and reports ---"
	rm -rf models/*
	rm -rf reports/*.csv
	@echo "Done."
