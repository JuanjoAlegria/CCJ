.PHONY: clean data lint requirements

#################################################################################
# GLOBALS                                                                       #
#################################################################################

project_dir := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
project_name = anillo_ccj
python_interpreter = python3


#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(python_interpreter) -m pip install -U pip setuptools wheel
	$(python_interpreter) -m pip install -r requirements.txt

## Make Dataset
data: requirements
	$(python_interpreter) src/data/make_dataset.py data/raw data/processed

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using pylint
lint:
	pylint src

## Test python environment is setup correctly
test_environment:
	$(python_interpreter) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

folders_names = first NegsiRNA Test

get_features:
	for folder_name in $(folders_names); do \
		$(python_interpreter) -m src.scripts.get_features \
			--imgs_dir data/$${folder_name} \
			--save_path results/$${folder_name}_results.csv ; \
	done;



clear_data:
	rm -r data/$(cancer_type)/raw/annotations ||:
	rm -r data/$(cancer_type)/interim ||:
	rm -r data/$(cancer_type)/annotations_details.csv ||:
	rm -r data/$(cancer_type)/annotations_summary.csv ||:
	mkdir data/$(cancer_type)/interim ||:

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=50 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
