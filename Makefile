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

branch_thickness_voronoi:
	$(python_interpreter) -m src.scripts.image_processing.compute_branch_thickness \
		--imgs_dir data/raw/images \
		--results_base_dir data/interim \
		--algorithm voronoi

branch_thickness_medial_axis:
	$(python_interpreter) -m src.scripts.image_processing.compute_branch_thickness \
		--imgs_dir data/raw/images \
		--results_base_dir data/interim \
		--algorithm medial_axis

texture_entropy_filter:
	$(python_interpreter) -m src.scripts.image_processing.compute_texture \
		--imgs_dir data/raw/images \
		--results_base_dir data/interim \
		--algorithm entropy

texture_std_filter:
	$(python_interpreter) -m src.scripts.image_processing.compute_texture \
		--imgs_dir data/raw/images \
		--results_base_dir data/interim \
		--algorithm std

skeleton_data:
	$(python_interpreter) -m src.scripts.image_processing.compute_skeleton_data \
		--imgs_dir data/raw/images \
		--results_base_dir data/interim

centroids_moments:
	$(python_interpreter) -m src.scripts.image_processing.compute_centroids_moments \
		--imgs_dir data/raw/images \
		--results_base_dir data/interim

blobs_data:\
branch_thickness_voronoi \
branch_thickness_medial_axis \
texture_entropy_filter \
texture_std_filter \
skeleton_data \
centroids_moments \
blobs_data
	$(python_interpreter) -m src.scripts.image_processing.compute_blobs_data \
		--imgs_dir data/raw/images \
		--results_base_dir data/interim

get_features:
	$(python_interpreter) -m src.scripts.get_features \
		--raw_dir data/raw \
		--interim_dir data/interim \
		--save_dir data/processed

clear_data:
	rm -r data/interim ||:
	mkdir data/interim ||:

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
