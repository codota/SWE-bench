# tabnine + swe bench eval
This evaluation is forked from https://github.com/swe-bench/SWE-bench, in attempt to provide a framework for us to test in the lab the effect of different retrieval and/or llms capabilities on code generation tasks. NOTE: the swe-bench folder is connected in git to the swe bench repo, to maintain consistency with the evaluation. make sure to push changes to the tabnine-evaluation-harness repo and not to the swe bench repo.

## how to run the evaluation
first, run the bm25_retrieval.py file to get retrieval chunks for each issue (task in the dataset).
then, run the create_text_dataset.py to create the prompt for each issue.
then, run run_api.py to make predictions with the relevant llm for the task.
lastly, run run_evaluation.py to evaluate the model generation.

See default values in each file for examples on how to run them.

## TODO
The current implementation serves as a starting point, but there are some upgrades that need to be made:
- [ ] add our chunker in the middle (currently each file is considered as a chunk)
- [ ] add additional retrieval components (i.e. reranker)

Alternatively, create some service that gets a repo and a query and returns the prompt, that will serve both the code understanding dataset and the swe bench evaluation.
