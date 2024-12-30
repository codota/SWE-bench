__version__ = "2.1.7"

from swe_bench.swebench.collect.build_dataset import main as build_dataset
from swe_bench.swebench.collect.get_tasks_pipeline import main as get_tasks_pipeline
from swe_bench.swebench.collect.print_pulls import main as print_pulls

from swe_bench.swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    MAP_REPO_VERSION_TO_SPECS,
)

from swe_bench.swebench.harness.docker_build import (
    build_image,
    build_base_images,
    build_env_images,
    build_instance_images,
    build_instance_image,
    close_logger,
    setup_logger,
)

from swe_bench.swebench.harness.docker_utils import (
    cleanup_container,
    remove_image,
    copy_to_container,
    exec_run_with_timeout,
    list_images,
)

from swe_bench.swebench.harness.grading import (
    compute_fail_to_pass,
    compute_pass_to_pass,
    get_logs_eval,
    get_eval_report,
    get_resolution_status,
    ResolvedStatus,
    TestStatus,
)

from swe_bench.swebench.harness.log_parsers import (
    MAP_REPO_TO_PARSER,
)

from swe_bench.swebench.harness.run_evaluation import (
    main as run_evaluation,
)

from swe_bench.swebench.harness.utils import (
    get_environment_yml,
    get_requirements,
)

from swe_bench.swebench.versioning.constants import (
    MAP_REPO_TO_VERSION_PATHS,
    MAP_REPO_TO_VERSION_PATTERNS,
)

from swe_bench.swebench.versioning.get_versions import (
    get_version,
    map_version_to_task_instances,
    get_versions_from_build,
    get_versions_from_web,
)

from swe_bench.swebench.versioning.utils import (
    split_instances,
)