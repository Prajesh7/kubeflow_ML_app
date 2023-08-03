import os
from absl import logging
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from pipeline import create_pipeline

PIPELINE_NAME = 'sail_prediction'
PIPELINE_ROOT = 'metadata'
DATA_PATH = 'data'
SERVING_DIR = 'models'

def run():
    metadata_config = kubeflow_dag_runner.get_default_metadata_config()
    tfx_image = '67a4b1138d2d' #image URI
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubefflow_metadata_config=metadata_config,
        tfx_image=tfx_image
    )
    
    kubeflow_dag_runner.KubeflowDagRunner(config = runner_config).run(
        create_pipeline(
            pipeline_name = PIPELINE_NAME,
            pipeline_root = PIPELINE_ROOT,
            serving_dir = SERVING_DIR,
            data_path = DATA_PATH
        )
    )
    

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()