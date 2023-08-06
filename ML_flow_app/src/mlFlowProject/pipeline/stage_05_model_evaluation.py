from mlFlowProject.config.configuration import ConfigurationManager
from mlFlowProject.components.model_evaluation import ModelEvaluation
from mlFlowProject import logger

STAGE_NAME = "model evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config_manager = ConfigurationManager()
            model_eval_config = config_manager.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config = model_eval_config)
            model_evaluation.log_into_mlflow()
        except Exception as e:
            raise e
        
        
if __name__ == "__main__":
    try:
        logger.info(f">>>> stage {STAGE_NAME} started <<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx========x")
    except Exception as e:
        logger.exception(e)
        raise e