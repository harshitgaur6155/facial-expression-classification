from facialExpressionClassify import logger
from facialExpressionClassify.config.configuration import ConfigurationManager
from facialExpressionClassify.components.prepare_base_model import PrepareBaseModel, CustomCNN



STAGE_NAME = "Base Model Preparation Stage"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()

        # Calls Pre-Trained Model class
        # prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)

        # Calls Custom Built Model class
        prepare_base_model = CustomCNN(config=prepare_base_model_config)
        prepare_base_model.build_custom_model()




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
