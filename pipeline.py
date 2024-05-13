from LegalAI.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from LegalAI.pipeline.stage_02_data_embedding import DataEmbeddingPipeline
from LegalAI.logging import logger
STAGE='stage_02 Data embedding'

try:
    print(f'---------  {STAGE}  started ---------')
    data_embedding_pipeline = DataEmbeddingPipeline()
    data_embedding_pipeline.main()
    print(f'---------  {STAGE} completed  ---------')
except Exception as e:
    print(f'---------  {STAGE} failed  ---------')
    logger.exception(e)
    


