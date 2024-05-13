from LegalAI.config.configuration import ConfigurationManager
from LegalAI.components.data_embedding import DataEmbedder
from LegalAI.logging import logger


class DataEmbeddingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config=ConfigurationManager()
        self.data_embedding_config=config.get_data_embedding_config()
        self.data_embedding=DataEmbedder(self.data_embedding_config)
