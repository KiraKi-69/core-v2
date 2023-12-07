from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.codecs import PandasCodec
from mlserver.errors import MLServerError
import pandas as pd
from fastapi import status
from mlserver.logging import logger


# https://github.com/SeldonIO/seldon-core/blob/v2/samples/examples/pandasquery/pandasquery/model.py

class ModelParametersMissing(MLServerError):
    def __init__(self, model_name: str, reason: str):
        super().__init__(f"Parameters missing for model {model_name} {reason}", status.HTTP_400_BAD_REQUEST)

class PandasReader(MLModel):
    
    async def load(self) -> bool:
        logger.info("Loading model")
        self.ready = True
        return self.ready
    
    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        input_df: pd.DataFrame = PandasCodec.decode_request(payload)
        # run query on input_df and save in output_df
        output_df = input_df
        
        if output_df.empty:
            logger.info("it is not work")
            output_df = pd.DataFrame({'status':["no rows satisfied "]})
        else:
            logger.info("it is WORK")
            output_df["status"] = "row satisfied "
        
        return PandasCodec.encode_response(self.name, output_df, self.version)