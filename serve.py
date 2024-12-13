import bentoml
from bentoml import Service
from bentoml.io import Image, JSON
import json


sports_model_ref = bentoml.pytorch.get("sports_classifier_model:latest")

sports_model_runner = sports_model_ref.to_runner()

svc = Service("sports_classifier_service", runners=[sports_model_runner])

@svc.api(input=Image(), output=JSON())
def predict(image):
    preprocess_fn = sports_model_ref.custom_objects["preprocess"]
    postprocess_fn = sports_model_ref.custom_objects["postprocess"]

    # Préprocess
    processed_input = preprocess_fn(image)  # => torch.Tensor shape [1, 3, 224, 224]

    # Appel du runner
    logits = sports_model_runner.run(processed_input)  # => shape [1, nb_classes]

    # Postprocess
    predictions = postprocess_fn(logits)

    return json.dumps(predictions)
