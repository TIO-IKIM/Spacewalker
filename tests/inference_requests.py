import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "SigLIP2_image"

with httpclient.InferenceServerClient("0.0.0.0:8000") as client:
    input0_data = np.random.rand(3, 224, 224).astype(np.uint8)
    inputs = [
        httpclient.InferInput(
            "input", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(input0_data)

    outputs = [
        httpclient.InferRequestedOutput("output"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    # async
    # response = client.async_infer(model_name, inputs, request_id=str(1), outputs=outputs).get_result()

    output0_data = response.as_numpy("output")

    print(output0_data.squeeze().shape)