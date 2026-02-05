import argparse
import os
import sys
from pathlib import Path
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

from experiment.inference_wrapper import InferenceWrapper
# from web_socket import WebSocketInferenceServer
from experiment.eval.robot import RobotInferenceServer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory."
    )
    parser.add_argument("--port", type=int, help="Port number for the server.", default=6666)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="0.0.0.0"
    )
    parser.add_argument("--denoising_steps", type=int, help="Number of denoising steps.", default=4)
    args = parser.parse_args()

    print("Testing model checkpoint: ", args.model_path)
    print("Port: ", args.port)
    
    policy = InferenceWrapper(
        model_path=args.model_path,
        denoising_steps=args.denoising_steps,
    )

    # Start the server
    server = RobotInferenceServer(policy, port=args.port)     # based ZeroMQ（ZMQ）
    # server = WebSocketInferenceServer(policy, host=args.host, port=args.port)   # based websocket
    print(f"[READY] server listening on port {args.port}", flush=True)
    server.run()