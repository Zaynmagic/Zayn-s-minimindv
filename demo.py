"""Optional local web interface for an exported vision-language model."""

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipImageProcessor

from inference import predict


def build_demo(model, tokenizer, processor):
    import gradio as gr

    def respond(image, question, max_new_tokens):
        try:
            return predict(
                image, question, int(max_new_tokens), model, tokenizer, processor
            )
        except ValueError as error:
            raise gr.Error(str(error)) from error

    return gr.Interface(
        fn=respond,
        inputs=[
            gr.Image(type="pil", label="Image"),
            gr.Textbox(value="Describe this image.", label="Question"),
            gr.Slider(1, 256, value=64, step=1, label="Maximum output tokens"),
        ],
        outputs=gr.Textbox(label="Response"),
        title="MiniMind-V",
        flagging_mode="never",
        analytics_enabled=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the local image-question demo")
    parser.add_argument(
        "--model", required=True, help="Path to a trusted local hf/ export"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, local_files_only=True
        )
        .to(args.device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    processor = SiglipImageProcessor.from_pretrained(args.model, local_files_only=True)
    build_demo(model, tokenizer, processor).launch(
        server_name="127.0.0.1", server_port=args.port, share=False
    )
