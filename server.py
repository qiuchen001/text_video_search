import gradio as gr
import torch
import argparse
from net_helper import net_helper
from PIL import Image
# from jina_clip_embeding import clip_embeding
from clip_embeding import clip_embeding
from milvus_operator import text_video_vector
from datetime import timedelta


def video_search(text):
    if text is None:
        print("没有任何输入！")
        return None

    # clip编码
    input_embeding = clip_embeding.embeding_text(text)
    input_embeding = input_embeding[0].detach().cpu().numpy()

    print("input_embeding:", input_embeding)

    results = text_video_vector.search_data(input_embeding)

    print("results:", results)

    video_paths = [result['video_id'] for result in results]
    at_seconds = [result['at_seconds'] for result in results]
    return video_paths, at_seconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true",
                        default=False, help="share gradio app")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    app = gr.Blocks(theme='default', title="video",
                    css=".gradio-container, .gradio-container button {background-color: #009FCC} "
                        "footer {visibility: hidden}")

    with app:
        with gr.Tabs():
            with gr.TabItem("video search"):
                with gr.Row():
                    with gr.Column():
                        text = gr.TextArea(label="Text",
                                           placeholder="description",
                                           value="", )
                        btn = gr.Button(value="search")

                    with gr.Column():
                        # Use a list to store video components and their corresponding time textboxes
                        video_time_components = []
                        for _ in range(5):
                            with gr.Column():  # Each video and its time in a separate column
                                video_comp = gr.components.Video(label=None)
                                time_comp = gr.components.Textbox(label="At Time")
                                video_time_components.append(video_comp)
                                video_time_components.append(time_comp)


                def set_video_time(video_paths, at_seconds):
                    updates = []
                    for i in range(len(video_paths)):
                        updates.append(gr.update(value=video_paths[i]))  # Update video component
                        updates.append(gr.update(value=str(timedelta(seconds=at_seconds[i]))))  # Update time textbox
                    return updates


                btn.click(
                    lambda x: set_video_time(*video_search(x)),
                    inputs=[text],
                    outputs=video_time_components,  # Pass the combined list of components
                    show_progress=True
                )

    ip_addr = net_helper.get_host_ip()
    app.queue().launch(show_api=False, share=True, server_name=ip_addr, server_port=9099)